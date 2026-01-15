"""
triage.py

Minimal, deterministic incident triage (no AI).

Purpose:
- Convert unstructured incident text into a structured JSON-like dict
- Provide a stable interface for later RAG + ML integration
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from enum import Enum


class Category(str, Enum):
    IT_OPS = "IT Ops"
    CUSTOMER_SUPPORT = "Customer Support"
    OPERATIONS = "Operations"
    ENGINEERING = "Engineering"
    GENERAL_OPS = "General Ops"

class Urgency(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


@dataclass
class IncidentTicket:
    ticket_id: str
    created_at_utc: str
    title: str
    description: str
    
    category: str
    urgency: str
    impact: str
    
    suspected_systems: List[str]
    
    #Hallucination prevention
    matched_keywords: Dict[str, List[str]]
    reasoning: List[str]
    confidence: float
    
    #Human in the loop
    needs_human_review: bool
    missing_info_questions: List[str]
    
    #Action
    next_actions: List[str]
    
    #RAG hooks (EMPTY, add in later)
    recommended_runbooks: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)  # {"doc_id": "...", "chunk_id":"...", "quote":"..."}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simple_ticket_id(text: str) -> str:
    # Deterministic-ish ID for demo purposes (not production)
    base = abs(hash(text)) % 10**8
    return f"INC-{base:08d}"

def _normalize(text: str) -> str:
    return (text or "").strip()

def _contains_any(text: str, keywords: List[str]) -> List[str]:
    t = text.lower()
    return any(k for k in keywords if k in t)

CATEGORY_RULES: List[Tuple[Category, List[str], List[str]]] = [
    # (category, keywords, suspected_systems)
    (Category.IT_OPS, ["login", "auth", "authentication", "sso", "password", "token", "vpn", "dns"], ["Authentication"]),
    (Category.CUSTOMER_SUPPORT, ["payment", "checkout", "refund", "charge", "billing", "invoice"], ["Payments/Billing"]),
    (Category.OPERATIONS, ["shipment", "delivery", "warehouse", "route", "fleet", "dispatch"], ["Logistics"]),
    (Category.ENGINEERING, ["build failed", "ci", "pipeline", "deploy", "release", "bug", "rollback"], ["CI/CD"]),
]

URGENCY_HIGH = ["outage", "down", "unable", "cannot", "can't", "sev1", "critical", "p0", "blocker"]
URGENCY_MED = ["slow", "intermittent", "sometimes", "degraded", "latency", "flaky"]

IMPACT_BROAD = ["multiple teams", "all users", "everyone", "company-wide", "entire org"]
IMPACT_CUSTOMER = ["customer", "customers", "clients", "buyers", "users affected"]


# Actions are centralized (less spaghetti)
ACTION_PLAYBOOK: Dict[Category, List[str]] = {
    Category.IT_OPS: [
        "Check service health dashboards and recent changes/deploys",
        "Collect an error message/code and a timestamp of a failing attempt",
        "Identify affected scope (which users/teams, which region, which environment)",
    ],
    Category.CUSTOMER_SUPPORT: [
        "Confirm scope (which customers, region, account tier) and collect examples",
        "Collect IDs (order/transaction/customer) and timestamps for failures",
        "Check third-party provider status pages if applicable",
    ],
    Category.OPERATIONS: [
        "Confirm affected locations/routes and time window",
        "Collect relevant IDs (shipment/order/vehicle) and current status",
        "Check upstream dependencies (vendors, inventory, dispatch systems)",
    ],
    Category.ENGINEERING: [
        "Identify failing step and capture logs/error output",
        "Check recent changes (PRs, releases) and rollback options",
        "Confirm environment (prod/stage), version, and reproduction steps",
    ],
    Category.GENERAL_OPS: [
        "Clarify the goal and success criteria",
        "Identify owner/team responsible",
        "Collect relevant IDs, timestamps, and any error details",
    ],
}


def _classify_category(raw: str) -> Tuple[Category, List[str], List[str], float, List[str]]:
    """
    Returns: (category, matched_keywords, suspected_systems, confidence, reasoning_lines)
    Conservative: if weak evidence, return GENERAL_OPS with lower confidence.
    """
    best_category = Category.GENERAL_OPS
    best_matches: List[str] = []
    best_suspected: List[str] = []
    reasoning: List[str] = []

    for cat, keywords, suspected in CATEGORY_RULES:
        matches = _contains_any(raw, keywords)
        if len(matches) > len(best_matches):
            best_category = cat
            best_matches = matches
            best_suspected = suspected

    # Simple confidence heuristic: more matches => higher confidence
    if len(best_matches) >= 3:
        conf = 0.85
    elif len(best_matches) == 2:
        conf = 0.70
    elif len(best_matches) == 1:
        conf = 0.55
    else:
        conf = 0.40  # general/unknown

    if best_category == Category.GENERAL_OPS:
        reasoning.append("No strong category keywords matched; defaulted to General Ops.")
    else:
        reasoning.append(f"Category inferred from keywords: {best_matches}.")

    return best_category, best_matches, best_suspected, conf, reasoning


def _classify_urgency(raw: str) -> Tuple[Urgency, List[str], float, List[str]]:
    matched_high = _contains_any(raw, URGENCY_HIGH)
    matched_med = _contains_any(raw, URGENCY_MED)

    reasoning: List[str] = []

    # Prefer high if any high indicators
    if matched_high:
        reasoning.append(f"Urgency set to High due to indicators: {matched_high}.")
        return Urgency.HIGH, matched_high, 0.80, reasoning

    if matched_med:
        reasoning.append(f"Urgency set to Medium due to indicators: {matched_med}.")
        return Urgency.MEDIUM, matched_med, 0.65, reasoning

    # If no indicators, be conservative but not alarmist
    reasoning.append("No urgency indicators found; set to Low.")
    return Urgency.LOW, [], 0.55, reasoning


def _infer_impact(raw: str) -> Tuple[str, List[str], List[str]]:
    broad = _contains_any(raw, IMPACT_BROAD)
    customer = _contains_any(raw, IMPACT_CUSTOMER)

    if broad:
        return "Broad impact (many users/teams)", broad, ["Impact inferred as broad due to keywords."]
    if customer:
        return "Customer-facing impact", customer, ["Impact inferred as customer-facing due to keywords."]
    return "Unknown/unclear impact", [], ["Impact not clearly specified; left as unknown."]


def _missing_info_questions(raw: str) -> List[str]:
    questions = []

    # Time window / start time
    if not _contains_any(raw, ["started", "since", "minutes", "hours", "today", "yesterday", "timestamp", "am", "pm"]):
        questions.append("When did this start (approx. time and timezone)?")

    # Error details
    if not _contains_any(raw, ["error", "message", "code", "screenshot", "log", "stacktrace", "trace"]):
        questions.append("Do you have an error message, code, or log snippet?")

    # Scope
    if not _contains_any(raw, ["affects", "impact", "users", "teams", "customers", "everyone", "all users"]):
        questions.append("Who is affected (team/customers/how many users)?")

    # Environment (often missing and critical)
    if not _contains_any(raw, ["prod", "production", "staging", "dev", "test environment"]):
        questions.append("Which environment is affected (prod/staging/dev)?")

    return questions

def triage_incident(text: str) -> Dict[str, Any]:
    """
    Convert raw incident text into a structured ticket.
    Deterministic + conservative by design.
    """
    raw = _normalize(text)
    if not raw:
        raise ValueError("incident text is required")

    title = raw.splitlines()[0][:80] if raw else "Untitled incident"

    matched_keywords: Dict[str, List[str]] = {}
    reasoning: List[str] = []

    category, cat_matches, suspected, cat_conf, cat_reason = _classify_category(raw)
    matched_keywords["category"] = cat_matches
    reasoning.extend(cat_reason)

    urgency, urg_matches, urg_conf, urg_reason = _classify_urgency(raw)
    matched_keywords["urgency"] = urg_matches
    reasoning.extend(urg_reason)

    impact, impact_matches, impact_reason = _infer_impact(raw)
    if impact_matches:
        matched_keywords["impact"] = impact_matches
    reasoning.extend(impact_reason)

    questions = _missing_info_questions(raw)

    # Combine confidence signals (simple weighted average)
    confidence = round((0.55 * cat_conf) + (0.35 * urg_conf) + (0.10 * (0.75 if impact != "Unknown/unclear impact" else 0.45)), 2)

    # Human review
    needs_review = False
    if category == Category.GENERAL_OPS and confidence < 0.55:
        needs_review = True
        reasoning.append("Low confidence category; recommend human review.")
    if urgency == Urgency.HIGH and "error" not in raw.lower() and "log" not in raw.lower():
        needs_review = True
        reasoning.append("High urgency without supporting error/log details; recommend human review.")
    if len(questions) >= 3:
        needs_review = True
        reasoning.append("Multiple missing critical fields; recommend collecting info before actioning.")

    next_actions = ACTION_PLAYBOOK.get(category, ACTION_PLAYBOOK[Category.GENERAL_OPS]).copy()

    # RAG hooks: suggest runbook types (IDs/names), but don't invent content
    recommended_runbooks = []
    if category == Category.IT_OPS and "Authentication" in suspected:
        recommended_runbooks = ["RBK-IT-AUTH-001", "RBK-IT-SSO-002"]
    elif category == Category.CUSTOMER_SUPPORT:
        recommended_runbooks = ["RBK-CS-PAYMENTS-010"]
    elif category == Category.ENGINEERING:
        recommended_runbooks = ["RBK-ENG-CICD-101"]
    elif category == Category.OPERATIONS:
        recommended_runbooks = ["RBK-OPS-LOGISTICS-050"]

    ticket = IncidentTicket(
        ticket_id=_simple_ticket_id(raw),
        created_at_utc=_utc_now_iso(),
        title=title,
        description=raw,
        category=category,
        urgency=urgency,
        impact=impact,
        suspected_systems=suspected,

        matched_keywords=matched_keywords,
        reasoning=reasoning,
        confidence=confidence,

        needs_human_review=needs_review,
        missing_info_questions=questions,
        next_actions=next_actions,

        recommended_runbooks=recommended_runbooks,
        citations=[],
    )

    return asdict(ticket)

