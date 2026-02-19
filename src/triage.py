"""
triage.py

Minimal, deterministic incident triage (no AI).

Purpose:
- Convert unstructured incident text into a structured JSON-like dict
- Provide a stable interface for later RAG + ML integration
"""

from __future__ import annotations

from curses import raw
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from src.retrieve import retrieve_runbooks
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
    return [k for k in keywords if k in t]

def _is_negated(text: str, keyword: str) -> bool:
    """
    Very small deterministic negation check for common patterns like:
    'no outage', 'no outages', 'not down', 'without outage'
    """
    t = text.lower()
    patterns = [
        f"no {keyword}",
        f"no {keyword}s",
        f"not {keyword}",
        f"without {keyword}",
        f"no {keyword} reported",
        f"no {keyword}s reported",
    ]
    return any(p in t for p in patterns)

CATEGORY_RULES: List[Tuple[Category, List[str], List[str]]] = [
    # (category, keywords, suspected_systems)
    (Category.IT_OPS, ["login", "auth", "authentication", "sso", "password", "token", "vpn", "dns"], ["Authentication"]),
    (Category.CUSTOMER_SUPPORT, ["payment", "checkout", "refund", "charge", "billing", "invoice"], ["Payments/Billing"]),
    (Category.OPERATIONS, ["shipment", "delivery", "warehouse", "route", "fleet", "dispatch"], ["Logistics"]),
    (Category.ENGINEERING, ["build failed", "ci", "pipeline", "deploy", "release", "bug", "rollback"], ["CI/CD"]),
]

URGENCY_HIGH = ["outage", "down", "unable", "cannot", "can't", "sev1", "critical", "p0", "blocker"]
URGENCY_MED = ["slow", "intermittent", "sometimes", "degraded", "latency", "flaky"]

PERF_KEYWORDS = [
    "slow", "slowness", "latency", "high latency", "degraded", "degradation",
    "timeout", "timed out", "time out",
    "load time", "page load", "pages load", "loading", "seconds",
    "response time", "5 seconds", "10 seconds",
    "db latency", "database latency", "query", "queries", "slow query",
    "cpu", "memory", "high load", "server load", "spike",
    "500", "502", "503", "504", "gateway timeout"
]

def _detect_performance(raw: str) -> List[str]:
    """
    Detect performance degradation indicators.
    Returns list of matched indicators (may include simple numeric patterns).
    """
    t = raw.lower()
    hits = []

    # Keyword hits
    hits.extend([k for k in PERF_KEYWORDS if k in t])

    # Simple pattern: "X seconds" (captures "5 seconds", "10 seconds", etc.)
    # Keep deterministic / lightweight: just check a few common forms
    for n in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "30"]:
        if f"{n} second" in t or f"{n}s" in t:
            hits.append(f"{n}s")

    # De-dup while preserving order
    seen = set()
    ordered = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered


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
    reasoning: List[str] = []

    # Only count HIGH matches that are not negated
    raw_low = raw.lower()
    matched_high = [k for k in URGENCY_HIGH if (k in raw_low and not _is_negated(raw, k))]
    matched_med = [k for k in URGENCY_MED if (k in raw_low and not _is_negated(raw, k))]

    if matched_high:
        reasoning.append(f"Urgency set to High due to indicators: {matched_high}.")
        return Urgency.HIGH, matched_high, 0.80, reasoning

    if matched_med:
        reasoning.append(f"Urgency set to Medium due to indicators: {matched_med}.")
        return Urgency.MEDIUM, matched_med, 0.65, reasoning

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
    
    # --- Priority override: performance degradation should route to Engineering ---
    perf_hits = _detect_performance(raw)

    # Require perf evidence anchored by strong indicators to avoid false positives
    anchors = {"latency", "degraded", "timeout", "timed out", "db latency", "server load", "503", "504", "gateway timeout"}
    has_anchor = any(h in anchors for h in perf_hits)

    strong_perf = any(h in perf_hits for h in ["latency", "timeout", "timed out", "degraded", "db latency", "504", "503"])
    has_payments_words = bool(_contains_any(raw, ["checkout", "payment", "refund", "billing"]))

    # Override when we have multiple perf signals AND at least one anchor,
    # or when a strong perf signal appears alongside payments words (classic "slow checkout" scenario).
    if (len(perf_hits) >= 2 and has_anchor) or (strong_perf and has_payments_words):
        # Override only if we were about to route to Customer Support or General Ops
        if category in (Category.CUSTOMER_SUPPORT, Category.GENERAL_OPS):
            category = Category.ENGINEERING
            matched_keywords["performance"] = perf_hits
            reasoning.append(f"Performance degradation indicators found ({perf_hits}); overriding category to Engineering.")

            suspected = ["Performance"] + (["CI/CD"] if _contains_any(raw, ["deploy", "release", "rollback", "ci", "pipeline"]) else [])

            # Lift confidence floor a bit since perf evidence is strong
            cat_conf = max(cat_conf, 0.70)

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
        recommended_runbooks = ["RBK-IT-AUTH-001"]
    elif category == Category.CUSTOMER_SUPPORT:
        recommended_runbooks = ["RBK-CS-PAYMENTS-010"]
    elif category == Category.ENGINEERING:
        # If performance signals were detected, prefer performance runbook
        if "performance" in matched_keywords:
            recommended_runbooks = ["RBK-ENG-PERF-120"]
        else:
            recommended_runbooks = ["RBK-ENG-CICD-101"]
    elif category == Category.OPERATIONS:
        # Distinguish inventory sync issues vs warehouse system outages
        wms_indicators = ["label", "printer", "print", "wms", "warehouse system", "packing", "shipping"]
        if _contains_any(raw, wms_indicators):
            recommended_runbooks = ["RBK-OPS-WMS-060"]
        else:
            recommended_runbooks = ["RBK-OPS-LOGISTICS-050"]

        
    citations = retrieve_runbooks(recommended_runbooks)

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
        citations = citations,
    )

    return asdict(ticket)

