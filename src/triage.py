"""
triage.py

Minimal, deterministic incident triage (no AI).

Purpose:
- Convert unstructured incident text into a structured JSON-like dict
- Provide a stable interface for later RAG + ML integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Tuple

from src.retrieve import retrieve_runbooks


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

    # Hallucination prevention
    matched_keywords: Dict[str, List[str]]
    reasoning: List[str]
    confidence: float

    # Human in the loop
    needs_human_review: bool
    missing_info_questions: List[str]

    # Action
    next_actions: List[str]

    # RAG hooks
    recommended_runbooks: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)  # {"doc_id": "...", "path":"...", "excerpt":"..."}


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
    Small deterministic negation check for common patterns like:
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


def _extract_http_statuses(raw: str) -> List[str]:
    """
    Extract HTTP status codes like 401/403/404/500/502/503/504 from text.
    """
    if not raw:
        return []
    hits = re.findall(r"\b(4\d{2}|5\d{2})\b", raw)
    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _extract_environment(raw: str) -> str:
    """
    Best-effort deterministic env detection from common tokens.
    Returns: "prod" | "staging" | "dev" | "".
    """
    t = raw.lower()
    if any(x in t for x in ["prod", "production", "live environment"]):
        return "prod"
    if any(x in t for x in ["staging", "stage", "uat", "preprod", "pre-prod"]):
        return "staging"
    if any(x in t for x in ["dev", "development", "test environment", "qa"]):
        return "dev"
    return ""


def _has_time_hint(raw: str) -> bool:
    t = raw.lower()
    return bool(
        _contains_any(t, ["started", "since", "minutes", "hours", "today", "yesterday", "timestamp", "am", "pm"])
        or re.search(r"\b\d{1,2}:\d{2}\b", t)  # 13:45
        or re.search(r"\b\d{1,2}\s*(am|pm)\b", t)  # 9am
    )


CATEGORY_RULES: List[Tuple[Category, List[str], List[str]]] = [
    # (category, keywords, suspected_systems)
    (
        Category.IT_OPS,
        [
            "login", "auth", "authentication", "sso", "password", "token", "vpn", "dns",
            "403", "401", "forbidden", "unauthorized", "permission", "permissions", "role", "rbac",
            "access denied", "iam", "idp", "certificate", "cert", "clock sync"
        ],
        ["Authentication"],
    ),
    (
        Category.CUSTOMER_SUPPORT,
        ["payment", "checkout", "refund", "charge", "billing", "invoice", "card", "bank", "processor"],
        ["Payments/Billing"],
    ),
    (
        Category.OPERATIONS,
        ["shipment", "delivery", "warehouse", "route", "fleet", "dispatch", "label", "printer", "packing", "shipping"],
        ["Logistics"],
    ),
    (
        Category.ENGINEERING,
        ["build failed", "ci", "pipeline", "deploy", "release", "bug", "rollback", "crash", "exception", "stacktrace"],
        ["CI/CD"],
    ),
]

URGENCY_HIGH = ["outage", "down", "unable", "cannot", "can't", "sev1", "critical", "p0", "blocker"]
URGENCY_MED = ["slow", "intermittent", "sometimes", "degraded", "latency", "flaky", "delay", "delayed"]

PERF_KEYWORDS = [
    "slow", "slowness", "latency", "high latency", "degraded", "degradation",
    "timeout", "timed out", "time out",
    "load time", "page load", "pages load", "loading", "seconds",
    "response time", "db latency", "database latency", "query", "queries", "slow query",
    "cpu", "memory", "high load", "server load", "spike",
    "500", "502", "503", "504", "gateway timeout",
]

IMPACT_BROAD = ["multiple teams", "all users", "everyone", "company-wide", "entire org", "site-wide", "system-wide"]
IMPACT_CUSTOMER = ["customer", "customers", "clients", "buyers", "users affected", "checkout", "orders"]


# Actions are centralized
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


def _detect_performance(raw: str) -> List[str]:
    """
    Detect performance degradation indicators.
    Returns list of matched indicators (may include simple numeric patterns).
    """
    t = raw.lower()
    hits: List[str] = []

    hits.extend([k for k in PERF_KEYWORDS if k in t])

    # "X seconds" / "Xs"
    for n in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "30"]:
        if f"{n} second" in t or f"{n}s" in t:
            hits.append(f"{n}s")

    # de-dupe preserve order
    seen = set()
    ordered: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered


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

    # Slightly better confidence model:
    # - base from keyword count
    # - boost if we extracted strong structured signals like HTTP codes
    http_codes = _extract_http_statuses(raw)
    env = _extract_environment(raw)

    if len(best_matches) >= 4:
        conf = 0.88
    elif len(best_matches) == 3:
        conf = 0.82
    elif len(best_matches) == 2:
        conf = 0.70
    elif len(best_matches) == 1:
        conf = 0.55
    else:
        conf = 0.40

    if http_codes:
        conf = min(0.90, conf + 0.05)
    if env:
        conf = min(0.90, conf + 0.03)

    if best_category == Category.GENERAL_OPS:
        reasoning.append("No strong category keywords matched; defaulted to General Ops.")
    else:
        reasoning.append(f"Category inferred from keywords: {best_matches}.")

    return best_category, best_matches, best_suspected, conf, reasoning


def _classify_urgency(raw: str) -> Tuple[Urgency, List[str], float, List[str]]:
    reasoning: List[str] = []

    raw_low = raw.lower()
    matched_high = [k for k in URGENCY_HIGH if (k in raw_low and not _is_negated(raw, k))]
    matched_med = [k for k in URGENCY_MED if (k in raw_low and not _is_negated(raw, k))]

    # If explicit 5xx present, treat as at least Medium, sometimes High
    http_codes = _extract_http_statuses(raw)
    has_5xx = any(c.startswith("5") for c in http_codes)

    if matched_high or (has_5xx and "outage" in raw_low and not _is_negated(raw, "outage")):
        hits = matched_high[:] or ["5xx"]
        reasoning.append(f"Urgency set to High due to indicators: {hits}.")
        return Urgency.HIGH, hits, 0.80, reasoning

    if matched_med or has_5xx:
        hits = matched_med[:] or ["5xx"]
        reasoning.append(f"Urgency set to Medium due to indicators: {hits}.")
        return Urgency.MEDIUM, hits, 0.65, reasoning

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
    questions: List[str] = []

    env = _extract_environment(raw)
    http_codes = _extract_http_statuses(raw)

    # Time window / start time
    if not _has_time_hint(raw):
        questions.append("When did this start (approx. time and timezone)?")

    # Error details (if we already saw an HTTP code, we can still ask for logs, but not “error code”)
    if not _contains_any(raw, ["error", "message", "screenshot", "log", "stacktrace", "trace"]) and not http_codes:
        questions.append("Do you have an error message, code, or log snippet?")
    elif not _contains_any(raw, ["log", "stacktrace", "trace"]):
        questions.append("Do you have logs/stack trace details that show the failure?")

    # Scope
    if not _contains_any(raw, ["affects", "impact", "users", "teams", "customers", "everyone", "all users"]):
        questions.append("Who is affected (team/customers/how many users)?")

    # Environment
    if not env:
        questions.append("Which environment is affected (prod/staging/dev)?")

    return questions


def _augment_suspected_systems(raw: str, suspected: List[str]) -> List[str]:
    """
    Add deterministic suspected systems based on extracted signals.
    """
    out = list(suspected or [])
    t = raw.lower()

    http_codes = _extract_http_statuses(raw)

    # AuthZ / RBAC hints (helps internal-tool 403 cases feel “smarter”)
    if any(x in t for x in ["403", "forbidden", "rbac", "permission", "access denied", "role"]):
        if "Authorization/RBAC" not in out:
            out.append("Authorization/RBAC")
        # Often paired with auth too
        if "Authentication" not in out:
            out.append("Authentication")

    # 401 is strongly auth-related
    if "401" in http_codes or "unauthorized" in t:
        if "Authentication" not in out:
            out.append("Authentication")

    # 5xx / perf hints
    if any(c.startswith("5") for c in http_codes):
        if "Web/API" not in out:
            out.append("Web/API")

    if any(x in t for x in ["db", "database", "query", "slow query", "db latency"]):
        if "Database" not in out:
            out.append("Database")

    if any(x in t for x in ["cache", "redis", "cdn"]):
        if "Caching/CDN" not in out:
            out.append("Caching/CDN")

    if any(x in t for x in ["queue", "backlog", "kafka", "sqs"]):
        if "Queues" not in out:
            out.append("Queues")

    # De-dupe preserve order
    seen = set()
    final: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final


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

    anchors = {"latency", "degraded", "timeout", "timed out", "db latency", "server load", "503", "504", "gateway timeout"}
    has_anchor = any(h in anchors for h in perf_hits)
    strong_perf = any(h in perf_hits for h in ["latency", "timeout", "timed out", "degraded", "db latency", "504", "503"])
    has_payments_words = bool(_contains_any(raw, ["checkout", "payment", "refund", "billing"]))

    if (len(perf_hits) >= 2 and has_anchor) or (strong_perf and has_payments_words):
        if category in (Category.CUSTOMER_SUPPORT, Category.GENERAL_OPS):
            category = Category.ENGINEERING
            matched_keywords["performance"] = perf_hits
            reasoning.append(f"Performance degradation indicators found ({perf_hits}); overriding category to Engineering.")
            suspected = ["Performance"] + (["CI/CD"] if _contains_any(raw, ["deploy", "release", "rollback", "ci", "pipeline"]) else [])
            cat_conf = max(cat_conf, 0.70)

    urgency, urg_matches, urg_conf, urg_reason = _classify_urgency(raw)
    matched_keywords["urgency"] = urg_matches
    reasoning.extend(urg_reason)

    impact, impact_matches, impact_reason = _infer_impact(raw)
    if impact_matches:
        matched_keywords["impact"] = impact_matches
    reasoning.extend(impact_reason)

    # Add structured signals into matched_keywords for transparency/debugging
    http_codes = _extract_http_statuses(raw)
    if http_codes:
        matched_keywords["http_status"] = http_codes
    env = _extract_environment(raw)
    if env:
        matched_keywords["environment"] = [env]

    suspected = _augment_suspected_systems(raw, suspected)

    questions = _missing_info_questions(raw)

    # Combine confidence signals (weighted average)
    confidence = round(
        (0.55 * cat_conf)
        + (0.35 * urg_conf)
        + (0.10 * (0.75 if impact != "Unknown/unclear impact" else 0.45)),
        2,
    )

    # Human review triggers
    needs_review = False
    if category == Category.GENERAL_OPS and confidence < 0.55:
        needs_review = True
        reasoning.append("Low confidence category; recommend human review.")
    if urgency == Urgency.HIGH and "error" not in raw.lower() and "log" not in raw.lower() and not http_codes:
        needs_review = True
        reasoning.append("High urgency without supporting error/log details; recommend human review.")
    if len(questions) >= 3:
        needs_review = True
        reasoning.append("Multiple missing critical fields; recommend collecting info before actioning.")

    next_actions = ACTION_PLAYBOOK.get(category, ACTION_PLAYBOOK[Category.GENERAL_OPS]).copy()

    # Runbook recommendations (DO NOT invent runbooks you don't have)
    recommended_runbooks: List[str] = []
    if category == Category.IT_OPS:
        # If it's auth / VPN / 401/403 / permissions, use the existing auth runbook you already have
        if any(s in suspected for s in ["Authentication", "Authorization/RBAC"]):
            recommended_runbooks = ["RBK-IT-AUTH-001"]
    elif category == Category.CUSTOMER_SUPPORT:
        recommended_runbooks = ["RBK-CS-PAYMENTS-010"]
    elif category == Category.ENGINEERING:
        if "performance" in matched_keywords:
            recommended_runbooks = ["RBK-ENG-PERF-120"]
        else:
            recommended_runbooks = ["RBK-ENG-CICD-101"]
    elif category == Category.OPERATIONS:
        # Distinguish inventory/WMS vs logistics
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
        citations=citations,
    )

    return asdict(ticket)