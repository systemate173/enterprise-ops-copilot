"""
triage.py

Minimal, deterministic incident triage (no AI).

Purpose:
- Convert unstructured incident text into a structured JSON-like dict
- Provide a stable interface for later RAG + ML integration
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List


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
    missing_info_questions: List[str]
    next_actions: List[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simple_ticket_id(text: str) -> str:
    # Deterministic-ish ID for demo purposes (not production)
    base = abs(hash(text)) % 10**8
    return f"INC-{base:08d}"


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def triage_incident(text: str) -> Dict:
    """
    Convert raw incident text into a structured ticket.
    This is intentionally simple and deterministic.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("incident text is required")

    title = raw.splitlines()[0][:80] if raw else "Untitled incident"

    # Category rules (simple v1)
    if _contains_any(raw, ["login", "auth", "authentication", "sso", "password", "token"]):
        category = "IT Ops"
        suspected = ["Authentication"]
    elif _contains_any(raw, ["payment", "checkout", "refund", "charge", "billing"]):
        category = "Customer Support"
        suspected = ["Payments/Billing"]
    elif _contains_any(raw, ["shipment", "delivery", "warehouse", "route", "fleet"]):
        category = "Operations"
        suspected = ["Logistics"]
    elif _contains_any(raw, ["build failed", "ci", "deploy", "release", "bug"]):
        category = "Engineering"
        suspected = ["CI/CD"]
    else:
        category = "General Ops"
        suspected = []

    # Urgency rules (simple v1)
    if _contains_any(raw, ["outage", "down", "cannot", "can't", "unable", "sev1", "critical"]):
        urgency = "High"
    elif _contains_any(raw, ["slow", "intermittent", "sometimes", "degraded"]):
        urgency = "Medium"
    else:
        urgency = "Low"

    # Impact (very basic)
    if _contains_any(raw, ["multiple teams", "all users", "everyone", "company-wide"]):
        impact = "Broad impact (many users/teams)"
    elif _contains_any(raw, ["customer", "clients", "buyers"]):
        impact = "Customer-facing impact"
    else:
        impact = "Unknown/unclear impact"

    # Missing info questions (helps reduce back-and-forth later)
    missing_questions = []
    if not _contains_any(raw, ["started", "since", "minutes", "hours", "today", "yesterday", "timestamp"]):
        missing_questions.append("When did this start (approx. time and timezone)?")
    if not _contains_any(raw, ["error", "message", "code", "screenshot", "log"]):
        missing_questions.append("Do you have an error message, code, or log snippet?")
    if not _contains_any(raw, ["affects", "impact", "users", "teams"]):
        missing_questions.append("Who is affected (team, customers, how many users)?")

    # Next actions (basic playbook)
    next_actions = []
    if category == "IT Ops" and "Authentication" in suspected:
        next_actions = [
            "Check auth/SSO service status and recent deploys",
            "Verify dependency health (IDP, token service, database)",
            "Collect a log snippet or error code from a failing login attempt",
        ]
    elif category == "Customer Support":
        next_actions = [
            "Confirm scope (which customers, which region, which payment method)",
            "Check recent changes and payment provider status",
            "Collect transaction IDs and timestamps for failing attempts",
        ]
    elif category == "Operations":
        next_actions = [
            "Confirm affected locations/routes and time window",
            "Check upstream dependencies (vendors, dispatch, inventory)",
            "Capture any IDs (shipment/order/vehicle) and current status",
        ]
    elif category == "Engineering":
        next_actions = [
            "Identify failing pipeline step and recent changes",
            "Collect build logs and error output",
            "Check recent deploy/release notes and rollback options",
        ]
    else:
        next_actions = [
            "Clarify the goal and success criteria",
            "Identify owner/team responsible",
            "Collect any relevant IDs, timestamps, and error details",
        ]

    ticket = IncidentTicket(
        ticket_id=_simple_ticket_id(raw),
        created_at_utc=_utc_now_iso(),
        title=title,
        description=raw,
        category=category,
        urgency=urgency,
        impact=impact,
        suspected_systems=suspected,
        missing_info_questions=missing_questions,
        next_actions=next_actions,
    )

    return asdict(ticket)
