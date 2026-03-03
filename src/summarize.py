from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from src.company_retrieve import retrieve_company_docs


# Headings we’ll try to mine for actionable bullets (in priority order)
HIGHLIGHT_SECTIONS = [
    "Initial Checks",
    "Immediate Actions",
    "Triage",
    "Resolution Steps",
    "Mitigation",
    "Rollback",
    "Verification",
]



def _clean_label(value: Any) -> str:
    """
    Turn Enum-like values into clean display strings.
    Handles:
      - Enum objects (value attr)
      - Strings like "Category.OPERATIONS" / "Urgency.HIGH"
      - Raw strings
    """
    if value is None:
        return "Unknown"

    # If it's an Enum instance, prefer its .value
    if hasattr(value, "value"):
        try:
            value = value.value
        except Exception:
            pass

    s = str(value).strip()

    # If it's "Category.OPERATIONS" or "Urgency.HIGH" -> take RHS
    if "." in s and (s.startswith("Category.") or s.startswith("Urgency.")):
        s = s.split(".", 1)[1].strip()

    # If it's an enum-style token "GENERAL_OPS" -> "General Ops"
    if re.fullmatch(r"[A-Z0-9_]+", s):
        s = s.replace("_", " ").title()

    # Common normalization
    s = s.replace("  ", " ").strip()

    # Preserve known casing conventions
    if s.lower() in ("it ops", "itops"):
        return "IT Ops"

    return s or "Unknown"


def _extract_section_block(md: str, heading: str) -> str:
    """
    Return markdown text under a given heading until the next heading.
    Heading match is case-insensitive and supports:
      - "# Heading"
      - "## Heading"
      - "### Heading"
    """
    if not md:
        return ""

    lines = md.splitlines()
    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+" + re.escape(heading) + r"\s*$", re.IGNORECASE)

    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if heading_re.match(line):
            start_idx = i + 1
            break

    if start_idx is None:
        return ""

    # Gather until next heading
    out: List[str] = []
    next_heading_re = re.compile(r"^\s{0,3}#{1,6}\s+")
    for j in range(start_idx, len(lines)):
        if next_heading_re.match(lines[j]):
            break
        out.append(lines[j])

    return "\n".join(out).strip()

def _extract_incident_facts(description: str, max_items: int = 5) -> List[str]:
    """
    Deterministically extract a few high-signal facts from the incident description.
    Keeps it simple: first non-empty lines, trimmed.
    """
    if not description:
        return []
    lines = [ln.strip() for ln in description.splitlines() if ln.strip()]
    # Return first N lines (stable + predictable)
    return lines[:max_items]


def _extract_bullets(text: str, max_items: int = 4) -> List[str]:
    """
    Extract bullet / numbered items deterministically.
    Accepts:
      - "- item"
      - "* item"
      - "1. item"
    """
    if not text:
        return []

    bullets: List[str] = []
    for line in text.splitlines():
        line = line.rstrip()

        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            item = m.group(1).strip()
        else:
            m = re.match(r"^\s*\d+\.\s+(.*)$", line)
            if not m:
                continue
            item = m.group(1).strip()

        # Skip empty / junk
        if not item:
            continue

        # Keep items compact (avoid huge wrapped bullets from excerpts)
        if len(item) > 200:
            item = item[:197].rstrip() + "..."

        bullets.append(item)

        if len(bullets) >= max_items:
            break

    return bullets


def _highlights_from_citations(citations: List[Dict[str, Any]], max_total: int = 4) -> List[str]:
    """
    Mine up to N highlight bullets across citations, in a deterministic order:
      - citation order
      - section priority order
      - bullet order in excerpt
    """
    highlights: List[str] = []
    seen = set()

    for c in citations or []:
        excerpt = (c.get("excerpt") or "").strip()
        if not excerpt:
            continue

        for section in HIGHLIGHT_SECTIONS:
            block = _extract_section_block(excerpt, section)
            for b in _extract_bullets(block, max_items=max_total):
                key = b.lower()
                if key in seen:
                    continue
                seen.add(key)
                highlights.append(b)
                if len(highlights) >= max_total:
                    return highlights

    return highlights

def _escalation_from_citations(citations: List[Dict[str, Any]], max_total: int = 2) -> List[str]:
    targets: List[str] = []
    seen = set()

    for c in citations or []:
        excerpt = (c.get("excerpt") or "").strip()
        if not excerpt:
            continue
        block = _extract_section_block(excerpt, "Escalation")
        for b in _extract_bullets(block, max_items=max_total):
            k = b.lower()
            if k in seen:
                continue
            seen.add(k)
            targets.append(b)
            if len(targets) >= max_total:
                return targets
    return targets


def summarize_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic summary layer (no AI).

    Output is a stable JSON contract that a UI/API could consume.
    """
    citations: List[Dict[str, Any]] = ticket.get("citations", []) or []
    runbooks: List[str] = ticket.get("recommended_runbooks", []) or []

    category = _clean_label(ticket.get("category", "Unknown"))
    urgency = _clean_label(ticket.get("urgency", "Unknown"))
    impact = _clean_label(ticket.get("impact", "Unknown/unclear impact"))

    description = str(ticket.get("description") or "")
    incident_facts = _extract_incident_facts(description, max_items=5)

    confidence = ticket.get("confidence", None)
    needs_review = bool(ticket.get("needs_human_review", False))
    suspected_systems = ticket.get("suspected_systems", []) or []

    # Runbook/citation grounding
    runbook_sources = [c.get("doc_id") for c in citations if c.get("doc_id")]

    #Ready should mean: we have citations/runbooks to ground output
    ready = bool(runbook_sources)
    blocker_reason = None if ready else "no_citations"

    company_sources: List[str] = []
    if ready:
        company_docs = retrieve_company_docs(
            ["TEAM_DIRECTORY", "SYSTEM_OWNERSHIP", "ESCALATION_POLICY", "INCIDENT_SEVERITY"],
            max_chars=250,
        )
        company_sources = [f"COMPANY:{d['doc_id']}" for d in company_docs]

    sources = _dedupe_preserve_order(runbook_sources + company_sources)

    # Deterministic highlights (only if we actually have citations)
    highlights: List[str] = _highlights_from_citations(citations, max_total=4) if ready else []

    # Escalation note: deterministic rule on urgency
    escalation_targets = _escalation_from_citations(citations, max_total=2) if ready else []
    escalation_note = None
    if urgency.lower() == "high":
        escalation_note = (
            "High urgency: notify on-call and the operational owner. "
            "Include start time, current status, scope (users/locations), and any error text/logs."
        )

    # Summary sentence
    summary = f"{category} incident with {urgency} urgency. Impact: {impact}."
    if confidence is not None:
        summary += f" Confidence: {confidence}."
    if needs_review:
        summary += " Human review recommended."
    if ready and highlights:
        summary += " Key runbook checks available."

    actions = ticket.get("next_actions", []) or []
    questions = ticket.get("missing_info_questions", []) or []

    return {
        "ready": ready,
        "blocker_reason": blocker_reason,
        "summary": summary,
        "incident_facts": incident_facts,
        "highlights": highlights,
        "escalation_note": escalation_note,
        "escalation_targets": escalation_targets,
        "key_details": {
            "ticket_id": ticket.get("ticket_id"),
            "category": category,
            "urgency": urgency,
            "impact": impact,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "suspected_systems": suspected_systems,
            "recommended_runbooks": runbooks,
        },
        "recommended_actions": actions[:5],
        "questions_to_answer": questions[:5],
        "sources": sources,
    }
