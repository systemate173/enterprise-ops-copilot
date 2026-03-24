from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.company_retrieve import retrieve_company_docs


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

    if hasattr(value, "value"):
        try:
            value = value.value
        except Exception:
            pass

    s = str(value).strip()

    if "." in s and (s.startswith("Category.") or s.startswith("Urgency.")):
        s = s.split(".", 1)[1].strip()

    if re.fullmatch(r"[A-Z0-9_]+", s):
        s = s.replace("_", " ").title()

    s = s.replace("  ", " ").strip()

    if s.lower() in ("it ops", "itops"):
        return "IT Ops"

    return s or "Unknown"


def _extract_section_block(md: str, heading: str) -> str:
    """
    Return markdown text under a given heading until the next heading.
    Heading match is case-insensitive and supports:
      - # Heading
      - ## Heading
      - ### Heading
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
    """
    if not description:
        return []

    lines = [line.strip() for line in description.splitlines() if line.strip()]
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

        match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if match:
            item = match.group(1).strip()
        else:
            match = re.match(r"^\s*\d+\.\s+(.*)$", line)
            if not match:
                continue
            item = match.group(1).strip()

        if not item:
            continue

        if len(item) > 200:
            item = item[:197].rstrip() + "..."

        bullets.append(item)

        if len(bullets) >= max_items:
            break

    return bullets


def _highlights_from_citations(citations: List[Dict[str, Any]], max_total: int = 4) -> List[str]:
    """
    Mine highlight bullets from runbook excerpts in deterministic order.
    """
    highlights: List[str] = []
    seen = set()

    for citation in citations or []:
        excerpt = (citation.get("excerpt") or "").strip()
        if not excerpt:
            continue

        for section in HIGHLIGHT_SECTIONS:
            block = _extract_section_block(excerpt, section)
            for bullet in _extract_bullets(block, max_items=max_total):
                key = bullet.lower()
                if key in seen:
                    continue
                seen.add(key)
                highlights.append(bullet)

                if len(highlights) >= max_total:
                    return highlights

    return highlights


def _escalation_from_citations(citations: List[Dict[str, Any]], max_total: int = 2) -> List[str]:
    targets: List[str] = []
    seen = set()

    for citation in citations or []:
        excerpt = (citation.get("excerpt") or "").strip()
        if not excerpt:
            continue

        block = _extract_section_block(excerpt, "Escalation")
        for bullet in _extract_bullets(block, max_items=max_total):
            key = bullet.lower()
            if key in seen:
                continue
            seen.add(key)
            targets.append(bullet)

            if len(targets) >= max_total:
                return targets

    return targets


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []

    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)

    return out


def _build_similarity_note(similar_incidents: List[Dict[str, Any]]) -> Optional[str]:
    """
    Build a safe reference-only similarity note.

    Important:
    - This does NOT import facts from past incidents.
    - It only signals that similar incidents exist and provides IDs/titles as references.
    """
    if not similar_incidents:
        return None

    top = similar_incidents[0]

    incident_id = str(top.get("incident_id") or "").strip()
    title = str(top.get("title") or "").strip()
    score = top.get("score")

    if not incident_id:
        return None

    if isinstance(score, (int, float)):
        return f"Similar past incident identified: {incident_id} (score={score:.3f}) — {title}."
    return f"Similar past incident identified: {incident_id} — {title}."


def summarize_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic summary layer (no AI).

    Output is a stable JSON contract that UI/API/LLM layers can consume.
    Similar incidents are references only and must not be treated as incident facts.
    """
    citations: List[Dict[str, Any]] = ticket.get("citations", []) or []
    runbooks: List[str] = ticket.get("recommended_runbooks", []) or []
    similar_incidents: List[Dict[str, Any]] = ticket.get("similar_incidents", []) or []

    category = _clean_label(ticket.get("category", "Unknown"))
    urgency = _clean_label(ticket.get("urgency", "Unknown"))
    impact = _clean_label(ticket.get("impact", "Unknown/unclear impact"))

    description = str(ticket.get("description") or "")
    incident_facts = _extract_incident_facts(description, max_items=5)

    confidence = ticket.get("confidence")
    needs_review = bool(ticket.get("needs_human_review", False))
    suspected_systems = ticket.get("suspected_systems", []) or []

    runbook_sources: List[str] = [
        str(doc_id).strip()
        for citation in citations
        for doc_id in [citation.get("doc_id")]
        if isinstance(doc_id, str) and doc_id.strip()
    ]

    ready = bool(runbook_sources)
    blocker_reason = None if ready else "no_citations"

    company_sources: List[str] = []
    if ready:
        company_docs = retrieve_company_docs(
            ["TEAM_DIRECTORY", "SYSTEM_OWNERSHIP", "ESCALATION_POLICY", "INCIDENT_SEVERITY"],
            max_chars=250,
        )
        company_sources = [
            f"COMPANY:{doc_id}"
            for doc in company_docs
            for doc_id in [doc.get("doc_id")]
            if isinstance(doc_id, str) and doc_id.strip()
        ]

    similar_sources: List[str] = []
    for item in similar_incidents:
        incident_id = str(item.get("incident_id") or "").strip()
        if incident_id:
            similar_sources.append(f"INCIDENT:{incident_id}")

    sources = _dedupe_preserve_order(runbook_sources + company_sources + similar_sources)

    highlights: List[str] = _highlights_from_citations(citations, max_total=4) if ready else []
    escalation_targets = _escalation_from_citations(citations, max_total=2) if ready else []

    escalation_note = None
    if urgency.lower() == "high":
        escalation_note = (
            "High urgency: notify on-call and the operational owner. "
            "Include start time, current status, scope (users/locations), and any error text/logs."
        )

    similarity_note = _build_similarity_note(similar_incidents)

    summary = f"{category} incident with {urgency} urgency. Impact: {impact}."
    if confidence is not None:
        summary += f" Confidence: {confidence}."
    if needs_review:
        summary += " Human review recommended."
    if ready and highlights:
        summary += " Key runbook checks available."
    if similarity_note:
        summary += " Similar historical incident reference available."

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
        "similarity_note": similarity_note,
        "key_details": {
            "ticket_id": ticket.get("ticket_id"),
            "category": category,
            "urgency": urgency,
            "impact": impact,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "suspected_systems": suspected_systems,
            "recommended_runbooks": runbooks,
            "similar_incidents": similar_incidents[:3],
        },
        "recommended_actions": actions[:5],
        "questions_to_answer": questions[:5],
        "sources": sources,
    }