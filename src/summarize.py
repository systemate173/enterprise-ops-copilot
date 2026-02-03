from __future__ import annotations

from typing import Dict, List


def summarize_ticket(ticket: Dict) -> Dict:
    """
    Deterministic summary layer (no AI).

    Output is a stable JSON contract that a UI/API could consume.
    """
    citations: List[Dict] = ticket.get("citations", []) or []
    runbooks: List[str] = ticket.get("recommended_runbooks", []) or []

    category = ticket.get("category", "Unknown")
    urgency = ticket.get("urgency", "Unknown")
    impact = ticket.get("impact", "Unknown/unclear impact")
    confidence = ticket.get("confidence", None)
    needs_review = bool(ticket.get("needs_human_review", False))

    sources = [c.get("doc_id") for c in citations if c.get("doc_id")]

    summary = f"{category} incident with {urgency} urgency. Impact: {impact}."
    if confidence is not None:
        summary += f" Confidence: {confidence}."
    if needs_review:
        summary += " Human review recommended."

    actions = ticket.get("next_actions", []) or []
    questions = ticket.get("missing_info_questions", []) or []

    ready = bool(sources)

    return {
        "ready": ready,
        "summary": summary,
        "key_details": {
            "ticket_id": ticket.get("ticket_id"),
            "category": category,
            "urgency": urgency,
            "impact": impact,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "suspected_systems": ticket.get("suspected_systems", []),
            "recommended_runbooks": runbooks,
        },
        "recommended_actions": actions[:5],
        "questions_to_answer": questions[:5],
        "sources": sources,
    }
