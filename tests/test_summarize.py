from src.summarize import summarize_ticket


def test_summarize_cleans_enum_like_strings_and_adds_escalation_note():
    ticket = {
        "ticket_id": "INC-00000001",
        "category": "Category.OPERATIONS",
        "urgency": "Urgency.HIGH",
        "impact": "Unknown/unclear impact",
        "confidence": 0.77,
        "needs_human_review": True,
        "suspected_systems": ["WMS"],
        "recommended_runbooks": ["RBK-OPS-WMS-060"],
        "next_actions": ["Do X", "Do Y"],
        "missing_info_questions": ["What region?"],
        "citations": [
            {
                "doc_id": "RBK-OPS-WMS-060",
                "path": "docs/runbooks/RBK-OPS-WMS-060.md",
                "excerpt": """# RBK-OPS-WMS-060

## Initial Checks
- Confirm affected warehouse(s) and time window
- Validate printer/scanner connectivity

## Resolution Steps
- Restart print spooler
- Fail over to backup printer
""",
            }
        ],
    }

    s = summarize_ticket(ticket)
    assert s["key_details"]["category"] == "Operations"
    assert s["key_details"]["urgency"] == "High"
    assert s["ready"] is True
    assert s["blocker_reason"] is None
    assert s["escalation_note"] is not None
    assert len(s["highlights"]) >= 1
    assert any("Confirm affected warehouse" in h for h in s["highlights"])



def test_summarize_sets_blocker_reason_when_not_ready():
    ticket = {
        "ticket_id": "INC-00000002",
        "category": "Operations",
        "urgency": "Medium",
        "impact": "Shipping delayed",
        "recommended_runbooks": [],
        "citations": [],  # <- key condition
    }

    s = summarize_ticket(ticket)
    assert s["ready"] is False
    assert s["blocker_reason"] == "no_citations"
    assert s["highlights"] == []
    assert s["sources"] == []
