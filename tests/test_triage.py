import pytest

from src.triage import triage_incident, Category, Urgency


def test_empty_input_raises():
    with pytest.raises(ValueError):
        triage_incident("   ")


def test_login_outage_classifies_it_ops_high():
    text = """Users cannot log in to the internal dashboard.
Error: Authentication service unavailable (503).
Started ~10 minutes ago. Affects multiple teams. Production."""
    ticket = triage_incident(text)

    assert ticket["category"] == Category.IT_OPS.value
    assert ticket["urgency"] == Urgency.HIGH.value
    assert "Authentication" in ticket["suspected_systems"]
    assert ticket["needs_human_review"] is False


def test_unknown_is_conservative_and_asks_questions():
    text = "Something seems off. Please look into it."
    ticket = triage_incident(text)

    assert ticket["category"] == Category.GENERAL_OPS.value
    assert ticket["confidence"] <= 0.6
    assert len(ticket["missing_info_questions"]) >= 2
    assert ticket["needs_human_review"] is True


def test_customer_billing_issue_routes_support():
    text = """Customers report charges failing at checkout.
Billing error occurs intermittently. Started today in production."""
    ticket = triage_incident(text)

    assert ticket["category"] == Category.CUSTOMER_SUPPORT.value
    assert ticket["urgency"] in [Urgency.MEDIUM.value, Urgency.HIGH.value]
    assert len(ticket["next_actions"]) > 0
