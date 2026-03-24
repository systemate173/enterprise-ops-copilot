from src.incident_store import StoredIncident
from src.similarity import retrieve_similar_incidents


def test_similarity_prefers_auth_incident():
    corpus = [
        StoredIncident(
            incident_id="INC-1",
            created_at_utc="2026-01-01T00:00:00+00:00",
            title="403 errors after role change",
            description="Users see 403 Forbidden after RBAC update. IAM policy changed.",
            category="IT Ops",
            urgency="High",
            impact="Multiple users affected",
        ),
        StoredIncident(
            incident_id="INC-2",
            created_at_utc="2026-01-02T00:00:00+00:00",
            title="Warehouse label printer down",
            description="Label printing system down; orders delayed.",
            category="Operations",
            urgency="High",
            impact="Shipping delayed",
        ),
    ]

    query = "403 forbidden after permissions update"
    sims = retrieve_similar_incidents(query, corpus, top_k=1, min_score=0.05)

    assert len(sims) == 1
    assert sims[0].incident_id == "INC-1"
    assert sims[0].score > 0.05


def test_similarity_empty_when_no_query_or_corpus():
    assert retrieve_similar_incidents("", [], top_k=3) == []