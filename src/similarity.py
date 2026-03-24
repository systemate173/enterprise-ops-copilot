from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.incident_store import StoredIncident


@dataclass(frozen=True)
class SimilarIncident:
    """
    Lightweight similarity result returned to triage/summarization layers.
    """

    incident_id: str
    score: float
    title: str
    category: str
    urgency: str


def _tokenize(text: str) -> List[str]:
    """
    Deterministic tokenization for fallback similarity mode.
    Keeps only lowercase alphanumeric tokens.
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    # Tiny stoplist to reduce obvious noise in fallback mode.
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
    }
    return [tok for tok in tokens if tok and tok not in stopwords]


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Deterministic no-dependency fallback:
    similarity = overlap(tokens) / union(tokens)
    """
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / float(len(a_tokens | b_tokens))


def _tfidf_cosine_similarity(
    query_text: str,
    corpus: List[StoredIncident],
) -> List[Tuple[int, float]]:
    """
    TF-IDF + cosine similarity implementation.
    Requires scikit-learn.

    Returns:
        List of (corpus_index, similarity_score)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [incident.to_compact_text() for incident in corpus]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(docs + [query_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    return [(i, float(scores[i])) for i in range(len(corpus))]


def retrieve_similar_incidents(
    query_text: str,
    corpus: List[StoredIncident],
    top_k: int = 3,
    min_score: float = 0.18,
) -> List[SimilarIncident]:
    """
    Deterministic incident similarity search.

    Behavior:
    - Prefer TF-IDF + cosine similarity if scikit-learn is available
    - Fall back to deterministic token Jaccard similarity otherwise

    Notes:
    - This is retrieval, not generation.
    - It is advisory only. It should not override deterministic triage logic.
    """
    query_text = (query_text or "").strip()
    if not query_text or not corpus:
        return []

    try:
        scored = _tfidf_cosine_similarity(query_text, corpus)
    except Exception:
        scored = [
            (i, _jaccard_similarity(query_text, corpus[i].to_compact_text()))
            for i in range(len(corpus))
        ]

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)

    out: List[SimilarIncident] = []
    for idx, score in ranked:
        if score < float(min_score):
            continue

        incident = corpus[idx]
        out.append(
            SimilarIncident(
                incident_id=incident.incident_id,
                score=float(score),
                title=incident.title,
                category=incident.category,
                urgency=incident.urgency,
            )
        )

        if len(out) >= top_k:
            break

    return out