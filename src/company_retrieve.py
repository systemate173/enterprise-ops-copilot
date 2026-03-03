from __future__ import annotations

from pathlib import Path
from typing import Dict, List


COMPANY_DIR = Path(__file__).parent.parent / "docs" / "company"


def retrieve_company_docs(doc_ids: List[str], max_chars: int = 900) -> List[Dict]:
    """
    Deterministically retrieve company knowledge docs by doc_id (without .md).
    Example doc_id: "TEAM_DIRECTORY" -> docs/company/TEAM_DIRECTORY.md

    Returns:
      [
        {"doc_id": "...", "path": "docs/company/....md", "excerpt": "..."},
        ...
      ]
    """
    results: List[Dict] = []

    for doc_id in doc_ids or []:
        safe = str(doc_id).strip()
        if not safe:
            continue

        path = COMPANY_DIR / f"{safe}.md"
        if not path.exists():
            continue

        content = path.read_text(encoding="utf-8").strip()
        excerpt = content[:max_chars]

        results.append(
            {
                "doc_id": safe,
                "path": str(path.relative_to(Path(__file__).parent.parent)),
                "excerpt": excerpt,
            }
        )

    return results