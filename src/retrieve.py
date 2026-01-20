from pathlib import Path
from typing import Dict, List


RUNBOOKS_DIR = Path(__file__).parent.parent / "docs" / "runbooks"


def retrieve_runbooks(runbook_ids: List[str], max_chars: int = 500) -> List[Dict]:
    """
    Deterministically retrieve runbook documents by ID.
    """
    results: List[Dict] = []

    for rbk_id in runbook_ids:
        path = RUNBOOKS_DIR / f"{rbk_id}.md"
        if not path.exists():
            continue

        content = path.read_text(encoding="utf-8").strip()
        excerpt = content[:max_chars]

        results.append(
            {
                "doc_id": rbk_id,
                "path": str(path),
                "excerpt": excerpt,
            }
        )

    return results
