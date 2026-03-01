from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


RUNBOOKS_DIR = Path(__file__).parent.parent / "docs" / "runbooks"


def _normalize_runbook_id(value: str) -> Optional[str]:
    """
    Accepts:
      - "RBK-ENG-CICD-101"
      - "RBK-ENG-CICD-101.md"
      - "docs/runbooks/RBK-ENG-CICD-101.md"
    Returns normalized ID "RBK-ENG-CICD-101" or None if invalid.
    """
    if not value:
        return None

    s = str(value).strip()

    # If someone passed a path, take just the filename
    s = Path(s).name

    # Remove ".md" if present
    if s.lower().endswith(".md"):
        s = s[:-3]

    s = s.strip()

    # Very small sanity check: expected runbook prefix
    if not s.startswith("RBK-"):
        return None

    return s


def _safe_excerpt(text: str, max_chars: int) -> str:
    """
    Deterministic excerpt: trim to max_chars but try not to cut mid-word.
    """
    if max_chars <= 0:
        return ""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t

    cut = t[:max_chars]
    # try to cut at last whitespace after 70% of window
    pivot = int(max_chars * 0.7)
    last_space = cut.rfind(" ")
    if last_space >= pivot:
        cut = cut[:last_space].rstrip()
    return cut.rstrip() + "..."


def retrieve_runbooks(runbook_ids: List[str], max_chars: int = 750) -> List[Dict]:
    """
    Deterministically retrieve runbook documents by ID.

    - Accepts IDs, filenames, or paths.
    - Skips missing/invalid inputs (still deterministic).
    - Returns:
        {"doc_id": "...", "path": "...", "excerpt": "..."}
    """
    results: List[Dict] = []

    for raw_id in runbook_ids or []:
        rbk_id = _normalize_runbook_id(raw_id)
        if not rbk_id:
            continue

        path = RUNBOOKS_DIR / f"{rbk_id}.md"
        if not path.exists():
            continue

        content = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        excerpt = _safe_excerpt(content, max_chars=max_chars)

        results.append(
            {
                "doc_id": rbk_id,
                "path": str(path.relative_to(Path(__file__).parent.parent)),
                "excerpt": excerpt,
            }
        )

    return results