from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class StoredIncident:
    """
    Minimal persistent incident record used for deterministic similarity search.

    Keep this schema intentionally small and stable so the incident memory layer
    remains easy to inspect, test, and evolve.
    """

    incident_id: str
    created_at_utc: str
    title: str
    description: str
    category: str
    urgency: str
    impact: str

    def to_compact_text(self) -> str:
        """
        Deterministic canonical text representation used for similarity search.
        """
        return "\n".join(
            [
                f"title: {self.title}",
                f"category: {self.category}",
                f"urgency: {self.urgency}",
                f"impact: {self.impact}",
                f"description: {self.description}",
            ]
        ).strip()


class IncidentStore:
    """
    Minimal JSONL-backed incident store.

    Characteristics:
    - deterministic
    - local-only
    - human-readable
    - easy to version or reset
    """

    def __init__(self, path: str = "data/incidents.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, incident: StoredIncident) -> None:
        """
        Append a single incident record as one JSON line.
        """
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")

    def load_all(self, limit: Optional[int] = None) -> List[StoredIncident]:
        """
        Load incidents in file order. Optionally cap the number loaded.
        """
        if not self.path.exists():
            return []

        out: List[StoredIncident] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                out.append(StoredIncident(**data))

                if limit is not None and len(out) >= limit:
                    break

        return out