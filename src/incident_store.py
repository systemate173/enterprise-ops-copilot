from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class StoredIncident:
    incident_id: str
    created_at_utc: str
    title: str
    description: str
    category: str
    urgency: str
    impact: str
    suspected_systems: List[str] = field(default_factory=list)
    recommended_runbooks: List[str] = field(default_factory=list)

    def to_compact_text(self) -> str:
        systems = ", ".join(self.suspected_systems)
        runbooks = ", ".join(self.recommended_runbooks)

        return "\n".join(
            [
                f"title: {self.title}",
                f"category: {self.category}",
                f"urgency: {self.urgency}",
                f"impact: {self.impact}",
                f"systems: {systems}",
                f"runbooks: {runbooks}",
                f"description: {self.description}",
            ]
        ).strip()


class IncidentStore:
    def __init__(self, path: str = "data/incidents.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, incident: StoredIncident) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")

    def load_all(self, limit: Optional[int] = None) -> List[StoredIncident]:
        if not self.path.exists():
            return []

        out: List[StoredIncident] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Backward-compatible with older rows
                data.setdefault("suspected_systems", [])
                data.setdefault("recommended_runbooks", [])

                out.append(StoredIncident(**data))

                if limit is not None and len(out) >= limit:
                    break

        return out