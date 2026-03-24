from pathlib import Path

from src.incident_store import IncidentStore, StoredIncident
from src.triage import triage_incident

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "incidents"


def main() -> None:
    store = IncidentStore()

    files = sorted([p for p in EXAMPLES_DIR.iterdir() if p.is_file() and p.suffix in {".txt", ".md"}])

    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        ticket = triage_incident(text)

        store.append(
            StoredIncident(
                incident_id=str(ticket.get("ticket_id") or ""),
                created_at_utc=str(ticket.get("created_at_utc") or ""),
                title=str(ticket.get("title") or ""),
                description=str(ticket.get("description") or ""),
                category=str(ticket.get("category") or ""),
                urgency=str(ticket.get("urgency") or ""),
                impact=str(ticket.get("impact") or ""),
                suspected_systems=[
                    str(x) for x in (ticket.get("suspected_systems") or []) if str(x).strip()
                ],
                recommended_runbooks=[
                    str(x) for x in (ticket.get("recommended_runbooks") or []) if str(x).strip()
                ],
            )
        )

    print(f"Seeded incident store from {len(files)} example files.")


if __name__ == "__main__":
    main()