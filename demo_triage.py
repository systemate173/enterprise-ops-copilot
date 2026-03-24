import json
from pathlib import Path

from src.ai_summary import ai_manager_summary
from src.incident_store import IncidentStore, StoredIncident
from src.retrieve import retrieve_runbooks
from src.summarize import summarize_ticket
from src.triage import triage_incident


EXAMPLES_DIR = Path(__file__).parent / "examples" / "incidents"


def _persist_ticket_for_similarity(ticket: dict) -> None:
    """
    Persist a minimal incident record so future triage runs can retrieve similar incidents.
    This is intentionally best-effort and should not block the demo flow.
    """
    try:
        IncidentStore().append(
            StoredIncident(
                incident_id=str(ticket.get("ticket_id") or ""),
                created_at_utc=str(ticket.get("created_at_utc") or ""),
                title=str(ticket.get("title") or ""),
                description=str(ticket.get("description") or ""),
                category=str(ticket.get("category") or ""),
                urgency=str(ticket.get("urgency") or ""),
                impact=str(ticket.get("impact") or ""),
            )
        )
    except Exception as exc:
        print(f"(Warning) Could not store incident for similarity retrieval: {exc}")


def _print_ticket(ticket: dict) -> None:
    print("\n--- TRIAGE OUTPUT ---")
    print(json.dumps(ticket, indent=2))

    runbooks = ticket.get("recommended_runbooks", [])

    print("\n--- RETRIEVED DOCUMENTATION ---")

    if not runbooks:
        print("(No runbooks recommended.)")
    else:
        docs = retrieve_runbooks(runbooks)
        if not docs:
            print("(No runbooks found for the recommended IDs.)")
        else:
            for doc in docs:
                print(f"\n[{doc['doc_id']}]")
                print(doc["excerpt"])

    summary = summarize_ticket(ticket)
    print("\n--- MANAGER SUMMARY (DETERMINISTIC) ---")
    print(json.dumps(summary, indent=2))

    print("\n--- MANAGER REPORT (AI, VALIDATED JSON) ---")
    ai = ai_manager_summary(summary)
    print(json.dumps(ai, indent=2))

    _persist_ticket_for_similarity(ticket)


def _load_example_files() -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted([p for p in EXAMPLES_DIR.iterdir() if p.is_file() and p.suffix in {".txt", ".md"}])


def _choose_example(files: list[Path]) -> str | None:
    if not files:
        print(f"No example files found in: {EXAMPLES_DIR}")
        return None

    print("\nChoose an example:")
    for i, f in enumerate(files, start=1):
        print(f"  {i}) {f.name}")

    choice = input("\nEnter a number (or press Enter to cancel): ").strip()
    if not choice:
        return None

    if not choice.isdigit():
        print("Invalid input. Please enter a number.")
        return None

    idx = int(choice)
    if idx < 1 or idx > len(files):
        print("Number out of range.")
        return None

    return files[idx - 1].read_text(encoding="utf-8")


def _paste_incident() -> str:
    print("\nPaste an incident description. End with an empty line.\n")
    lines: list[str] = []

    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main() -> None:
    print("Enterprise Ops Copilot — Triage Demo")

    files = _load_example_files()

    while True:
        print("\nMenu:")
        print("  1) Run triage on an example incident file")
        print("  2) Paste an incident manually")
        print("  3) Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            text = _choose_example(files)
            if not text:
                continue
            ticket = triage_incident(text)
            _print_ticket(ticket)

        elif choice == "2":
            text = _paste_incident()
            if not text:
                print("No input provided.")
                continue
            ticket = triage_incident(text)
            _print_ticket(ticket)

        elif choice == "3":
            print("Bye.")
            return

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()