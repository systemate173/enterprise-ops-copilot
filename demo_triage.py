

import json
from src.triage import triage_incident


def main() -> None:
    print("Paste an incident description. End with an empty line.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        print("No input provided.")
        return

    ticket = triage_incident(text)
    print("\n--- TRIAGE OUTPUT ---")
    print(json.dumps(ticket, indent=2))


if __name__ == "__main__":
    main()
