from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional


# Ollama local server + model
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")

# --- "Company knowledge" (v1) ---
# Deterministic team directory / escalation options.
# This is the simplest form of RAG: you control what "the company knows".

ESCALATION_ALLOWLIST_BY_CATEGORY = {
    "IT Ops": ["IT Ops On-Call", "IAM/Security", "Network"],
    "Engineering": ["Platform", "SRE", "Mobile", "Frontend"],
    "Customer Support": ["Customer Support Lead", "Payments Engineering", "Finance"],
    "Operations": ["Warehouse Ops", "Logistics Ops", "Operations Lead"],
    "General Ops": ["IT Ops On-Call", "Operations Lead", "Engineering On-Call"],
}

ESCALATION_ALLOWLIST_BY_SYSTEM = {
    "Authentication": ["IAM/Security", "IT Ops On-Call"],
    "CI/CD": ["Platform", "SRE"],
    "Payments/Billing": ["Payments Engineering", "Finance"],
    "Performance": ["SRE", "Platform"],
    "Logistics": ["Logistics Ops", "Warehouse Ops"],
}

def _allowed_escalation_teams(det: Dict[str, Any]) -> List[str]:
    key = det.get("key_details") or {}
    category = str(key.get("category") or "General Ops")
    systems = key.get("suspected_systems") or []

    allowed = set(ESCALATION_ALLOWLIST_BY_CATEGORY.get(category, []))
    for s in systems:
        for team in ESCALATION_ALLOWLIST_BY_SYSTEM.get(str(s), []):
            allowed.add(team)

    # Safe fallback (never empty)
    out = sorted(allowed) if allowed else ["IT Ops On-Call", "Engineering On-Call", "Operations Lead"]
    return out

# JSON keys we require from the model
REPORT_KEYS = [
    "executive_update",
    "what_we_know",
    "what_we_dont_know",
    "hypotheses",
    "recommended_plan",
    "escalation",
    "sources",
]

# Strict schema for Ollama structured outputs
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_update": {"type": "string"},
        "what_we_know": {"type": "array", "items": {"type": "string"}},
        "what_we_dont_know": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "why": {"type": "string"},
                    "how_to_validate": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
                },
                "required": ["hypothesis", "why", "how_to_validate", "confidence"],
                "additionalProperties": False,
            },
        },
        "recommended_plan": {
            "type": "object",
            "properties": {
                "now": {"type": "array", "items": {"type": "string"}},
                "next": {"type": "array", "items": {"type": "string"}},
                "if_unresolved": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["now", "next", "if_unresolved"],
            "additionalProperties": False,
        },
        "escalation": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": REPORT_KEYS,
    "additionalProperties": False,
}


def _normalize_list(value: Any, max_items: int = 5) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            s = str(x).strip()
            if not s:
                continue
            if len(s) > 220:
                s = s[:217].rstrip() + "..."
            out.append(s)
            if len(out) >= max_items:
                break
        return out
    s = str(value).strip()
    return [s] if s else []


def _normalize_plan(value: Any) -> Dict[str, List[str]]:
    empty = {"now": [], "next": [], "if_unresolved": []}
    if not isinstance(value, dict):
        return empty

    return {
        "now": _normalize_list(value.get("now"), max_items=5),
        "next": _normalize_list(value.get("next"), max_items=5),
        "if_unresolved": _normalize_list(value.get("if_unresolved"), max_items=5),
    }


def _normalize_hypotheses(value: Any, max_items: int = 4) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []

    out: List[Dict[str, str]] = []
    allowed_conf = {"Low", "Medium", "High"}

    overconfident = ["definitely", "confirmed", "root cause is", "caused by"]

    def trim(s: str, n: int = 240) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 3].rstrip() + "..."

    for item in value:
        if not isinstance(item, dict):
            continue

        hyp = trim(str(item.get("hypothesis") or ""))
        why = trim(str(item.get("why") or ""))
        validate = trim(str(item.get("how_to_validate") or ""))
        conf = str(item.get("confidence") or "").strip()

        if conf not in allowed_conf:
            conf = "Low"

        combined = f"{hyp} {why} {validate}".lower()
        if any(w in combined for w in overconfident):
            hyp = "Suspected issue; needs validation"
            why = why or "Unknown"
            validate = validate or "Check logs/metrics to validate"
            conf = "Low"

        if not hyp or not validate:
            continue

        out.append(
            {
                "hypothesis": hyp,
                "why": why or "Unknown",
                "how_to_validate": validate,
                "confidence": conf,
            }
        )

        if len(out) >= max_items:
            break

    return out


def _build_prompt(det: Dict[str, Any]) -> str:
    summary = (det.get("summary") or "").strip()
    highlights = det.get("highlights", []) or []
    incident_facts = det.get("incident_facts", []) or []
    actions = det.get("recommended_actions", []) or []
    questions = det.get("questions_to_answer", []) or []
    escalation_targets = det.get("escalation_targets", []) or []
    allowed_teams = _allowed_escalation_teams(det)
    sources = det.get("sources", []) or []

    def bullets(items: List[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) if items else "- (none)"

    srcs = ", ".join(sources) if sources else "(none)"

    schema_hint = (
        "{\n"
        '  "executive_update": string,\n'
        '  "what_we_know": [string, ...],\n'
        '  "what_we_dont_know": [string, ...],\n'
        '  "hypotheses": [\n'
        '    {\n'
        '      "hypothesis": string,\n'
        '      "why": string,\n'
        '      "how_to_validate": string,\n'
        '      "confidence": "Low" | "Medium" | "High"\n'
        '    }, ...\n'
        "  ],\n"
        '  "recommended_plan": {\n'
        '    "now": [string, ...],\n'
        '    "next": [string, ...],\n'
        '    "if_unresolved": [string, ...]\n'
        "  },\n"
        '  "escalation": string,\n'
        '  "sources": [string, ...]\n'
        "}"
    )

    return (
        "You are a senior incident commander writing a reliable, business-useful incident report.\n\n"
        "CRITICAL RULES (do not break):\n"
        "1) Use ONLY the Allowed Facts below.\n"
        "2) Do NOT invent root cause, metrics, or timelines.\n"
        "3) If you truly cannot form a plausible hypothesis from the Allowed Facts, use 'Unknown'.\n"
        "4) Output MUST be valid JSON and MUST match the schema exactly.\n"
        "5) Do NOT include any text outside the JSON.\n\n"
        f"Schema:\n{schema_hint}\n\n"
        "Allowed Facts:\n"
        f"- Deterministic summary: {summary}\n"
        f"- Runbook highlights:\n{bullets(highlights)}\n"
        f"- Recommended actions:\n{bullets(actions)}\n"
        f"- Incident facts:\n{bullets(incident_facts)}\n"
        f"- Open questions:\n{bullets(questions)}\n"
        f"- Escalation targets (from runbooks):\n{bullets(escalation_targets)}\n"
        f"- Allowed escalation teams (choose from this list ONLY): {', '.join(allowed_teams)}\n"
        f"- Sources: {srcs}\n\n"
        "Guidance:\n"
        "- Avoid repeating the deterministic summary verbatim; rephrase into clear business language.\n"
        "- recommended_plan.if_unresolved should be escalation or deeper investigation steps (not normal mitigation steps).\n"
        "- executive_update: 1–2 sentences for leadership.\n"
        "- what_we_know: 3–6 bullets derived from summary/highlights.\n"
        "- what_we_dont_know: 2–5 bullets derived from open questions.\n"
        "- Escalation must choose from allowed teams; do not output Unknown.\n"
        "- hypotheses: Provide 2–4 plausible hypotheses (NOT facts) consistent with Allowed Facts.\n"
        "- Each hypothesis MUST include why it fits the facts and how to validate it.\n"
        "- Use cautious language (e.g., 'Possibly', 'Could be', 'Suspected') and NEVER state a root cause as confirmed.\n"
        "- confidence must be Low/Medium/High based on strength of the Allowed Facts.\n"
        "- recommended_plan MUST be an object with keys now/next/if_unresolved, each a list of strings.\n"
        "- escalation: If escalation targets exist, name them; otherwise 'Unknown'.\n"
    )


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": REPORT_SCHEMA,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.1,
            "num_predict": 450,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()
    except Exception:
        return ""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _validate_and_fix_report(report: Dict[str, Any], sources: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None

    if sorted(report.keys()) != sorted(REPORT_KEYS):
        return None

    cleaned: Dict[str, Any] = {}

    # ---- strings ----
    cleaned["executive_update"] = str(report.get("executive_update") or "Unknown").strip() or "Unknown"

    cleaned["escalation"] = str(report.get("escalation") or "Unknown").strip() or "Unknown"
    # Make escalation look clean (not a full runbook sentence)
    esc = cleaned["escalation"]
    esc_low = esc.lower()
    if esc_low.startswith("escalate to "):
        esc = esc[len("escalate to "):].strip()
    if " if " in esc.lower():
        esc = esc.split(" if ", 1)[0].strip()
    cleaned["escalation"] = esc or "Unknown"

    # ---- lists ----
    cleaned["what_we_know"] = _normalize_list(report.get("what_we_know"), max_items=6) or ["Unknown"]
    cleaned["what_we_dont_know"] = _normalize_list(report.get("what_we_dont_know"), max_items=5) or ["Unknown"]

    # ---- hypotheses ----
    hyps = _normalize_hypotheses(report.get("hypotheses"), max_items=4)
    cleaned["hypotheses"] = hyps or [
        {
            "hypothesis": "Unknown",
            "why": "Unknown",
            "how_to_validate": "Collect logs/metrics to validate a suspected cause.",
            "confidence": "Low",
        }
    ]

    # ---- plan ----
    plan = _normalize_plan(report.get("recommended_plan"))

    # Optional safety check (stronger):
    # Keep "if_unresolved" for escalation/deeper investigation, not normal mitigation.
    def looks_like_mitigation(line: str) -> bool:
        l = (line or "").lower()
        return any(k in l for k in ["roll back", "rollback", "retry", "disable", "restart", "clear cache"])

    moved: List[str] = []
    kept: List[str] = []
    for x in plan.get("if_unresolved", []):
        if looks_like_mitigation(x):
            moved.append(x)
        else:
            kept.append(x)

    plan["if_unresolved"] = kept

    # Move mitigation-ish lines into "next" instead
    for x in moved:
        if x not in plan["next"]:
            plan["next"].append(x)

    # If if_unresolved is empty after filtering, put a sane default
    if not plan["if_unresolved"]:
        # Prefer using the cleaned escalation target if it exists
        if cleaned["escalation"] != "Unknown":
            plan["if_unresolved"] = [f"Escalate to {cleaned['escalation']}."]
        else:
            plan["if_unresolved"] = ["Escalate to the appropriate on-call/owner and continue deeper investigation."]

    # If the whole plan is empty somehow, hard fallback
    if not (plan["now"] or plan["next"] or plan["if_unresolved"]):
        plan = {"now": ["Unknown"], "next": ["Unknown"], "if_unresolved": ["Unknown"]}

    cleaned["recommended_plan"] = plan

    # ---- sources ----
    cleaned["sources"] = sources

    return cleaned


def ai_manager_summary(det_summary: Dict[str, Any]) -> Dict[str, Any]:
    ready = bool(det_summary.get("ready", False))
    sources = det_summary.get("sources", []) or []

    if not ready or not sources:
        return {
            "ai_report": None,
            "used_sources": [],
            "refusal": "AI report unavailable: no citations/sources to ground the output.",
        }

    prompt = _build_prompt(det_summary)
    text = _ollama_generate(prompt)

    if not text:
        return {
            "ai_report": None,
            "used_sources": sources,
            "refusal": "AI unavailable: Ollama is not reachable. Start it with `ollama serve`.",
        }

    parsed = _extract_json(text)
    if not parsed:
        return {
            "ai_report": None,
            "used_sources": sources,
            "refusal": "AI output invalid: model did not return valid JSON.",
        }

    cleaned = _validate_and_fix_report(parsed, sources)
    if not cleaned:
        return {
            "ai_report": None,
            "used_sources": sources,
            "refusal": "AI output invalid: JSON did not match the required schema.",
        }

    return {
        "ai_report": cleaned,
        "used_sources": sources,
        "refusal": None,
    }