from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from src.company_retrieve import retrieve_company_docs


# Ollama local server + model
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")


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
        "escalation": {"type": "string"},  # MUST be a TEAM name, not a sentence
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": REPORT_KEYS,
    "additionalProperties": False,
}


# --- "Company knowledge" allowlists (deterministic) ---
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


def _normalize_list(value: Any, max_items: int = 6) -> List[str]:
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

    def trim(s: str, n: int = 240) -> str:
        s = s.strip()
        return s if len(s) <= n else s[: n - 3].rstrip() + "..."

    for item in value:
        if not isinstance(item, dict):
            continue

        hyp = str(item.get("hypothesis") or "").strip()
        why = str(item.get("why") or "").strip()
        validate = str(item.get("how_to_validate") or "").strip()
        conf = str(item.get("confidence") or "").strip()

        if conf not in allowed_conf:
            conf = "Low"

        overconfident = ["definitely", "confirmed", "root cause is", "caused by"]
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
                "hypothesis": trim(hyp),
                "why": trim(why or "Unknown"),
                "how_to_validate": trim(validate),
                "confidence": conf,
            }
        )
        if len(out) >= max_items:
            break

    return out


def _allowed_escalation_teams(det: Dict[str, Any]) -> List[str]:
    key = det.get("key_details") or {}
    category = str(key.get("category") or "General Ops")
    systems = key.get("suspected_systems") or []

    allowed = set(ESCALATION_ALLOWLIST_BY_CATEGORY.get(category, []))
    for s in systems:
        for team in ESCALATION_ALLOWLIST_BY_SYSTEM.get(str(s), []):
            allowed.add(team)

    # safe fallback
    out = sorted(allowed) if allowed else ["IT Ops On-Call", "Engineering On-Call", "Operations Lead"]
    return out


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    s = text.strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": REPORT_SCHEMA,
        "options": {
            "temperature": 0.0,      # deterministic
            "top_p": 1.0,
            "repeat_penalty": 1.1,
            "num_predict": 520,      # room for hypotheses objects
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


def _build_prompt(det: Dict[str, Any]) -> str:
    summary = (det.get("summary") or "").strip()
    incident_facts = det.get("incident_facts", []) or []
    highlights = det.get("highlights", []) or []
    actions = det.get("recommended_actions", []) or []
    questions = det.get("questions_to_answer", []) or []
    sources = det.get("sources", []) or []

    allowed_teams = _allowed_escalation_teams(det)

    # Always include company docs as allowed facts (RAG v1)
    company_docs = retrieve_company_docs(
        ["TEAM_DIRECTORY", "SYSTEM_OWNERSHIP", "ESCALATION_POLICY", "INCIDENT_SEVERITY"],
        max_chars=900,
    )
    company_sources = [f"COMPANY:{d['doc_id']}" for d in company_docs]

    def bullets(items: List[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) if items else "- (none)"

    company_context = "\n\n".join(
        [f"[{d['doc_id']}]\n{d.get('excerpt','')}".strip() for d in company_docs]
    ).strip() or "(none)"

    schema_text = (
        "{\n"
        '  "executive_update": string,\n'
        '  "what_we_know": [string, ...],\n'
        '  "what_we_dont_know": [string, ...],\n'
        '  "hypotheses": [\n'
        '    {"hypothesis": string, "why": string, "how_to_validate": string, "confidence": "Low|Medium|High"}, ...\n'
        "  ],\n"
        '  "recommended_plan": {"now": [string,...], "next": [string,...], "if_unresolved": [string,...]},\n'
        '  "escalation": string,\n'
        '  "sources": [string, ...]\n'
        "}"
    )

    # Combine sources: deterministic runbooks + company docs
    all_sources = sources + company_sources
    srcs = ", ".join(all_sources) if all_sources else "(none)"

    return (
        "You are a senior incident commander writing a reliable, business-useful incident report.\n\n"
        "CRITICAL RULES (do not break):\n"
        "1) Use ONLY the Allowed Facts below.\n"
        "2) Do NOT invent root cause, metrics, or timelines.\n"
        "3) Hypotheses are allowed BUT must be clearly labeled as hypotheses and include validation steps.\n"
        "4) Output MUST be valid JSON and MUST match the schema exactly.\n"
        "5) Do NOT include any text outside the JSON.\n"
        "6) escalation MUST be a TEAM NAME chosen from the Allowed escalation teams list.\n\n"
        f"Schema:\n{schema_text}\n\n"
        "Allowed Facts:\n"
        f"- Deterministic summary: {summary}\n"
        f"- Incident facts:\n{bullets(incident_facts)}\n"
        f"- Runbook highlights:\n{bullets(highlights)}\n"
        f"- Recommended actions:\n{bullets(actions)}\n"
        f"- Open questions:\n{bullets(questions)}\n"
        f"- Company context:\n{company_context}\n"
        f"- Allowed escalation teams (choose ONE): {', '.join(allowed_teams)}\n"
        f"- Sources: {srcs}\n\n"
        "Guidance:\n"
        "- executive_update: 1–2 sentences for leadership.\n"
        "- what_we_know: 3–6 bullets derived from summary/facts/highlights.\n"
        "- what_we_dont_know: 2–5 bullets derived from open questions or missing context.\n"
        "- hypotheses: 2–4 plausible hypotheses consistent with Allowed Facts; include validation steps.\n"
        "- confidence should be Low/Medium/High depending on evidence strength.\n"
        "- recommended_plan must be grounded in the recommended actions/runbook highlights.\n"
        "- escalation must be a TEAM NAME only (no sentence).\n"
    )


def _validate_and_fix_report(report: Dict[str, Any], sources: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None

    if sorted(report.keys()) != sorted(REPORT_KEYS):
        return None

    cleaned: Dict[str, Any] = {}

    cleaned["executive_update"] = str(report.get("executive_update") or "Unknown").strip() or "Unknown"

    # Escalation cleaning (team name only)
    esc = str(report.get("escalation") or "").strip()
    esc_low = esc.lower()

    # Remove common runbook sentence wrappers
    if esc_low.startswith("escalate to "):
        esc = esc[len("escalate to ") :].strip()

    # Remove conditional clause
    if " if " in esc_low:
        esc = esc.split(" if ", 1)[0].strip()

    # Remove trailing punctuation
    esc = esc.strip(" .:-")

    cleaned["escalation"] = esc or "Unknown"

    cleaned["what_we_know"] = _normalize_list(report.get("what_we_know"), max_items=6) or ["Unknown"]
    cleaned["what_we_dont_know"] = _normalize_list(report.get("what_we_dont_know"), max_items=5) or ["Unknown"]

    hyps = _normalize_hypotheses(report.get("hypotheses"), max_items=4)
    cleaned["hypotheses"] = hyps or [
        {
            "hypothesis": "Unknown",
            "why": "Unknown",
            "how_to_validate": "Collect logs/metrics to validate a suspected cause.",
            "confidence": "Low",
        }
    ]

    plan = _normalize_plan(report.get("recommended_plan"))
    if not (plan["now"] or plan["next"] or plan["if_unresolved"]):
        plan = {"now": ["Unknown"], "next": ["Unknown"], "if_unresolved": ["Unknown"]}
    cleaned["recommended_plan"] = plan

    # Force sources to deterministic sources (runbooks + company docs)
    cleaned["sources"] = sources

    return cleaned


def ai_manager_summary(det_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI incident report (validated JSON) grounded by citations + company docs.

    Returns:
      {
        "ai_report": dict | None,
        "used_sources": [ids],
        "refusal": str | None
      }
    """
    # Sources from deterministic layer (runbooks)
    sources = det_summary.get("sources", []) or []

    # Company sources always available (if files exist)
    company_docs = retrieve_company_docs(
        ["TEAM_DIRECTORY", "SYSTEM_OWNERSHIP", "ESCALATION_POLICY", "INCIDENT_SEVERITY"],
        max_chars=400,
    )
    company_sources = [f"COMPANY:{d['doc_id']}" for d in company_docs]

    # Ready rule: allow AI if either runbooks OR company docs exist
    ready = bool(sources or company_sources)

    if not ready:
        return {
            "ai_report": None,
            "used_sources": [],
            "refusal": "AI report unavailable: no sources (runbooks or company docs) available to ground the output.",
        }

    prompt = _build_prompt(det_summary)
    text = _ollama_generate(prompt)

    used_sources = sources + company_sources

    if not text:
        return {
            "ai_report": None,
            "used_sources": used_sources,
            "refusal": "AI unavailable: Ollama is not reachable. Start it with `ollama serve`.",
        }

    parsed = _extract_json(text)
    if not parsed:
        return {
            "ai_report": None,
            "used_sources": used_sources,
            "refusal": "AI output invalid: model did not return valid JSON.",
        }

    cleaned = _validate_and_fix_report(parsed, used_sources)
    if not cleaned:
        return {
            "ai_report": None,
            "used_sources": used_sources,
            "refusal": "AI output invalid: JSON did not match the required schema.",
        }

    return {
        "ai_report": cleaned,
        "used_sources": used_sources,
        "refusal": None,
    }