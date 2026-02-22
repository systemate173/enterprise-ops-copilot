from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional


# Ollama local server + model
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")


# JSON keys we require from the model
REPORT_KEYS = [
    "executive_update",
    "what_we_know",
    "what_we_dont_know",
    "most_likely_causes",
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
        "most_likely_causes": {"type": "array", "items": {"type": "string"}},
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

def _normalize_plan(value: Any) -> Dict[str, List[str]]:
    empty = {"now": [], "next": [], "if_unresolved": []}
    if not isinstance(value, dict):
        return empty

    return {
        "now": _normalize_list(value.get("now"), max_items=5),
        "next": _normalize_list(value.get("next"), max_items=5),
        "if_unresolved": _normalize_list(value.get("if_unresolved"), max_items=5),
    }

def _build_prompt(det: Dict[str, Any]) -> str:
    summary = (det.get("summary") or "").strip()
    highlights = det.get("highlights", []) or []
    actions = det.get("recommended_actions", []) or []
    questions = det.get("questions_to_answer", []) or []
    escalation_targets = det.get("escalation_targets", []) or []
    sources = det.get("sources", []) or []

    def bullets(items: List[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) if items else "- (none)"

    srcs = ", ".join(sources) if sources else "(none)"

    schema = (
    "{\n"
    '  "executive_update": string,\n'
    '  "what_we_know": [string, ...],\n'
    '  "what_we_dont_know": [string, ...],\n'
    '  "most_likely_causes": [string, ...],\n'
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
        "3) If a field cannot be supported from Allowed Facts, write 'Unknown' (or ['Unknown']).\n"
        "4) Output MUST be valid JSON and MUST match the schema exactly.\n"
        "5) Do NOT include any text outside the JSON.\n\n"
        f"Schema:\n{schema}\n\n"
        "Allowed Facts:\n"
        f"- Deterministic summary: {summary}\n"
        f"- Runbook highlights:\n{bullets(highlights)}\n"
        f"- Recommended actions:\n{bullets(actions)}\n"
        f"- Open questions:\n{bullets(questions)}\n"
        f"- Escalation targets (from runbooks):\n{bullets(escalation_targets)}\n"
        f"- Sources: {srcs}\n\n"
        "Guidance:\n"
        "- Avoid repeating the deterministic summary verbatim; rephrase into clear business language.\n"
        "- executive_update: 1–2 sentences for leadership.\n"
        "- what_we_know: 3–6 bullets derived from summary/highlights.\n"
        "- what_we_dont_know: 2–5 bullets derived from open questions.\n"
        "- most_likely_causes: ONLY if explicitly supported; otherwise ['Unknown'].\n"
        "- recommended_plan MUST be an object with keys now/next/if_unresolved, each a list of strings.\n"
        "- escalation: If escalation targets exist, name them; otherwise 'Unknown'.\n"
        "- When unsure, say 'suspected' or 'possibly' instead of stating as fact.\n"
    )

def _ollama_generate(prompt: str) -> str:
    """
    Call local Ollama. Tuned for low-hallucination structured output.
    """
    payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json",  # keep simple JSON mode (schema optional for now)
    "options": {
        "temperature": 0.0,   # CRITICAL: deterministic output
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "num_predict": 350,
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
        # Do not crash the app if Ollama is down
        return ""

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    s = text.strip()

    # Try direct parse first
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Balanced-brace extraction: find first JSON object
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
                candidate = s[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None

def _normalize_list(value: Any, max_items: int = 5) -> List[str]:
    """
    Ensure lists are lists of compact strings.
    """
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
    # If model returned a string by mistake, wrap it
    s = str(value).strip()
    return [s] if s else []


def _validate_and_fix_report(report: Dict[str, Any], sources: List[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None

    if sorted(report.keys()) != sorted(REPORT_KEYS):
        return None

    cleaned: Dict[str, Any] = {}

    # Strings
    cleaned["executive_update"] = str(report.get("executive_update") or "Unknown").strip() or "Unknown"
    cleaned["escalation"] = str(report.get("escalation") or "Unknown").strip() or "Unknown"

    # Lists
    cleaned["what_we_know"] = _normalize_list(report.get("what_we_know"), max_items=6) or ["Unknown"]
    cleaned["what_we_dont_know"] = _normalize_list(report.get("what_we_dont_know"), max_items=5) or ["Unknown"]
    mlc = _normalize_list(report.get("most_likely_causes"), max_items=5) or ["Unknown"]
    # Hard rule: if it contains invented language like "likely" but no concrete support, force Unknown.
    if any("likely" in x.lower() for x in mlc):
        mlc = ["Unknown"]
    cleaned["most_likely_causes"] = mlc
    plan = _normalize_plan(report.get("recommended_plan"))
    if not (plan["now"] or plan["next"] or plan["if_unresolved"]):
        plan = {"now": ["Unknown"], "next": ["Unknown"], "if_unresolved": ["Unknown"]}
    cleaned["recommended_plan"] = plan


    # Force sources
    cleaned["sources"] = sources

    return cleaned


def ai_manager_summary(det_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Citation-locked AI incident report (validated JSON).

    Returns:
      {
        "ai_report": dict | None,   # validated report JSON
        "used_sources": [ids],
        "refusal": str | None
      }
    """
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
