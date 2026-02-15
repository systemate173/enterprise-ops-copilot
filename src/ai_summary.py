
from __future__ import annotations
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional


# Ollama local server + model
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")


def _build_prompt(det_summary: Dict[str, Any]) -> str:
    """
    Llama 3 does best with clear constraints and a concrete output format.
    We still return a Python dict (JSON-serializable) from our code; the model only returns text.
    """
    summary = (det_summary.get("summary") or "").strip()
    highlights = det_summary.get("highlights", []) or []
    sources = det_summary.get("sources", []) or []

    bullets = "\n".join([f"- {h}" for h in highlights]) if highlights else "- (none)"
    srcs = ", ".join(sources) if sources else "(none)"

    return (
        "You are an enterprise incident assistant.\n"
        "Task: Write a concise manager update.\n"
        "Rules:\n"
        "1) Use ONLY the provided summary + highlights + sources.\n"
        "2) Do NOT invent metrics, timelines, root cause, or impact.\n"
        "3) 1–2 sentences max.\n"
        "4) End with: 'Sources: <comma-separated ids>'\n\n"
        f"Deterministic summary: {summary}\n"
        f"Highlights:\n{bullets}\n"
        f"Sources available: {srcs}\n"
    )


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": 120,
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
        # IMPORTANT: don't crash the app/demo if Ollama isn't running
        return ""


def _postprocess(text: str, sources: List[str]) -> str:
    """
    Keep the output clean and enforce the 'Sources:' suffix.
    """
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # If model forgot sources line, append it deterministically
    if "Sources:" not in text:
        if sources:
            text = f"{text} Sources: {', '.join(sources)}"
        else:
            text = f"{text} Sources: (none)"
    return text

def ai_manager_summary(det_summary: Dict[str, Any]) -> Dict[str, Any]:
    ready = bool(det_summary.get("ready", False))
    sources = det_summary.get("sources", []) or []

    if not ready or not sources:
        return {
            "ai_summary": None,
            "used_sources": [],
            "refusal": "AI summary unavailable: no citations/sources to ground the output.",
        }

    prompt = _build_prompt(det_summary)
    text = _ollama_generate(prompt)

    if not text:
        return {
            "ai_summary": None,
            "used_sources": sources,
            "refusal": "AI unavailable: Ollama is not reachable. Start it with `ollama serve`.",
        }

    cleaned = _postprocess(text, sources)

    return {
        "ai_summary": cleaned,
        "used_sources": sources,
        "refusal": None,
    }

