"""
cricket_tools/llm_fallback.py
=============================
Provider-agnostic LLM fallback + rephrasing utilities.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Dict, Any

# ✅ Add this block:
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import get_llm_client

log = logging.getLogger("LLMFallback")

# ---------------------------------------------------------------------
# 🧠 1. Generic LLM call helper
# ---------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    """
    Send a plain text prompt to whichever LLM provider is configured.
    Returns the raw text response (never raises).
    """
    try:
        provider, model, client = get_llm_client()
        if provider == "gemini":
            model_obj = client.GenerativeModel(model)
            resp = model_obj.generate_content(prompt)
            return resp.text.strip()

        elif provider == "ollama":
            import requests
            url = f"{client.base_url}/api/generate"
            payload = {"model": model, "prompt": prompt}
            r = client.post(url, json=payload, timeout=60)
            return r.json().get("response", "").strip()

        else:  # openai / default
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()

    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return "(LLM fallback unavailable)"

# ---------------------------------------------------------------------
# 🪄 2. Rephrase vague question (for display / retry)
# ---------------------------------------------------------------------
def llm_rephrase(question: str, context: Dict[str, Any] | None = None) -> str:
    """
    Ask the LLM to rephrase the user's question clearly,
    optionally enriching with current context.
    """
    ctx = ""
    if context:
        ctx = f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    prompt = (
        f"Rephrase this cricket-related question clearly and completely,"
        f" preserving meaning but adding missing specifics if possible.{ctx}\n\n"
        f"Question: {question}"
    )
    return _call_llm(prompt)

# ---------------------------------------------------------------------
# 💬 3. Fallback answer when no structured tool result
# ---------------------------------------------------------------------
def llm_fallback_answer(question: str, trace: list | None = None) -> Dict[str, Any]:
    """
    Ask the LLM directly to answer the cricket question.
    Returns a dict similar to other tool outputs:
      { 'status': 'final', 'answer': '<text>', 'trace': [...] }
    """
    trace = trace or []
    prompt = (
        "You are a cricket expert. Answer concisely but informatively.\n\n"
        f"User question: {question}"
    )
    text = _call_llm(prompt)
    trace.append({"action": "llm_fallback", "thought": "No structured tool matched", "answer": text})
    return {"status": "final", "answer": text, "trace": trace}

if __name__ == "__main__":
    q = "Who scored the most runs for India in World Cup 2023?"
    print(llm_fallback_answer(q))