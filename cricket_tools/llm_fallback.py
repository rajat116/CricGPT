"""
llm_fallback.py — Structured fallback for each query type (Step-8)
"""

from __future__ import annotations
import json, re, logging
from typing import Dict, Any
from .config import get_llm_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("LLMFallback")

# ---------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    try:
        provider, model, client = get_llm_client()
        if provider == "gemini":
            model_obj = client.GenerativeModel(model)
            resp = model_obj.generate_content(prompt)
            return resp.text.strip()
        elif provider == "ollama":
            url = f"{client.base_url}/api/generate"
            payload = {"model": model, "prompt": prompt}
            r = client.post(url, json=payload, timeout=60)
            return r.json().get("response", "").strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return "(LLM unavailable)"

# ---------------------------------------------------------------------
def llm_fallback_answer(question: str, trace: list | None = None, action: str | None = None) -> Dict[str, Any]:
    """
    Structured LLM fallback: emits JSON in schema consistent with intended tool/action.
    """
    trace = trace or []
    act = (action or "").lower()

    # --- 1️⃣ Choose schema template based on action ---
    if "team" in act:
        schema = {
            "team": "<team name>",
            "matches": "<int>",
            "wins": "<int>",
            "losses": "<int>",
            "run_rate": "<float>",
            "top_scorer": "<name>",
            "comment": "<short summary>",
        }
        task = "Provide IPL team performance summary in JSON."
    elif "compare" in act:
        schema = {
            "playerA": {"name": "<str>", "runs": "<int>", "avg": "<float>", "strike_rate": "<float>"},
            "playerB": {"name": "<str>", "runs": "<int>", "avg": "<float>", "strike_rate": "<float>"},
            "comparison": "<short text>",
        }
        task = "Compare two IPL players' batting performance."
    elif "top" in act:
        schema = {
            "metric": "<runs_batter or wickets>",
            "season": "<year>",
            "players": [{"name": "<str>", "value": "<float>"}],
            "comment": "<short note>",
        }
        task = "List top IPL players for the given metric."
    else:
        # default = player stats
        schema = {
            "player": "<name>",
            "matches": "<int>",
            "runs": "<int>",
            "avg": "<float>",
            "strike_rate": "<float>",
            "fours": "<int>",
            "sixes": "<int>",
            "wickets": "<int>",
            "economy": "<float>",
            "comment": "<short summary>",
        }
        task = "Provide IPL player performance summary."

    # --- 2️⃣ LLM prompt ---
    prompt = f"""
You are a cricket analytics assistant. Answer the following query about IPL cricket.

Always respond with ONLY a valid JSON (no markdown fences, no prose) 
matching this schema:
{json.dumps(schema, indent=2)}

If any field is unknown, set it to null (not empty string).

Question: {question}
"""

    text = _call_llm(prompt)

    # --- 3️⃣ Parse clean JSON ---
    clean = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"```$", "", clean).strip("`\n ")
    try:
        parsed = json.loads(clean)
    except Exception as e:
        log.warning(f"Invalid fallback JSON: {e}\n{text}")
        parsed = {"comment": text[:200]}

    result = {
        "status": "final",
        "action": f"llm_fallback_{act or 'generic'}",
        "args": {"topic": question},
        "data": parsed,
        "answer": parsed.get("comment", text),
        "reasoning": f"Structured fallback for action={act or 'generic'}",
        "scope": "General",
        "trace": trace + [{
            "action": f"llm_fallback_{act or 'generic'}",
            "args": {"topic": question},
            "result": parsed,
            "thought": f"Structured fallback schema used for {act or 'generic'}"
        }],
    }
    return result
