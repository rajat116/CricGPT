#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_formatter.py — Universal Natural-Language Formatter

Handles ALL:
  - Ambiguous names → LLM clarification
  - Fallback → direct answer already natural
  - Structured IPL stats → STRICT summarization
  - RAW analytics tools (momentum, H2H, partnerships, pressure…) → NL summary
  - Comparison results → NL summary
  - Top lists → NL summary
"""

from __future__ import annotations
from typing import Dict, Any
import json
from .config import get_llm_client


# ------------------------------------------------------------
# LLM helper
# ------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    provider, model, client = get_llm_client()

    try:
        if provider == "gemini":
            obj = client.GenerativeModel(model)
            r = obj.generate_content(prompt)
            return r.text.strip()

        elif provider == "ollama":
            resp = client.post(
                f"{client.base_url}/api/generate",
                json={"model": model, "prompt": prompt},
                timeout=60
            )
            return resp.json().get("response", "").strip()

        else:  # OpenAI/compatible
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return r.choices[0].message.content.strip()

    except Exception:
        return ""


# ------------------------------------------------------------
# MAIN FORMATTER
# ------------------------------------------------------------
def format_natural_answer(question: str, result: Dict[str, Any]) -> str:
    """
    Fully universal NL formatter.
    """

    # --------------------------------------------------------
    # 0️⃣ AMBIGUOUS → use LLM to rewrite as friendly clarification
    # --------------------------------------------------------
    if result.get("status") == "ambiguous":
        query = result.get("query") or "this name"
        options = result.get("options", [])
        js = json.dumps({"query": query, "options": options}, indent=2)

        prompt = f"""
You are an IPL cricket chatbot assistant.

A user asked about "{query}", but the name is ambiguous.

Rewrite a friendly clarification message listing ONLY the provided options.

RULES:
- DO NOT add any new player names.
- DO NOT invent statistics.
- Only rephrase the clarification naturally.
- Keep it short and conversational.

Data:
{js}

Write the message:
"""
        text = _call_llm(prompt)
        if text:
            return text.strip()

        # deterministic fallback
        out = f"I found multiple IPL players named '{query}':\n"
        for o in options:
            out += f"• {o}\n"
        out += "\nWhich one did you mean?"
        return out

    # --------------------------------------------------------
    # 1️⃣ FALLBACK → answer already natural
    # --------------------------------------------------------
    action = (result.get("action") or "").lower()
    if action.startswith("llm_fallback"):
        return result.get("answer") or "(No fallback answer available.)"

    # --------------------------------------------------------
    # NLP summaries for raw analytics tools
    # --------------------------------------------------------

    def _nl_h2h(res):
        a = res.get("teamA", "Team A")
        b = res.get("teamB", "Team B")
        rec = res.get("records", {})

        m = rec.get("matches", "unavailable")
        wA = rec.get("winsA", "unavailable")
        wB = rec.get("winsB", "unavailable")

        return (
            f"Head-to-head between **{a}** and **{b}**:\n"
            f"- Total matches: **{m}**\n"
            f"- {a} wins: **{wA}**\n"
            f"- {b} wins: **{wB}**\n"
        )


    def _nl_momentum(res):
        return (
            f"Momentum worm chart for **{res.get('teamA')} vs {res.get('teamB')}**.\n"
            "It shows over-by-over scoring progression for both teams."
        )


    def _nl_phase(res):
        phases = ", ".join(res.get("phases", {}).keys())
        return (
            f"Phase-wise dominance analysis for **{res.get('teamA')} vs {res.get('teamB')}**.\n"
            f"Available phases: {phases}"
        )


    def _nl_threat(res):
        return (
            f"Bowling threat map for **{res.get('team')}**.\n"
            "It highlights wicket probability and pressure across phases."
        )


    def _nl_partnership(res):
        if "error" in res:
            return res["error"]

        match_id = res.get("match_id")
        count = len(res.get("nodes", []))
        return (
            f"Partnership graph for match **{match_id}**.\n"
            f"Total batting pairs: **{count}**."
        )


    def _nl_pressure(res):
        overs = len(res.get("over", []))
        return (
            f"Pressure progression across **{overs} overs** for match **{res.get('match_id')}**."
        )


    def _nl_winprob(res):
        overs = len(res.get("over", []))
        return (
            f"Win-probability timeline across **{overs} overs** for match **{res.get('match_id')}**."
        )


    RAW_NL = {
        "team_head_to_head": _nl_h2h,
        "team_momentum": _nl_momentum,
        "team_phase_dominance": _nl_phase,
        "team_threat_map": _nl_threat,
        "partnership_graph": _nl_partnership,
        "pressure_series": _nl_pressure,
        "win_probability_series": _nl_winprob,
    }

    # --------------------------------------------------------
    # 1.5️⃣ RAW ANALYTICS TOOLS → SKIP LLM (avoid JSON crash)
    # --------------------------------------------------------
    RAW_DATA_TOOLS = {
        "team_head_to_head",
        "team_momentum",
        "team_phase_dominance",
        "team_threat_map",
        "partnership_graph",
        "pressure_series",
        "win_probability_series",
    }

    # --------------------------------------------------------
    # Use our natural-language summaries for raw analytics tools
    # --------------------------------------------------------
    if action in RAW_NL:
        try:
            # These tools store their data inside "data", except H2H & partnership
            data = result.get("data", result)
            return RAW_NL[action](data)
        except Exception:
            return f"Here is the {action.replace('_',' ')} data."

    # --------------------------------------------------------
    # 2️⃣ STRUCTURED OUTPUT (ALL TOOL TYPES)
    #    We pass WHOLE result to LLM for universal summarization
    # --------------------------------------------------------
    prompt = f"""
You are an IPL cricket analytics assistant.

Convert the following structured result into a clean natural-language explanation.

Rules:
- Use EXACT numeric values without changing them.
- DO NOT invent matches, facts, or numbers.
- If something is missing / empty, say "unavailable".
- You may summarize series/curves graphically (e.g., "Win probability rose steadily after over 10").
- Keep the answer IPL-specific and concise.
- Be friendly and clear.
- NEVER hallucinate.

User question:
{question}

Structured result:
{json.dumps(result, indent=2)}

Write the natural-language answer:
"""
    text = _call_llm(prompt)
    if text:
        return text.strip()

    return "(Unable to generate natural-language answer.)"