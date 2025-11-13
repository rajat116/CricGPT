#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
runner.py — Phase-2 Unified Backend Layer

This file exposes a single universal function:
    run_query(message, backend="llm_reasoning", plot=False, fallback=True, session_id=None)

It handles:
    - planner → tool execution
    - memory usage
    - natural language formatting (via llm_formatter)
    - optional plotting
    - stable return object usable by CLI / FastAPI / Streamlit
"""

from __future__ import annotations

import uuid
from typing import Optional, Dict, Any

from .agent import CricketAgent
from .llm_formatter import format_natural_answer
from . import visuals


# --------------------------------------------------------------------
# Unified backend interface
# --------------------------------------------------------------------
def run_query(
    message: str,
    backend: str = "llm_reasoning",
    plot: bool = False,
    fallback: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Universal orchestration layer for CLI/API/UI.

    Returns a stable structure:
    {
        "reply": "... natural text ...",
        "act": "... tool name ...",
        "result": {... structured tool output ...},
        "plot_path": "... or None ...",
        "meta": {},
        "trace": [...],
        "session_id": "..."
    }
    """

    # 1. Create agent with selected backend
    agent = CricketAgent(backend=backend)
    agent.fallback_enabled = fallback
    agent.natural_language = False  # We apply NL ourselves here (centralized)

    # 2. Execute agent pipeline
    out = agent.run(message)
    act = out.get("action")
    result = out.get("result") or out.get("data") or {}

    # 3. Natural-language formatter (Phase-1)
    reply = format_natural_answer(message, out)

    # 4. Optional plot generation
    plot_path = None
    if plot:
        try:
            plot_path = visuals.auto_plot_from_result(result, act)
        except Exception:
            plot_path = None

    # 5. Stable session id
    if not session_id:
        session_id = str(uuid.uuid4())

    # 6. Final payload for UI/API
    return {
        "reply": reply,
        "act": act,
        "result": result,
        "plot_path": plot_path,
        "meta": {},
        "trace": out.get("trace", []),
        "session_id": session_id,
    }