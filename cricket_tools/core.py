"""
core.py — central interface layer for all cricket analytics operations.
Acts as the single gateway between user queries and backend logic.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from .smart_names import resolve_player_smart
from .stats import get_player_stats, get_bowler_stats
from .ml_model import predict_future_performance
from .ml_build import build_ml_features

# ---------------------------------------------------------------------
# Helper: normalize date inputs
# ---------------------------------------------------------------------
def _format_period(start: Optional[str], end: Optional[str]) -> str:
    s = start or "start"
    e = end or "end"
    return f"{s}–{e}"

# ---------------------------------------------------------------------
# Registry of roles and handlers
# ---------------------------------------------------------------------
_ROLE_REGISTRY = {
    "batter": get_player_stats,
    "bowler": get_bowler_stats,
    "predict": predict_future_performance,
}

# ---------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------
def cricket_query(
    query: str,
    role: str = "batter",
    start: Optional[str] = None,
    end: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified interface for all player queries.
    role: 'batter' | 'bowler' | 'predict' | future roles
    """

    ds_name, canon_name, status, hint = resolve_player_smart(query)

    # ---- 1️⃣ Exact or high-confidence match ----
    if status == "ok":
        handler = _ROLE_REGISTRY.get(role)
        if handler:
            data = handler(
                canon_name,
                dataset_name=ds_name,
                start=start,
                end=end,
                **kwargs,
            )
        else:
            data = {"error": f"Unknown role '{role}'."}

        return {
            "status": "ok",
            "player": canon_name,
            "dataset_name": ds_name,
            "period": _format_period(start, end),
            "role": role,
            "data": data,
        }

    # ---- 2️⃣ Confirmation needed ----
    if status == "confirm":
        return {
            "status": "confirm",
            "query": query,
            "suggested": canon_name,
            "hint": hint,
            "role": role,
        }

    # ---- 3️⃣ Ambiguous ----
    if status == "ambiguous":
        return {
            "status": "ambiguous",
            "query": query,
            "options": canon_name,
            "hint": hint,
            "role": role,
        }

    # ---- 4️⃣ Not found ----
    return {
        "status": "not_found",
        "query": query,
        "hint": hint or "No player matched.",
        "role": role,
    }