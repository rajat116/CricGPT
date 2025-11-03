"""
core.py — central interface layer for all cricket analytics operations.
This acts as the single gateway between user queries and backend logic.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from .smart_names import resolve_player_smart
from .stats import get_player_stats, get_bowler_stats

# ---------------------------------------------------------------------
# Helper: normalize date inputs
# ---------------------------------------------------------------------
def _format_period(start: Optional[str], end: Optional[str]) -> str:
    s = start or "start"
    e = end or "end"
    return f"{s}–{e}"

# ---------------------------------------------------------------------
# Main unified entry point
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
    role: 'batter' | 'bowler' | later: 'allrounder' | 'captain' | etc.
    """

    ds_name, canon_name, status, hint = resolve_player_smart(query)

    # ---- 1️⃣ If resolver found an exact or high-confidence match ----
    if status == "ok":
        if role == "batter":
            data = get_player_stats(canon_name, start=start, end=end, **kwargs)
        elif role == "bowler":
            data = get_bowler_stats(canon_name, start=start, end=end, **kwargs)
        else:
            # Future expansion: call combined models
            data = {}

        return {
            "status": "ok",
            "player": canon_name,
            "dataset_name": ds_name,
            "period": _format_period(start, end),
            "role": role,
            "data": data,
        }

    # ---- 2️⃣ If resolver suggests confirmation ----
    if status == "confirm":
        return {
            "status": "confirm",
            "query": query,
            "suggested": canon_name,
            "hint": hint,
            "role": role,
        }

    # ---- 3️⃣ If ambiguous (multiple possible names) ----
    if status == "ambiguous":
        return {
            "status": "ambiguous",
            "query": query,
            "options": canon_name,  # list of candidates
            "hint": hint,
            "role": role,
        }

    # ---- 4️⃣ If not found ----
    return {
        "status": "not_found",
        "query": query,
        "hint": hint or "No player matched.",
        "role": role,
    }

# ---------------------------------------------------------------------
# Example stub: future ML / prediction / fantasy integration
# ---------------------------------------------------------------------
def predict_future_performance(player: str, venue: Optional[str] = None) -> Dict[str, Any]:
    """
    Placeholder for Step 4 (ML layer). Will use trained model later.
    """
    return {
        "status": "todo",
        "player": player,
        "message": "ML prediction model not yet implemented.",
        "venue": venue,
    }