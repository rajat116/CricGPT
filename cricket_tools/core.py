"""
core.py — central interface layer for all cricket analytics operations.
Acts as the single gateway between user queries and backend logic.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from .smart_names import resolve_player_smart
from .stats import (
    get_player_stats,
    get_bowler_stats,
    compare_players,        # 🆕 Step-6
    get_team_stats,         # 🆕 Step-6
    get_top_players,        # 🆕 Step-6
)
from .ml_model import predict_future_performance
from .ml_build import build_ml_features
from .entity_matcher import normalize_entity


# ---------------------------------------------------------------------
# 🧩 Entity resolution (NEW)
# ---------------------------------------------------------------------
try:
    from .entity_matcher import EntityMatcher
    _ENTITY_MATCHER = EntityMatcher.from_dataset()
except Exception as _e:
    _ENTITY_MATCHER = None
    print(f"⚠️ EntityMatcher init failed: {_e}")

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

# 🆕 Step-6: new registry for non-player analytics
_EXTENDED_ACTIONS = {
    "compare": compare_players,
    "team": get_team_stats,
    "top": get_top_players,
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
    Unified interface for all cricket analytics.
    role:
      'batter' | 'bowler' | 'predict'  (player-centric)
      'compare' | 'team' | 'top'       (multi-entity / context-aware)
    """

    # ----------------------------------------------------------
    # 🧩 1. Handle extended analytics modes first (no name resolve)
    # ----------------------------------------------------------
    if role in _EXTENDED_ACTIONS:
        handler = _EXTENDED_ACTIONS[role]

        try:
            import inspect
            sig = inspect.signature(handler)
            call_kwargs = {}

            # 🧠 Resolve fuzzy entities (team / venue / city) if available (robust)
            if _ENTITY_MATCHER:
                for ent_key in ("team", "venue", "city"):
                    if ent_key in kwargs and kwargs[ent_key]:
                        try:
                            resolved = _ENTITY_MATCHER.resolve(kwargs[ent_key], etype=ent_key, top_k=1)
                            if resolved and isinstance(resolved, list) and "name" in resolved[0]:
                                kwargs[ent_key] = resolved[0]["name"]
                        except Exception:
                            pass

            # Build call kwargs dynamically for flexibility
            for key in ("start", "end", "team", "venue", "city", "season", "metric", "n", "playerA", "playerB"):
                if key in sig.parameters and key in kwargs:
                    call_kwargs[key] = kwargs[key]

            result = handler(**call_kwargs)
            return {
                "status": "ok",
                "role": role,
                "query": query,
                "period": _format_period(start, end),
                "data": result,
            }
        except Exception as e:
            return {"status": "error", "role": role, "message": str(e)}

    # ----------------------------------------------------------
    # 🎯 2. Existing player-centric roles (unchanged)
    # ----------------------------------------------------------
    ds_name, canon_name, status, hint = resolve_player_smart(query)

    # ---- Exact or high-confidence match ----
    if status == "ok":
        handler = _ROLE_REGISTRY.get(role)
        if handler:
            import inspect
            sig = inspect.signature(handler)
            # 🩹 FIX: only pass arguments that exist in handler signature
            base = {"start": start, "end": end}
            call_kwargs = {k: v for k, v in {**base, **kwargs}.items() if k in sig.parameters}
            data = handler(canon_name, **call_kwargs)
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

    # ---- Confirmation needed ----
    if status == "confirm":
        return {
            "status": "confirm",
            "query": query,
            "suggested": canon_name,
            "hint": hint,
            "role": role,
        }

    # ---- Ambiguous ----
    if status == "ambiguous":
        return {
            "status": "ambiguous",
            "query": query,
            "options": canon_name,
            "hint": hint,
            "role": role,
        }

    # ---- Not found ----
    return {
        "status": "not_found",
        "query": query,
        "hint": hint or "No player matched.",
        "role": role,
    }