# cricket_tools/stats.py
"""
Compute batting and bowling statistics from IPL deliveries dataframe.
"""

# cricket_tools/stats.py
import pandas as pd
from typing import Dict, Optional
from cricket_tools.filters import load_dataset, filter_by_date
from cricket_tools.names import resolve_player

# ----------------------------
# Batting statistics
# ----------------------------
def get_player_stats(
    player: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = "data/processed/ipl_deliveries.parquet",
) -> Dict:
    """Return basic batting stats for a player (name can be loose: 'Rohit', 'Kohli', etc.)."""
    df = load_dataset(path)
    # resolve to canonical name present in dataset (e.g., 'RG Sharma')
    resolved, candidates = resolve_player(player, role="bat")
    if not resolved:
        return {
            "player": player,
            "resolved_to": None,
            "candidates": candidates,
            "innings": 0
        }

    df = df[df["batsman"].str.lower() == resolved.lower()]
    df = filter_by_date(df, start, end)
    if df.empty:
        return {
            "player": player,
            "resolved_to": resolved,
            "candidates": candidates,
            "innings": 0
        }

    runs = int(df["runs_batter"].sum())
    balls = int(df["runs_batter"].count())
    dismissals = int(df["wicket_player_out"].str.lower().eq(resolved.lower()).sum())
    sixes = int((df["runs_batter"] == 6).sum())
    fours = int((df["runs_batter"] == 4).sum())
    innings = int(df["match_id"].nunique())

    avg = round(runs / dismissals, 2) if dismissals > 0 else None
    sr = round((runs / balls) * 100, 2) if balls > 0 else 0.0

    return {
        "player": player,
        "resolved_to": resolved,
        "candidates": candidates[:10],
        "innings": innings,
        "runs": runs,
        "balls": balls,
        "fours": fours,
        "sixes": sixes,
        "average": avg,
        "strike_rate": sr,
        "dismissals": dismissals,
        "period": f"{start or 'start'}–{end or 'end'}",
    }


# ----------------------------
# Bowling statistics
# ----------------------------
def get_bowler_stats(
    bowler: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = "data/processed/ipl_deliveries.parquet",
) -> Dict:
    """Return bowling stats for a bowler (name can be loose: 'Bumrah', 'Jasprit', etc.)."""
    df = load_dataset(path)
    resolved, candidates = resolve_player(bowler, role="bowl")
    if not resolved:
        return {
            "bowler": bowler,
            "resolved_to": None,
            "candidates": candidates,
            "overs": 0
        }

    dff = df[df["bowler"].str.lower() == resolved.lower()]
    dff = filter_by_date(dff, start, end)
    if dff.empty:
        return {
            "bowler": bowler,
            "resolved_to": resolved,
            "candidates": candidates,
            "overs": 0
        }

    balls = int(len(dff))
    overs = round(balls / 6, 1) if balls else 0.0
    runs_conceded = int(dff["runs_total"].sum())
    wickets = int(dff["wicket_player_out"].notna().sum())

    economy = round(runs_conceded / overs, 2) if overs > 0 else 0.0
    bowling_sr = round(balls / wickets, 2) if wickets > 0 else None

    return {
        "bowler": bowler,
        "resolved_to": resolved,
        "candidates": candidates[:10],
        "overs": overs,
        "runs_conceded": runs_conceded,
        "wickets": wickets,
        "economy": economy,
        "strike_rate": bowling_sr,
        "period": f"{start or 'start'}–{end or 'end'}",
    }
