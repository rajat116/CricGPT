# analytics.py — Data computation layer (no plotting)
# ---------------------------------------------------
# Safe to import from agent + visuals. Returns pandas DataFrames or dicts.
from __future__ import annotations
import numpy as np
import pandas as pd

from cricket_tools.filters import load_dataset, apply_filters, get_team_variants
from cricket_tools.entity_matcher import normalize_entity


# -------------------------------
# Helpers
# -------------------------------
def _phase_from_over(over: int) -> str:
    if over <= 6:
        return "Powerplay"
    elif over <= 15:
        return "Middle"
    return "Death"

def _find_match_id(teamA: str, teamB: str, season: str | int | None = None) -> str | None:
    """Return the first match_id between teamA and teamB in a given season."""
    df = load_dataset()
    df = apply_filters(df, season=season)
    varA = set(get_team_variants(normalize_entity(teamA, "team") or teamA))
    varB = set(get_team_variants(normalize_entity(teamB, "team") or teamB))
    sub = df[(df["team_batting"].isin(varA) & df["team_bowling"].isin(varB)) |
             (df["team_batting"].isin(varB) & df["team_bowling"].isin(varA))]
    if sub.empty:
        return None
    return sub["match_id"].iloc[0]

# ---------------------------------------------------------------------
# Team head-to-head matrix — win percentage by venue & season
# ---------------------------------------------------------------------
def team_head_to_head(teamA: str, teamB: str, start=None, end=None, season=None) -> dict:
    df = load_dataset()

    teamA_norm = normalize_entity(teamA, kind="team") or teamA
    teamB_norm = normalize_entity(teamB, kind="team") or teamB
    varA = set(get_team_variants(teamA_norm)) | {teamA_norm}
    varB = set(get_team_variants(teamB_norm)) | {teamB_norm}

    df = apply_filters(df, start=start, end=end, season=season)
    mask = (df["team_batting"].isin(varA) & df["team_bowling"].isin(varB)) | \
           (df["team_batting"].isin(varB) & df["team_bowling"].isin(varA))
    df = df[mask]
    if df.empty or "match_winner" not in df.columns:
        return {"teamA": teamA_norm, "teamB": teamB_norm, "records": []}

    winners = df[["match_id", "season", "venue", "match_winner"]].drop_duplicates(subset=["match_id"])
    g = winners.groupby(["season", "venue"])["match_winner"]
    winsA = g.apply(lambda s: s.isin(varA).sum())
    winsB = g.apply(lambda s: s.isin(varB).sum())
    merged = pd.concat([winsA, winsB], axis=1)
    merged.columns = ["wins_teamA", "wins_teamB"]
    merged["total"] = merged["wins_teamA"] + merged["wins_teamB"]
    merged = merged.reset_index()
    merged["win_ratio_teamA"] = merged.apply(
        lambda r: (r["wins_teamA"] / r["total"]) if r["total"] else np.nan, axis=1
    )
    recs = merged[["season", "venue", "win_ratio_teamA"]].to_dict(orient="records")
    return {"teamA": teamA_norm, "teamB": teamB_norm, "records": recs}


# ---------------------------------------------------------------------
# Momentum worm — averaged over-by-over cumulative runs
# ---------------------------------------------------------------------
def team_momentum(teamA: str, teamB: str, season=None) -> dict:
    df = load_dataset()
    df = apply_filters(df, season=season)

    teamA_norm = normalize_entity(teamA, kind="team") or teamA
    teamB_norm = normalize_entity(teamB, kind="team") or teamB
    varA = set(get_team_variants(teamA_norm)) | {teamA_norm}
    varB = set(get_team_variants(teamB_norm)) | {teamB_norm}

    sub = df[df["team_batting"].isin(varA | varB)].copy()
    if sub.empty:
        return {"teamA": teamA_norm, "teamB": teamB_norm, "data": []}

    g = (
        sub.groupby(["team_batting", "match_id", "over"])["runs_total"]
        .sum()
        .reset_index()
    )
    avg_over = (
        g.groupby(["team_batting", "over"])["runs_total"]
        .mean()
        .reset_index()
    )
    avg_over = avg_over.sort_values(["team_batting", "over"])
    avg_over["cumulative_runs"] = avg_over.groupby("team_batting")["runs_total"].cumsum()

    data = avg_over[["team_batting", "over", "cumulative_runs"]].to_dict(orient="records")
    return {"teamA": teamA_norm, "teamB": teamB_norm, "data": data}


# ---------------------------------------------------------------------
# Phase dominance — RPO in Powerplay/Middle/Death for both teams
# ---------------------------------------------------------------------
def team_phase_dominance(teamA: str, teamB: str, season=None) -> dict:
    df = load_dataset()
    df = apply_filters(df, season=season)

    teamA_norm = normalize_entity(teamA, kind="team") or teamA
    teamB_norm = normalize_entity(teamB, kind="team") or teamB
    varA = set(get_team_variants(teamA_norm)) | {teamA_norm}
    varB = set(get_team_variants(teamB_norm)) | {teamB_norm}

    sub = df[df["team_batting"].isin(varA | varB)].copy()
    if sub.empty:
        return {"teamA": teamA_norm, "teamB": teamB_norm, "phases": []}

    sub["phase"] = sub["over"].apply(_phase_from_over)
    phase_rpo = (
        sub.groupby(["team_batting", "phase"])["runs_total"]
        .mean()
        .reset_index(name="rpo")
    )

    return {"teamA": teamA_norm, "teamB": teamB_norm, "phases": phase_rpo.to_dict(orient="records")}


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------
def compute_phase(over: int) -> str:
    return _phase_from_over(int(over))


def venue_baseline(metric: str = "runs_per_over") -> pd.DataFrame:
    df = load_dataset()
    grouped = (
        df.groupby(["venue", "match_id"])["runs_total"]
        .sum()
        .reset_index()
    )
    per_venue = grouped.groupby("venue")["runs_total"].mean().reset_index()
    per_venue.rename(columns={"runs_total": "avg_runs"}, inplace=True)
    return per_venue


def player_variance(season=None) -> pd.DataFrame:
    df = load_dataset()
    df = apply_filters(df, season=season)
    bat = df.groupby(["batsman", "match_id", "over"])["runs_batter"].sum().reset_index()
    stats = bat.groupby("batsman")["runs_batter"].agg(["mean", "var", "count"]).reset_index()
    stats.rename(columns={"mean": "mean_rpo", "var": "var_rpo", "count": "overs"}, inplace=True)
    return stats


def partnership_graph(match_id) -> dict:
    df = load_dataset()
    df = df[df["match_id"] == match_id]
    if df.empty:
        return {"nodes": [], "edges": []}

    pairs = []
    for (match, team, innings_no, striker, non_striker), g in df.groupby(
        ["match_id", "team_batting", "innings_no", "batsman", "non_striker"]
    ):
        runs_together = g["runs_total"].sum()
        pairs.append({
            "source": striker,
            "target": non_striker,
            "weight": float(runs_together),
            "innings_no": innings_no,
            "team_batting": team
        })

    # Build nodes: include run totals for sizing if you wish
    player_runs = df.groupby("batsman")["runs_batter"].sum().to_dict()
    nodes = [{"id": name, "runs": float(player_runs.get(name, 0))} for name in player_runs]

    return {"nodes": nodes, "edges": pairs, "match_id": match_id}

def team_threat_map(team: str, season=None) -> pd.DataFrame:
    df = load_dataset()
    df = apply_filters(df, season=season)

    team_norm = normalize_entity(team, kind="team") or team
    variants = get_team_variants(team_norm)
    sub = df[df["team_bowling"].isin(variants)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["bowler", "phase", "wickets"])

    sub["phase"] = sub["over"].apply(_phase_from_over)
    credited = {"bowled", "caught", "lbw", "stumped", "hit wicket"}
    wk_mask = sub["wicket_player_out"].notna() & sub["wicket_kind"].isin(credited)
    sub = sub[wk_mask]
    return sub.groupby(["bowler", "phase"])["wicket_player_out"].count().reset_index(name="wickets")


'''def pressure_series(match_id: str) -> pd.DataFrame:
    df = load_dataset()
    df = df[df["match_id"] == match_id]
    if df.empty or "innings" not in df.columns:
        return pd.DataFrame(columns=["over", "current_rr", "required_rr"])

    innings2 = df[df["innings"] == 2]
    if innings2.empty:
        return pd.DataFrame(columns=["over", "current_rr", "required_rr"])

    total_target = df[df["innings"] == 1]["runs_total"].sum()
    over_sum = innings2.groupby("over")["runs_total"].sum().cumsum()
    overs = over_sum.index
    current_rr = over_sum / overs
    required_rr = (total_target - over_sum) / (20 - overs)
    return pd.DataFrame({"over": overs, "current_rr": current_rr, "required_rr": required_rr})'''


'''def win_probability_series(match_id: str) -> pd.DataFrame:
    df = load_dataset()
    df = df[df["match_id"] == match_id]
    if df.empty or "innings" not in df.columns:
        return pd.DataFrame(columns=["ball", "win_prob"])

    innings2 = df[df["innings"] == 2].copy().sort_values(["over", "ball_in_over"])
    innings2["cum_runs"] = innings2["runs_total"].cumsum()
    target = df[df["innings"] == 1]["runs_total"].sum()
    innings2["runs_req"] = target - innings2["cum_runs"]
    innings2["balls_left"] = (20 * 6) - (innings2["over"] * 6 + innings2["ball_in_over"])
    innings2["rrr"] = innings2["runs_req"] / (innings2["balls_left"] / 6)
    innings2["win_prob"] = 1 / (1 + np.exp(0.5 * (innings2["rrr"] - 8)))
    return innings2[["over", "ball_in_over", "rrr", "win_prob"]]'''

def pressure_series(match_id: str) -> pd.DataFrame:
    df = load_dataset()
    df = df[df["match_id"] == match_id]
    
    if df.empty or "innings_no" not in df.columns:
        return pd.DataFrame(columns=["over", "current_rr", "required_rr"])

    innings2 = df[df["innings_no"] == 2]
    if innings2.empty:
        return pd.DataFrame(columns=["over", "current_rr", "required_rr"])

    total_target = df[df["innings_no"] == 1]["runs_total"].sum()
    
    # ✅ Ensure overs start from 1, not 0
    innings2 = innings2[innings2["over"] >= 1]
    
    over_sum = (
        innings2.groupby("over")["runs_total"]
        .sum()
        .cumsum()
        .reset_index()
    )

    overs = over_sum["over"]
    cumulative_runs = over_sum["runs_total"]

    overs_safe = overs.replace(0, np.nan)
    current_rr = (cumulative_runs / overs_safe).ffill()

    overs_remaining = (20 - overs).replace(0, np.nan)
    required_rr = ((total_target - cumulative_runs) / overs_remaining).ffill()

    return pd.DataFrame({
        "over": overs,
        "current_rr": current_rr,
        "required_rr": required_rr
    })

def win_probability_series(match_id: str) -> pd.DataFrame:
    df = load_dataset()
    df = df[df["match_id"] == match_id]
    
    if df.empty or "innings_no" not in df.columns:
        return pd.DataFrame(columns=["ball", "win_prob"])

    innings2 = df[df["innings_no"] == 2].copy().sort_values(["over", "ball_in_over"])
    if innings2.empty:
        return pd.DataFrame(columns=["ball", "win_prob"])

    innings2["cum_runs"] = innings2["runs_total"].cumsum()
    target = df[df["innings_no"] == 1]["runs_total"].sum()
    
    innings2["runs_req"] = target - innings2["cum_runs"]
    innings2["balls_left"] = (20 * 6) - (innings2["over"] * 6 + innings2["ball_in_over"])
    
    # --- ⬇️ FINAL FIX ⬇️ ---
    # Apply the same .to_series() fix here
    balls_remaining_safe = (innings2["balls_left"] / 6).replace(0, np.nan)
    innings2["rrr"] = innings2["runs_req"] / balls_remaining_safe
    innings2["rrr"] = innings2["rrr"].fillna(method='ffill')
    # --- ⬆️ END OF FIX ⬆️ ---
    
    innings2["win_prob"] = 1 / (1 + np.exp(0.5 * (innings2["rrr"] - 8)))
    return innings2[["over", "ball_in_over", "rrr", "win_prob"]]

# ---------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------
def cumulative_runs(team: str, season=None, match_id=None) -> pd.DataFrame:
    """
    Legacy convenience to return per-match over aggregates;
    Use team_momentum for averaged cumulative lines across matches.
    """
    df = load_dataset()
    df = apply_filters(df, season=season, team=team)
    if match_id:
        df = df[df["match_id"] == match_id]
    if df.empty:
        return pd.DataFrame(columns=["over", "cumulative_runs"])

    team_norm = normalize_entity(team, kind="team") or team
    variants = get_team_variants(team_norm)
    team_df = df[df["team_batting"].isin(variants)]
    grouped = team_df.groupby(["match_id", "over"])["runs_total"].sum().reset_index()
    grouped["phase"] = grouped["over"].apply(_phase_from_over)
    grouped["cumulative_runs"] = grouped.groupby("match_id")["runs_total"].cumsum()
    return grouped[["match_id", "over", "cumulative_runs"]]


def phase_stats(teamA: str, teamB: str | None = None, season=None, match_id=None) -> dict:
    """
    Legacy generic phase stats; prefer team_phase_dominance for A vs B plots.
    """
    df = load_dataset()
    df = apply_filters(df, season=season)
    if match_id:
        df = df[df["match_id"] == match_id]
    if df.empty:
        return {}

    out = {}
    teams = [teamA] + ([teamB] if teamB else [])
    for team in teams:
        team_norm = normalize_entity(team, kind="team") or team
        variants = get_team_variants(team_norm)
        sub = df[df["team_batting"].isin(variants)].copy()
        if sub.empty:
            continue
        sub["phase"] = sub["over"].apply(_phase_from_over)
        stats = (
            sub.groupby("phase")
            .agg(runs=("runs_total", "sum"), overs=("over", "nunique"))
            .reset_index()
        )
        stats["rpo"] = stats["runs"] / stats["overs"]
        out[team_norm] = stats.to_dict(orient="records")
    return out


def team_h2h_table(teamA: str, teamB: str) -> pd.DataFrame:
    res = team_head_to_head(teamA, teamB)
    recs = res.get("records", [])
    if not recs:
        return pd.DataFrame(columns=["season", "venue", "win_pct"])
    df = pd.DataFrame(recs).rename(columns={"win_ratio_teamA": "win_pct"})
    return df