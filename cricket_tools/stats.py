import pandas as pd
from functools import lru_cache
from cricket_tools.smart_names import resolve_player_smart
from cricket_tools.filters import load_dataset, apply_filters
from cricket_tools.normalization import normalize_entity  # 🧩 for canonical names

DATA_PATH = "data/processed/ipl_deliveries.parquet"


# ---------------------------------------------------------------------
# Helper — filter a player's data by date and team
# ---------------------------------------------------------------------
def prepare_filtered_data(player_ds, start=None, end=None, team=None):
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    if team:
        team_norm = normalize_entity(team, kind="team")
        df = df[(df["team_batting"] == team_norm) | (df["team_bowling"] == team_norm)]

    mask = (df["batsman"] == player_ds) | (df["bowler"] == player_ds)
    return df[mask]


# ---------------------------------------------------------------------
# 🏏 Individual Player Stats
# ---------------------------------------------------------------------
@lru_cache(maxsize=32)
def get_player_stats(name_query, start=None, end=None, team=None):
    ds_name, canon_name, status, hint = resolve_player_smart(name_query)
    if status not in ("ok", "confirm"):
        return {"query": name_query, "error": "player_not_found", "hint": hint}

    df = prepare_filtered_data(ds_name, start, end, team)
    if df.empty:
        return {"player": canon_name, "innings": 0, "note": "no data in this range"}

    bat_df = df[df["batsman"] == ds_name]
    if bat_df.empty:
        return {"player": canon_name, "innings": 0, "note": "no batting data"}

    runs = bat_df["runs_batter"].sum()
    balls = len(bat_df)
    dismissals = bat_df["wicket_player_out"].eq(ds_name).sum()
    innings = bat_df["match_id"].nunique()
    avg = runs / dismissals if dismissals else None
    strike_rate = (runs / balls * 100) if balls else 0

    return {
        "player": canon_name,
        "dataset_name": ds_name,
        "innings": int(innings),
        "runs": int(runs),
        "balls": int(balls),
        "avg": round(avg, 2) if avg else "∞",
        "strike_rate": round(strike_rate, 2),
        "dismissals": int(dismissals),
        "period": f"{start or 'start'}–{end or 'end'}",
        "team_filter": team or "any",
    }


@lru_cache(maxsize=32)
def get_bowler_stats(name_query, start=None, end=None, team=None):
    ds_name, canon_name, status, hint = resolve_player_smart(name_query)
    if status not in ("ok", "confirm"):
        return {"query": name_query, "error": "player_not_found", "hint": hint}

    df = prepare_filtered_data(ds_name, start, end, team)
    if df.empty:
        return {"player": canon_name, "overs": 0, "note": "no data in this range"}

    bowl_df = df[df["bowler"] == ds_name]
    if bowl_df.empty:
        return {"player": canon_name, "overs": 0, "note": "no bowling data"}

    runs_conceded = bowl_df["runs_total"].sum()
    wickets = bowl_df["wicket_player_out"].notna().sum()
    overs = len(bowl_df) / 6
    economy = runs_conceded / overs if overs else 0

    return {
        "player": canon_name,
        "dataset_name": ds_name,
        "overs": round(overs, 1),
        "runs_conceded": int(runs_conceded),
        "wickets": int(wickets),
        "economy": round(economy, 2),
        "period": f"{start or 'start'}–{end or 'end'}",
        "team_filter": team or "any",
        "rows": len(bowl_df),
    }


# ---------------------------------------------------------------------
# 🧩 Multi-Entity / Context-Aware Functions
# ---------------------------------------------------------------------
def compare_players(playerA, playerB, start=None, end=None, team=None, venue=None, city=None):
    """
    Compare two players' batting and bowling summary.
    - Uses smart_names to handle ambiguity.
    - If one player has no data, shows it but doesn't fail.
    - Returns consistent structure for agent consumption.
    """
    df = load_dataset()
    results = {}

    for player in [playerA, playerB]:
        ds_name, canon_name, status, hint = resolve_player_smart(player)

        # 🧩 Case 1: Player not confidently resolved
        if status not in ("ok", "confirm"):
            results[player] = {"error": "player_not_found", "hint": hint}
            continue

        # 🧩 Case 2: Player resolved but no records
        filtered = apply_filters(df, start=start, end=end, team=team, player=ds_name, venue=venue, city=city)
        if filtered.empty:
            results[canon_name] = {"note": "no data in selected filters"}
            continue

        # 🎯 Batting summary
        bat_df = filtered[filtered["batsman"] == ds_name]
        runs = bat_df["runs_batter"].sum()
        balls = len(bat_df)
        dismissals = bat_df["wicket_player_out"].eq(ds_name).sum()
        avg = runs / dismissals if dismissals else None
        sr = (runs / balls * 100) if balls else 0

        # 🎯 Bowling summary
        bowl_df = filtered[filtered["bowler"] == ds_name]
        wickets = bowl_df["wicket_player_out"].notna().sum()
        runs_conceded = bowl_df["runs_total"].sum()
        overs = len(bowl_df) / 6
        eco = runs_conceded / overs if overs else 0

        # ✅ Compact summary
        results[canon_name] = {
            "runs": int(runs),
            "avg": round(avg, 2) if avg else "∞",
            "sr": round(sr, 2),
            "wickets": int(wickets),
            "eco": round(eco, 2),
            "matches": int(filtered["match_id"].nunique()),
        }

    # 🧾 If only one player had valid data, make it clear
    valid_players = [k for k, v in results.items() if "note" not in v and "error" not in v]
    if len(valid_players) == 1:
        results["info"] = f"Data unavailable for one player; showing available stats for {valid_players[0]} only."

    return {
        "comparison": results,
        "filters": {"start": start, "end": end, "team": team, "venue": venue, "city": city},
    }

# ---------------------------------------------------------------------
# 🧩 Team Performance Summary
# ---------------------------------------------------------------------
def get_team_stats(team, start=None, end=None, season=None, venue=None, city=None):
    """Aggregate realistic team performance stats (1 row per match for wins)."""
    team_norm = normalize_entity(team, kind="team")

    df = load_dataset()
    df = apply_filters(
        df,
        start=start,
        end=end,
        season=season,
        team=team_norm,
        venue=venue,
        city=city,
    )

    if df.empty:
        return {"team": team_norm, "note": "no data"}

    # --- Basic aggregates ---
    total_runs = df[df["team_batting"] == team_norm]["runs_total"].sum()
    wickets = df[df["team_bowling"] == team_norm]["wicket_player_out"].notna().sum()
    matches = df["match_id"].nunique()

    # --- Compute wins correctly ---
    if "match_winner" in df.columns:
        # get 1 unique entry per match_id and winner
        winners = df[["match_id", "match_winner"]].drop_duplicates(subset=["match_id"])
        wins = winners["match_winner"].eq(team_norm).sum()
    else:
        wins = 0

    win_ratio = round((wins / matches * 100), 2) if matches else 0

    return {
        "team": team_norm,
        "matches": int(matches),
        "total_runs": int(total_runs),
        "wickets_taken": int(wickets),
        "wins": int(wins),
        "win_ratio": win_ratio,
        "filters": {
            "start": start,
            "end": end,
            "season": season,
            "venue": venue,
            "city": city,
        },
    }

# ---------------------------------------------------------------------
# 🧩 Top Player Aggregation
# ---------------------------------------------------------------------
def get_top_players(metric="runs_batter", season=None, n=5, venue=None, city=None, team=None):
    """
    Return top-N players by metric (runs_batter or wickets) with optional
    season, venue, city, and team filters.
    Auto-detects wicket columns like 'wicket_player_out' and counts them.
    """
    df = load_dataset()
    team_norm = normalize_entity(team, kind="team") if team else None

    # Apply all filters
    df = apply_filters(df, season=season, team=team_norm, venue=venue, city=city)
    if df.empty:
        return {"metric": metric, "note": "no data for given filters"}

    # --- Handle runs ---
    if metric in ["runs", "runs_batter", "batsman_runs"]:
        metric_col = next((c for c in ["runs_batter", "batsman_runs", "runs"]
                           if c in df.columns), None)
        if not metric_col:
            return {"metric": metric, "note": "no data for given filters"}
        grouped = (
            df.groupby("batsman")[metric_col]
              .sum()
              .sort_values(ascending=False)
              .head(n)
        )
        results = [{"player": p, metric_col: int(v)} for p, v in grouped.items()]

    # --- Handle wickets ---
    elif metric in ["wickets", "wicket_player_out"]:
        if "bowler" not in df.columns or "wicket_player_out" not in df.columns:
            return {"metric": metric, "note": "no wicket data available"}

        grouped = (
            df[df["wicket_player_out"].notna()]
            .groupby("bowler")["wicket_player_out"]
            .count()
            .sort_values(ascending=False)
            .head(n)
        )
        results = [{"player": p, "wickets": int(v)} for p, v in grouped.items()]

    else:
        return {"metric": metric, "note": "unsupported metric"}

    return {
        "metric": metric,
        "season": season,
        "venue": venue,
        "city": city,
        "team": team_norm,
        "top_players": results,
    }