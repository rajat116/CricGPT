import pandas as pd
from functools import lru_cache
from cricket_tools.smart_names import resolve_player_smart
from cricket_tools.filters import load_dataset, apply_filters, get_team_variants
from cricket_tools.entity_matcher import normalize_entity

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
        # 🩹 FIX: normalize and match all historical aliases (Bangalore ↔ Bengaluru)
        try:
            team_norm = normalize_entity(team, kind="team") or team
        except Exception:
            team_norm = team
        from cricket_tools.filters import get_team_variants
        variants = set(get_team_variants(team_norm)) | {team_norm}
        df = df[df["team_batting"].isin(variants) | df["team_bowling"].isin(variants)]

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

    # 🩹 FIX: exclude wides/noballs when counting legal balls
    extras = bat_df["extras_type"].fillna("")
    legal_mask = ~extras.isin(["wides", "noballs"])
    balls = int(legal_mask.sum())

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

    # 🩹 FIX: only bowler-credited dismissals
    credited = {"bowled", "caught", "lbw", "stumped", "hit wicket"}
    wk_mask = bowl_df["wicket_player_out"].notna() & bowl_df["wicket_kind"].isin(credited)
    wickets = int(wk_mask.sum())

    # 🩹 FIX: count only legal balls for overs
    extras_b = bowl_df["extras_type"].fillna("")
    legal_mask_b = ~extras_b.isin(["wides", "noballs"])
    legal_balls = int(legal_mask_b.sum())
    overs = legal_balls / 6 if legal_balls else 0
    economy = (runs_conceded / overs) if overs else 0

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
    """
    df = load_dataset()
    results = {}

    for player in [playerA, playerB]:
        ds_name, canon_name, status, hint = resolve_player_smart(player)

        if status not in ("ok", "confirm"):
            results[player] = {"error": "player_not_found", "hint": hint}
            continue

        filtered = apply_filters(df, start=start, end=end, team=team, player=ds_name, venue=venue, city=city)
        if filtered.empty:
            results[canon_name] = {"note": "no data in selected filters"}
            continue

        # --- Batting summary ---
        bat_df = filtered[filtered["batsman"] == ds_name]
        runs = bat_df["runs_batter"].sum()

        # 🩹 FIX: legal balls only
        extras = bat_df["extras_type"].fillna("")
        legal_mask = ~extras.isin(["wides", "noballs"])
        balls = int(legal_mask.sum())

        dismissals = bat_df["wicket_player_out"].eq(ds_name).sum()
        avg = runs / dismissals if dismissals else None
        sr = (runs / balls * 100) if balls else 0

        # --- Bowling summary ---
        bowl_df = filtered[filtered["bowler"] == ds_name]
        runs_conceded = bowl_df["runs_total"].sum()

        # 🩹 FIX: legal overs + bowler-only wickets
        credited = {"bowled", "caught", "lbw", "stumped", "hit wicket"}
        wk_mask = bowl_df["wicket_player_out"].notna() & bowl_df["wicket_kind"].isin(credited)
        wickets = int(wk_mask.sum())

        extras_b = bowl_df["extras_type"].fillna("")
        legal_mask_b = ~extras_b.isin(["wides", "noballs"])
        legal_balls = int(legal_mask_b.sum())
        overs = legal_balls / 6 if legal_balls else 0
        eco = (runs_conceded / overs) if overs else 0

        results[canon_name] = {
            "runs": int(runs),
            "avg": round(avg, 2) if avg else "∞",
            "sr": round(sr, 2),
            "wickets": int(wickets),
            "eco": round(eco, 2),
            "matches": int(filtered["match_id"].nunique()),
        }

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
    """Aggregate realistic team performance stats (season/date/location aware)."""
    print(f"[DEBUG] get_team_stats called with team={team}, start={start}, end={end}, season={season}, venue={venue}, city={city}")

    team_norm = normalize_entity(team, kind="team")
    print(f"[DEBUG] normalize_entity → {team_norm}")

    variants = get_team_variants(team_norm)
    print(f"[DEBUG] Team variants → {variants}")

    df = load_dataset()
    print(f"[DEBUG] Dataset loaded: {len(df)} rows, seasons={sorted(df['season'].unique().tolist())[:3]} ... {sorted(df['season'].unique().tolist())[-3:]}")

    before = len(df)
    df = apply_filters(df, start=start, end=end, season=season, team=None, venue=venue, city=city)
    print(f"[DEBUG] After apply_filters: {len(df)} rows (removed {before - len(df)})")

    team_df = df[df["team_batting"].isin(variants) | df["team_bowling"].isin(variants)]
    print(f"[DEBUG] After restricting to team variants: {len(team_df)} rows")

    if team_df.empty:
        print("[DEBUG] team_df is EMPTY. Dumping quick check:")
        print("Unique teams in filtered df:", df["team_batting"].unique()[:10])
        print("Unique seasons in filtered df:", df["season"].unique()[:10])
        print("Date range:", df["date"].min(), "→", df["date"].max())
        return {"team": team_norm, "note": "no data"}

    total_runs = team_df[team_df["team_batting"].isin(variants)]["runs_total"].sum()
    wickets = team_df[team_df["team_bowling"].isin(variants)]["wicket_player_out"].notna().sum()
    matches = team_df["match_id"].nunique()

    if "match_winner" in team_df.columns:
        winners = team_df[["match_id", "match_winner"]].drop_duplicates(subset=["match_id"])
        wins = winners["match_winner"].isin(variants).sum()
    else:
        wins = 0

    win_ratio = round((wins / matches * 100), 2) if matches else 0
    print(f"[DEBUG] Computed aggregates: matches={matches}, wins={wins}, win_ratio={win_ratio}")

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
    """
    df = load_dataset()
    df = apply_filters(df, season=season, team=team, venue=venue, city=city)
    if df.empty:
        return {"metric": metric, "note": "no data for given filters"}

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

    elif metric in ["wickets", "wicket_player_out"]:
        if "bowler" not in df.columns or "wicket_player_out" not in df.columns:
            return {"metric": metric, "note": "no wicket data available"}

        # 🩹 FIX: only bowler-credited wickets
        credited = {"bowled", "caught", "lbw", "stumped", "hit wicket"}
        dfw = df[df["wicket_player_out"].notna() & df["wicket_kind"].isin(credited)]
        grouped = (
            dfw.groupby("bowler")["wicket_player_out"]
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
        "team": team,
        "top_players": results,
    }