import pandas as pd
from cricket_tools.smart_names import resolve_player_smart

DATA_PATH = "data/processed/ipl_deliveries.parquet"


def prepare_filtered_data(player_ds, start=None, end=None, team=None):
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    if team:
        df = df[(df["team_batting"] == team) | (df["team_bowling"] == team)]

    mask = (df["batsman"] == player_ds) | (df["bowler"] == player_ds)
    return df[mask]


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