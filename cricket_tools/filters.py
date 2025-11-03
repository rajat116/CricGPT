# cricket_tools/filters.py
"""
Filter utilities for the processed IPL deliveries dataframe.
"""

import pandas as pd
from typing import Optional, List


def load_dataset(path: str = "data/processed/ipl_deliveries.parquet") -> pd.DataFrame:
    """Load cached IPL deliveries parquet file."""
    df = pd.read_parquet(path)
    # ensure datetime for filters
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def filter_by_player(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """All deliveries involving this player (as batsman)."""
    mask = df["batsman"].str.contains(player, case=False, na=False)
    return df[mask]


def filter_by_bowler(df: pd.DataFrame, bowler: str) -> pd.DataFrame:
    mask = df["bowler"].str.contains(bowler, case=False, na=False)
    return df[mask]


def filter_by_team(df: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (
        df["team_batting"].str.contains(team, case=False, na=False)
        | df["team_bowling"].str.contains(team, case=False, na=False)
    )
    return df[mask]


def filter_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter by inclusive date range (yyyy-mm-dd)."""
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df


def filter_by_venue(df: pd.DataFrame, venue: str) -> pd.DataFrame:
    mask = df["venue"].str.contains(venue, case=False, na=False)
    return df[mask]


def multi_filter(
    df: pd.DataFrame,
    player: Optional[str] = None,
    team: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    venue: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper for combined filters."""
    if player:
        df = filter_by_player(df, player)
    if team:
        df = filter_by_team(df, team)
    if start_date or end_date:
        df = filter_by_date(df, start_date, end_date)
    if venue:
        df = filter_by_venue(df, venue)
    return df
