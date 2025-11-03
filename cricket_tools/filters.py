# cricket_tools/filters.py
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")

def load_dataset() -> pd.DataFrame:
    """Load full IPL deliveries dataset."""
    return pd.read_parquet(DATA_PATH)

def filter_by_dates(df, start=None, end=None):
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df

def filter_by_teams(df, team=None):
    if team:
        mask = (df["batting_team"] == team) | (df["bowling_team"] == team)
        df = df[mask]
    return df

def filter_by_player(df, player_ds_name):
    """Return deliveries where player participated (batting or bowling)."""
    return df[(df["batsman"] == player_ds_name) | (df["bowler"] == player_ds_name)]

def prepare_filtered_data(player_ds_name, start=None, end=None, team=None):
    df = load_dataset()
    df = filter_by_dates(df, start, end)
    df = filter_by_teams(df, team)
    df = filter_by_player(df, player_ds_name)
    return df