import pandas as pd
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from .normalization import normalize_entity
import re

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")

# ---------------------------------------------------------------------
# Canonical city normalization map (for known equivalences)
# ---------------------------------------------------------------------
CITY_EQUIVALENTS = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",  # ensure both point to canonical
}

@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load full IPL deliveries dataset and ensure correct dtypes."""
    df = pd.read_parquet(DATA_PATH)
    if df["date"].dtype != "datetime64[ns]":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def filter_by_dates(df, start=None, end=None):
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df

def filter_by_teams(df, team=None):
    if team:
        mask = (df["team_batting"] == team) | (df["team_bowling"] == team)
        df = df[mask]
    return df

def filter_by_player(df, player_ds_name):
    """Return deliveries where player participated (batting or bowling)."""
    return df[(df["batsman"] == player_ds_name) | (df["bowler"] == player_ds_name)]

def prepare_filtered_data(player_ds_name, start=None, end=None, team=None):
    """Existing helper used by core modules."""
    df = load_dataset()
    df = filter_by_dates(df, start, end)
    df = filter_by_teams(df, team)
    df = filter_by_player(df, player_ds_name)
    return df


# -------------------------------------------------------------------------
# 🆕 Step-6 Additions: Location- and Context-Aware Filtering
# -------------------------------------------------------------------------

def filter_by_location(df, venue=None, city=None):
    """
    Filter DataFrame by venue or city with canonical normalization and
    robust matching for known equivalences like Bangalore ↔ Bengaluru.
    """
    df = df.copy()  # ✅ prevents SettingWithCopyWarning
    # --- Normalize inputs ---
    venue_norm = normalize_entity(venue, kind="venue") if venue else None
    city_norm = normalize_entity(city, kind="city") if city else None

    # --- Ensure string columns (avoid NaN/None warnings) ---
    df["venue"] = df["venue"].astype(str)
    df["city"] = df["city"].astype(str)

    # --- Filter by venue ---
    if venue_norm:
        # Normalize for punctuation-insensitive matching
        venue_clean = re.sub(r"[^a-z0-9]", "", venue_norm.lower())
        df["_venue_clean"] = df["venue"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        df = df[df["_venue_clean"].str.contains(venue_clean, na=False)]
        df = df.drop(columns=["_venue_clean"])

    # --- Filter by city ---
    if city_norm:
        city_lower = city_norm.strip().lower()

        # Handle Bangalore/Bengaluru equivalence
        if city_lower in {"bangalore", "bengaluru"}:
            city_pattern = r"\b(?:bangalore|bengaluru)\b"
        else:
            city_pattern = re.escape(city_lower)

        df = df[df["city"].str.lower().str.contains(city_pattern, na=False, regex=True)]

    return df

def apply_filters(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    season: str | int | None = None,
    team: str | None = None,
    player: str | None = None,
    venue: str | None = None,
    city: str | None = None,
) -> pd.DataFrame:
    """
    Apply all supported filters (date, season, team, player, venue, city)
    in a single unified call.
    """
    # Date / season filters
    if season:
        df = df[df["season"].astype(str) == str(season)]
    df = filter_by_dates(df, start, end)

    # Team / player filters
    df = filter_by_teams(df, team)
    if player:
        df = filter_by_player(df, player)

    # Location filters
    df = filter_by_location(df, venue, city)

    return df