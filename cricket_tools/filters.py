import pandas as pd
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")

# ---------------------------------------------------------------------
# Canonical team equivalences
# ---------------------------------------------------------------------
TEAM_EQUIVALENTS = {
    # RCB dual names
    "Royal Challengers Bangalore": ["Royal Challengers Bengaluru"],
    "Royal Challengers Bengaluru": ["Royal Challengers Bangalore"],

    # Other historical renames
    "Delhi Daredevils": ["Delhi Capitals"],
    "Delhi Capitals": ["Delhi Daredevils"],

    "Kings XI Punjab": ["Punjab Kings"],
    "Punjab Kings": ["Kings XI Punjab"],

    # Pune variants in IPL history
    "Rising Pune Supergiant": ["Rising Pune Supergiants"],
    "Rising Pune Supergiants": ["Rising Pune Supergiant"],
}

def get_team_variants(team: str) -> list[str]:
    """Return a small set of equivalent team names (both directions)."""
    if not team:
        return []
    variants = {team}
    if team in TEAM_EQUIVALENTS:
        variants.update(TEAM_EQUIVALENTS[team])
    return sorted(variants)

# ---------------------------------------------------------------------
# Canonical city normalization map (for known equivalences)
# ---------------------------------------------------------------------
CITY_EQUIVALENTS = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
}

# ---------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load full IPL deliveries dataset and ensure correct dtypes."""
    df = pd.read_parquet(DATA_PATH)
    if df["date"].dtype != "datetime64[ns]":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
def filter_by_dates(df, start=None, end=None):
    """Filter DataFrame by start and end date."""
    # 🩹 FIX: drop invalid or missing dates first
    if "date" in df.columns:
        n_nan = df["date"].isna().sum()
        if n_nan:
            print(f"[WARN] Dropping {n_nan} rows with invalid dates")
            df = df[df["date"].notna()]

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df

def filter_by_teams(df, team=None):
    """Filter deliveries by team name (batting or bowling)."""
    if team:
        try:
            from cricket_tools.entity_matcher import normalize_entity
            normalized = normalize_entity(team, kind="team") or team  # 🩹 FIX
            team = normalized
        except Exception:
            pass

        variants = get_team_variants(team)
        mask = df["team_batting"].isin(variants) | df["team_bowling"].isin(variants)
        df = df[mask]
    return df

def filter_by_player(df, player_ds_name):
    """Return deliveries where player participated (batting or bowling)."""
    # 🩹 FIX: guard against empty player name
    if not player_ds_name:
        return df
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

    venue_norm = venue
    city_norm = city

    df["venue"] = df["venue"].astype(str)
    df["city"] = df["city"].astype(str)

    # --- Filter by venue ---
    if venue_norm:
        venue_clean = re.sub(r"[^a-z0-9]", "", venue_norm.lower())
        df["_venue_clean"] = df["venue"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        df = df[df["_venue_clean"].str.contains(venue_clean, na=False)]
        df = df.drop(columns=["_venue_clean"])

    # --- Filter by city ---
    if city_norm:
        city_lower = city_norm.strip().lower()
        if city_lower in {"bangalore", "bengaluru"}:
            city_pattern = r"(?:bangalore|bengaluru)"
        else:
            city_pattern = re.escape(city_lower)

        # 🩹 FIX: use word-bounded regex to avoid false positives
        df = df[df["city"].str.lower().str.contains(fr"\b{city_pattern}\b", na=False, regex=True)]

    return df

# ---------------------------------------------------------------------
# Unified filter pipeline
# ---------------------------------------------------------------------
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
    in a single unified call, with debug tracing.
    """
    print(f"[DEBUG] apply_filters() called with season={season}, start={start}, end={end}, team={team}, venue={venue}, city={city}")
    print(f"[DEBUG] Initial rows: {len(df)}")

    # 🩹 FIX: avoid double season/date filtering
    if season and not (start or end):
        df = df[df["season"].astype(str) == str(season)]
        print(f"[DEBUG] After season filter ({season}): {len(df)} rows")
    elif start or end:
        before = len(df)
        df = filter_by_dates(df, start, end)
        print(f"[DEBUG] After date filter: {len(df)} rows (removed {before - len(df)})")

    # --- Team filter ---
    if team:
        before = len(df)
        df = filter_by_teams(df, team)
        print(f"[DEBUG] After team filter ({team}): {len(df)} rows (removed {before - len(df)})")

    # --- Player filter ---
    if player:
        before = len(df)
        df = filter_by_player(df, player)
        print(f"[DEBUG] After player filter ({player}): {len(df)} rows (removed {before - len(df)})")

    # --- Location filter ---
    before = len(df)
    df = filter_by_location(df, venue, city)
    print(f"[DEBUG] After location filter: {len(df)} rows (removed {before - len(df)})")

    return df