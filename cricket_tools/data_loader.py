# cricket_tools/data_loader.py
"""
Parse Cricsheet IPL YAML files and flatten deliveries into a DataFrame.

Usage:
    python -m cricket_tools.data_loader
"""

import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/ipl")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "ipl_deliveries.parquet"


def parse_match(file_path: Path) -> pd.DataFrame:
    """Parse one IPL match YAML → DataFrame of deliveries."""
    with open(file_path, "r") as f:
        match = yaml.safe_load(f)

    info = match.get("info", {})
    innings_list = match.get("innings", [])
    if not innings_list:
        return pd.DataFrame()

    records = []
    match_id = file_path.stem
    date = info.get("dates", [None])[0]
    venue = info.get("venue")
    city = info.get("city")
    teams = info.get("teams", [None, None])
    season = str(date)[:4] if date else None
    winner = (info.get("outcome") or {}).get("winner")

    for i, innings in enumerate(innings_list, start=1):
        innings_name = list(innings.keys())[0]
        team_batting = innings[innings_name].get("team")
        team_bowling = [t for t in teams if t != team_batting]
        team_bowling = team_bowling[0] if team_bowling else None

        deliveries = innings[innings_name].get("deliveries", [])
        for delivery in deliveries:
            ball_id = list(delivery.keys())[0]
            data = delivery[ball_id]

            over = int(str(ball_id).split(".")[0])
            ball_in_over = int(str(ball_id).split(".")[1])

            runs = data.get("runs", {})
            extras = data.get("extras", {})
            wicket = data.get("wicket", {})

            records.append({
                "match_id": match_id,
                "season": season,
                "date": date,
                "venue": venue,
                "city": city,
                "team_batting": team_batting,
                "team_bowling": team_bowling,
                "innings_no": i,
                "over": over,
                "ball_in_over": ball_in_over,
                "batsman": data.get("batsman"),
                "non_striker": data.get("non_striker"),
                "bowler": data.get("bowler"),
                "runs_batter": runs.get("batsman", 0),
                "runs_extras": runs.get("extras", 0),
                "runs_total": runs.get("total", 0),
                "extras_type": list(extras.keys())[0] if extras else None,
                "wicket_player_out": wicket.get("player_out"),
                "wicket_kind": wicket.get("kind"),
                "match_winner": winner,
            })

    return pd.DataFrame.from_records(records)


def build_dataset():
    """Parse all YAMLs and save parquet."""
    all_files = sorted(RAW_DIR.glob("*.yaml"))
    if not all_files:
        print(f"⚠️ No YAML files found in {RAW_DIR}")
        return

    all_dfs = []
    for f in tqdm(all_files, desc="Parsing IPL matches"):
        try:
            df = parse_match(f)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"❌ Error {f}: {e}")

    if not all_dfs:
        print("No data parsed successfully.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Coerce 'date' to datetime (invalid => NaT)
    full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")

    # Ensure string columns are consistent (avoid mixed types)
    str_cols = [
        "venue", "city", "team_batting", "team_bowling",
        "batsman", "non_striker", "bowler",
        "extras_type", "wicket_player_out", "wicket_kind", "match_winner"
    ]
    for c in str_cols:
        full_df[c] = full_df[c].astype("string")

    print(f"Saving {len(full_df):,} deliveries to {OUTPUT_PATH}")
    full_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(full_df):,} deliveries to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()
