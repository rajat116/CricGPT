"""
ml_build_bowl.py — prepares ML-ready bowling features from IPL deliveries dataset
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")
OUT_PATH = Path("data/processed/ml_features_bowl.parquet")

def build_ml_bowling_features():
    """Build aggregated player-level bowling stats for ML training."""
    if not DATA_PATH.exists():
        print(f"❌ Source file not found: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)

    # ✅ Exclude wides and no-balls for legal deliveries
    legal_df = df[~df["extras_type"].isin(["wides", "noballs"])]

    # Aggregate by bowler
    features = (
        legal_df.groupby("bowler")
        .agg(
            matches=("match_id", "nunique"),
            runs_conceded=("runs_total", "sum"),
            balls=("ball_in_over", "count"),
            wickets=("wicket_player_out", lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    # Derived bowling metrics
    features["overs"] = features["balls"] / 6
    features["economy"] = features.apply(
        lambda r: r["runs_conceded"] / r["overs"] if r["overs"] > 0 else float("inf"), axis=1
    )
    features["wickets_per_match"] = features.apply(
        lambda r: r["wickets"] / r["matches"] if r["matches"] > 0 else 0.0, axis=1
    )

    # Fill missing values where appropriate
    features.fillna(0, inplace=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH)

    print(f"✅ Bowling features saved to {OUT_PATH}")
    print(f"📊 {len(features)} bowler feature rows written")

if __name__ == "__main__":
    build_ml_bowling_features()