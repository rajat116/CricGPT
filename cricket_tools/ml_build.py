"""
ml_build.py — prepares ML-ready batting features from IPL deliveries dataset
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")
OUT_PATH = Path("data/processed/ml_features.parquet")

def build_ml_features():
    """Build aggregated player-level batting stats for ML training."""
    if not DATA_PATH.exists():
        print(f"❌ Source file not found: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)

    # Aggregate by batsman
    features = (
        df.groupby("batsman")
        .agg(
            matches=("match_id", "nunique"),
            runs=("runs_batter", "sum"),
            balls=("runs_batter", "count"),  # ✅ simpler & accurate
            dismissals=("wicket_player_out", lambda x: x.notna().sum()),
            fours=("runs_batter", lambda x: (x == 4).sum()),
            sixes=("runs_batter", lambda x: (x == 6).sum()),
        )
        .reset_index()
    )

    # Derived batting metrics
    features["strike_rate"] = 100 * features["runs"] / features["balls"].clip(lower=1)
    features["avg"] = features.apply(
        lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else float("inf"), axis=1
    )

    # Clean up and save
    features.fillna(0, inplace=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH)

    print(f"✅ ML features saved to {OUT_PATH}")
    print(f"📊 {len(features)} player feature rows written")

if __name__ == "__main__":
    build_ml_features()