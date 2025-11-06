"""
ml_model.py — trains regression models to estimate
batting runs per match and bowling wickets per match
from aggregated player features.
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ---------------------------------------------------------------------
# Paths
FEATURES_PATH_BAT = Path("data/processed/ml_features.parquet")
FEATURES_PATH_BOWL = Path("data/processed/ml_features_bowl.parquet")

MODEL_DIR = Path("models")
MODEL_PATH_BAT = MODEL_DIR / "performance_model_bat.pkl"
META_PATH_BAT = MODEL_DIR / "performance_model_bat_meta.json"
MODEL_PATH_BOWL = MODEL_DIR / "performance_model_bowl.pkl"
META_PATH_BOWL = MODEL_DIR / "performance_model_bowl_meta.json"

# ---------------------------------------------------------------------
# Feature definitions
FEATURE_COLS_BAT = ["balls", "dismissals", "fours", "sixes", "strike_rate", "avg"]
TARGET_COL_BAT = "runs_per_match"

FEATURE_COLS_BOWL = ["balls", "wickets", "runs_conceded", "economy", "overs"]
TARGET_COL_BOWL = "wickets_per_match"

# ---------------------------------------------------------------------
def _load_features_bat() -> pd.DataFrame:
    if not FEATURES_PATH_BAT.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH_BAT}")
    df = pd.read_parquet(FEATURES_PATH_BAT).copy()
    df[TARGET_COL_BAT] = df["runs"] / df["matches"].clip(lower=1)
    df[FEATURE_COLS_BAT] = df[FEATURE_COLS_BAT].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _load_features_bowl() -> pd.DataFrame:
    if not FEATURES_PATH_BOWL.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH_BOWL}")
    df = pd.read_parquet(FEATURES_PATH_BOWL).copy()
    df[FEATURE_COLS_BOWL] = df[FEATURE_COLS_BOWL].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ---------------------------------------------------------------------
def train_performance_model_bat(random_state: int = 42) -> dict:
    """Train RandomForest to predict batting runs_per_match."""
    df = _load_features_bat()
    X = df[FEATURE_COLS_BAT].values
    y = df[TARGET_COL_BAT].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH_BAT)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(MODEL_PATH_BAT),
        "feature_cols": FEATURE_COLS_BAT,
        "target_col": TARGET_COL_BAT,
        "metrics": {"mse": mse, "rmse": rmse, "n_train": len(X_train), "n_test": len(X_test)},
        "sklearn_model": "RandomForestRegressor",
    }
    with open(META_PATH_BAT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Batting model saved: {MODEL_PATH_BAT} (RMSE={rmse:.3f})")
    return {"model_path": str(MODEL_PATH_BAT), "rmse": rmse}


def train_performance_model_bowl(random_state: int = 42) -> dict:
    """Train RandomForest to predict bowling wickets_per_match."""
    df = _load_features_bowl()
    X = df[FEATURE_COLS_BOWL].values
    y = df[TARGET_COL_BOWL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH_BOWL)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(MODEL_PATH_BOWL),
        "feature_cols": FEATURE_COLS_BOWL,
        "target_col": TARGET_COL_BOWL,
        "metrics": {"mse": mse, "rmse": rmse, "n_train": len(X_train), "n_test": len(X_test)},
        "sklearn_model": "RandomForestRegressor",
    }
    with open(META_PATH_BOWL, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Bowling model saved: {MODEL_PATH_BOWL} (RMSE={rmse:.3f})")
    return {"model_path": str(MODEL_PATH_BOWL), "rmse": rmse}

def predict_future_performance(player: str, dataset_name: str = None, **kwargs) -> Dict[str, Any]:
    """
    Predict future performance using both batting and bowling models if available.
    Runs both models when the player appears in both datasets.
    """

    # --- Load both feature sets ---
    df_bat = pd.read_parquet(FEATURES_PATH_BAT) if FEATURES_PATH_BAT.exists() else pd.DataFrame()
    df_bowl = pd.read_parquet(FEATURES_PATH_BOWL) if FEATURES_PATH_BOWL.exists() else pd.DataFrame()

    player_l = player.lower()
    result_data = {}

    # ---------------- Batting Prediction ----------------
    if not df_bat.empty:
        found_bat = df_bat[df_bat["batsman"].astype(str).str.lower() == player_l]
        if found_bat.empty and " " in player_l:
            last = player_l.split()[-1].lower()
            found_bat = df_bat[df_bat["batsman"].astype(str).str.lower().str.contains(last)]

        if not found_bat.empty:
            row = found_bat.iloc[0]
            model = joblib.load(MODEL_PATH_BAT)
            with open(META_PATH_BAT, "r") as f:
                meta = json.load(f)
            x = row[meta["feature_cols"]].values.reshape(1, -1)
            pred = float(model.predict(x)[0])
            result_data["batting_prediction"] = {
                "predicted_runs_per_match": round(pred, 2),
                "inputs": {col: float(row[col]) for col in meta["feature_cols"]},
                "model": meta["sklearn_model"],
                "trained_at": meta.get("created_at"),
            }

    # ---------------- Bowling Prediction ----------------
    if not df_bowl.empty:
        found_bowl = df_bowl[df_bowl["bowler"].astype(str).str.lower() == player_l]
        if found_bowl.empty and " " in player_l:
            last = player_l.split()[-1].lower()
            found_bowl = df_bowl[df_bowl["bowler"].astype(str).str.lower().str.contains(last)]

        if not found_bowl.empty:
            row = found_bowl.iloc[0]
            model = joblib.load(MODEL_PATH_BOWL)
            with open(META_PATH_BOWL, "r") as f:
                meta = json.load(f)
            x = row[meta["feature_cols"]].values.reshape(1, -1)
            pred = float(model.predict(x)[0])
            result_data["bowling_prediction"] = {
                "predicted_wickets_per_match": round(pred, 2),
                "inputs": {col: float(row[col]) for col in meta["feature_cols"]},
                "model": meta["sklearn_model"],
                "trained_at": meta.get("created_at"),
            }

    # ---------------- Fallback ----------------
    if not result_data:
        return {"error": f"No features found for resolved player '{player}'."}

    # ---------------- Final combined output ----------------
    return {
        "player": player,
        "dataset_name": dataset_name or "unknown",
        **result_data,
    }

# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_performance_model_bat()
    train_performance_model_bowl()