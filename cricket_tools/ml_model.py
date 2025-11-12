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
    df.replace([np.inf, -np.inf, "inf", "-inf"], np.nan, inplace=True)
    df[FEATURE_COLS_BAT] = df[FEATURE_COLS_BAT].fillna(0)
    return df

def _load_features_bowl() -> pd.DataFrame:
    if not FEATURES_PATH_BOWL.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH_BOWL}")
    df = pd.read_parquet(FEATURES_PATH_BOWL).copy()
    df[TARGET_COL_BOWL] = df["wickets"] / df["matches"].clip(lower=1)   # ✅ added
    df.replace([np.inf, -np.inf, "inf", "-inf"], np.nan, inplace=True)
    df[FEATURE_COLS_BOWL] = df[FEATURE_COLS_BOWL].fillna(0)
    return df

# ---------------------------------------------------------------------
def _train_rf(X, y, random_state: int = 42):
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    return model, {"mse": round(mse, 6), "rmse": round(rmse, 3), "n_train": len(X_train), "n_test": len(X_test)}

# ---------------------------------------------------------------------
def train_performance_model_bat(random_state: int = 42) -> dict:
    df = _load_features_bat()
    X, y = df[FEATURE_COLS_BAT].values, df[TARGET_COL_BAT].values
    model, metrics = _train_rf(X, y, random_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH_BAT)
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(MODEL_PATH_BAT),
        "feature_cols": FEATURE_COLS_BAT,
        "target_col": TARGET_COL_BAT,
        "metrics": metrics,
        "sklearn_model": "RandomForestRegressor",
    }
    json.dump(meta, open(META_PATH_BAT, "w"), indent=2)
    print(f"✅ Batting model saved: {MODEL_PATH_BAT} (RMSE={metrics['rmse']})")
    return {"model_path": str(MODEL_PATH_BAT), "rmse": metrics["rmse"]}

def train_performance_model_bowl(random_state: int = 42) -> dict:
    df = _load_features_bowl()
    X, y = df[FEATURE_COLS_BOWL].values, df[TARGET_COL_BOWL].values
    model, metrics = _train_rf(X, y, random_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH_BOWL)
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(MODEL_PATH_BOWL),
        "feature_cols": FEATURE_COLS_BOWL,
        "target_col": TARGET_COL_BOWL,
        "metrics": metrics,
        "sklearn_model": "RandomForestRegressor",
    }
    json.dump(meta, open(META_PATH_BOWL, "w"), indent=2)
    print(f"✅ Bowling model saved: {MODEL_PATH_BOWL} (RMSE={metrics['rmse']})")
    return {"model_path": str(MODEL_PATH_BOWL), "rmse": metrics["rmse"]}

# ---------------------------------------------------------------------
_model_cache = {}

def _get_model(path, meta_path):
    if path not in _model_cache:
        _model_cache[path] = {
            "model": joblib.load(path),
            "meta": json.load(open(meta_path))
        }
    return _model_cache[path]

def predict_future_performance(player: str, dataset_name: str = None, **kwargs) -> Dict[str, Any]:
    """Predict future performance using both batting and bowling models if available."""
    df_bat = pd.read_parquet(FEATURES_PATH_BAT) if FEATURES_PATH_BAT.exists() else pd.DataFrame()
    df_bowl = pd.read_parquet(FEATURES_PATH_BOWL) if FEATURES_PATH_BOWL.exists() else pd.DataFrame()
    player_l = player.lower()
    result_data = {}

    # Batting
    if not df_bat.empty:
        found_bat = df_bat[df_bat["batsman"].astype(str).str.lower() == player_l]
        if found_bat.empty and " " in player_l:
            last = player_l.split()[-1].lower()
            found_bat = df_bat[df_bat["batsman"].astype(str).str.lower().str.contains(last)]
        if not found_bat.empty:
            row = found_bat.iloc[0]
            obj = _get_model(MODEL_PATH_BAT, META_PATH_BAT)
            x = row[obj["meta"]["feature_cols"]].values.reshape(1, -1)
            pred = float(obj["model"].predict(x)[0])
            result_data["batting_prediction"] = {
                "predicted_runs_per_match": round(pred, 2),
                "inputs": {c: float(row[c]) for c in obj["meta"]["feature_cols"]},
                "model": obj["meta"]["sklearn_model"],
                "trained_at": obj["meta"].get("created_at"),
            }

    # Bowling
    if not df_bowl.empty:
        found_bowl = df_bowl[df_bowl["bowler"].astype(str).str.lower() == player_l]
        if found_bowl.empty and " " in player_l:
            last = player_l.split()[-1].lower()
            found_bowl = df_bowl[df_bowl["bowler"].astype(str).str.lower().str.contains(last)]
        if not found_bowl.empty:
            row = found_bowl.iloc[0]
            obj = _get_model(MODEL_PATH_BOWL, META_PATH_BOWL)
            x = row[obj["meta"]["feature_cols"]].values.reshape(1, -1)
            pred = float(obj["model"].predict(x)[0])
            result_data["bowling_prediction"] = {
                "predicted_wickets_per_match": round(pred, 2),
                "inputs": {c: float(row[c]) for c in obj["meta"]["feature_cols"]},
                "model": obj["meta"]["sklearn_model"],
                "trained_at": obj["meta"].get("created_at"),
            }

    if not result_data:
        return {"error": f"No features found for resolved player '{player}'."}

    return {"player": player, "dataset_name": dataset_name or "unknown", **result_data}

# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_performance_model_bat()
    train_performance_model_bowl()