"""
ml_model.py — trains a regression model to estimate batting
runs per match from aggregated player features. Also exposes a
predict function used by core.py (role="predict").
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
FEATURES_PATH = Path("data/processed/ml_features.parquet")
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "performance_model.pkl"
META_PATH     = MODEL_DIR / "performance_model_meta.json"

# ---------------------------------------------------------------------
FEATURE_COLS = ["balls", "dismissals", "fours", "sixes", "strike_rate", "avg"]
TARGET_COL   = "runs_per_match"

# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH).copy()
    df[TARGET_COL] = df["runs"] / df["matches"].clip(lower=1)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ---------------------------------------------------------------------
def train_performance_model(random_state: int = 42) -> dict:
    """Train RandomForest to predict runs_per_match."""
    df = _load_features()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Manual RMSE computation for compatibility
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(MODEL_PATH),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": {"mse": mse, "rmse": rmse, "n_train": len(X_train), "n_test": len(X_test)},
        "sklearn_model": "RandomForestRegressor",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model saved: {MODEL_PATH}")
    print(f"📝 Metadata   : {META_PATH}")
    print(f"📉 RMSE       : {rmse:.3f}")

    return {"model_path": str(MODEL_PATH), "rmse": rmse}

# ---------------------------------------------------------------------
def predict_future_performance(player: str, dataset_name: str = None, **kwargs) -> Dict[str, Any]:
    """
    Predict future runs per match using trained model and ML features parquet.
    Expects `player` and (optionally) `dataset_name` to be already resolved by core.py.
    """
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return {"error": "Model not trained. Please run train_performance_model() first."}

    df = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    # --- Use dataset_name first; fall back to player if needed
    search_names = [n for n in [dataset_name, player] if n]
    found = pd.DataFrame()

    for n in search_names:
        subset = df[df["batsman"].astype(str).str.lower() == n.lower()]
        if not subset.empty:
            found = subset
            break

    # Fallback: partial last-name search
    if found.empty and " " in player:
        last = player.split()[-1].lower()
        mask = df["batsman"].astype(str).str.lower().str.contains(last)
        found = df[mask]

    if found.empty:
        return {"error": f"No features found for resolved player '{player}'."}

    row = found.iloc[0]
    x = row[meta["feature_cols"]].values.reshape(1, -1)
    pred_rpm = float(model.predict(x)[0])

    return {
        "player": player,
        "dataset_name": dataset_name or "unknown",
        "predicted_runs_per_match": round(pred_rpm, 2),
        "inputs": {col: float(row[col]) for col in meta["feature_cols"]},
        "model": meta["sklearn_model"],
        "trained_at": meta.get("created_at"),
    }

# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_performance_model()