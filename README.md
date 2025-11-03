## Step 0 — Setup & Data Verification

✅ Environment and dependencies installed  
✅ Downloaded complete IPL dataset from Cricsheet  
✅ Verified YAML loading and structure  
✅ Confirmed `deliveries` schema (used from now on)

### Step 1 — Data Parser & Cache
✅ Flattened all IPL YAML files into a single Parquet dataset  
✅ Columns: match_id, season, teams, batsman, bowler, runs, extras, wickets etc.  
✅ Output: `data/processed/ipl_deliveries.parquet`

### Step 2 — Smart Name Resolver (Balanced Semantic Search)

This module (`cricket_tools/smart_names.py`) implements an advanced player-name
resolver that can understand partial, misspelled, or ambiguous inputs.

### Key Features
- Handles **typos**, **initials**, and **surname-only** queries.
- Uses **phonetic** (Metaphone) + **fuzzy** + **semantic** similarity.
- Auto-confirms only when highly confident (balanced thresholds).
- Returns `"ok"`, `"confirm"`, `"ambiguous"`, or `"not_found"` for agent logic.

### Example Usage
```python
from cricket_tools.smart_names import resolve_player_smart

print(resolve_player_smart("Rohitt Sharm"))```
# ('Rohit Gurunath Sharma', 'confirm', 'Did you mean **Rohit Gurunath Sharma**? If not maybe: Mohit Mahipal Sharma, Rahul Sharma.')

### Step 3 — Player Stats & Role Handlers (`stats.py`, `filters.py`)

This step adds **analytical logic** for batters and bowlers, separating data retrieval from core orchestration.

### Modules and Responsibilities

| File | Purpose |
|------|----------|
| `stats.py` | Computes player-level aggregates (runs, balls, dismissals, strike rate, average, etc.) using the processed deliveries data. |
| `filters.py` | Applies optional filters (season range, venue, team matchup, etc.) to the deliveries DataFrame before aggregation. |

### Functions

```python
from cricket_tools.stats import get_player_stats, get_bowler_stats```

## Step 4 — ML Prediction Integration (`ml_build.py`, `ml_model.py`, `core.py`)

This step introduces the **machine-learning layer** that predicts a player’s expected *runs per match* from historical IPL data.  
It plugs into the unified API so you can query via `core.cricket_query(..., role="predict")`.

---

### Pipeline Overview

1. **`ml_build.py`** – aggregates delivery-level records into player-level features.  
2. **`ml_model.py`** – trains a `RandomForestRegressor` to predict *average runs per match*.  
3. **`core.py`** – routes queries with `role="predict"` to `predict_future_performance()`.

---

### Feature Construction

Source: `data/processed/ipl_deliveries.parquet` → Output: `data/processed/ml_features.parquet`

| Feature        | Meaning                         |
|----------------|----------------------------------|
| `matches`      | number of unique matches         |
| `runs`         | total runs scored                |
| `balls`        | total balls faced                |
| `dismissals`   | times out                        |
| `fours`, `sixes` | boundary counts               |
| `strike_rate`  | `100 * runs / balls`             |
| `avg`          | `runs / dismissals` (NaN→0 safe) |

Build features:

```bash
python -m cricket_tools.ml_build

