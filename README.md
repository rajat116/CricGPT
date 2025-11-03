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

print(resolve_player_smart("Rohitt Sharm"))'''
# ('Rohit Gurunath Sharma', 'confirm', 'Did you mean **Rohit Gurunath Sharma**? If not, maybe: Mohit Mahipal Sharma, Rahul Sharma.')
