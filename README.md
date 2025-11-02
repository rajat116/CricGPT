## Step 0 — Setup & Data Verification

✅ Environment and dependencies installed  
✅ Downloaded complete IPL dataset from Cricsheet  
✅ Verified YAML loading and structure  
✅ Confirmed `deliveries` schema (used from now on)

### Step 1 — Data Parser & Cache
✅ Flattened all IPL YAML files into a single Parquet dataset  
✅ Columns: match_id, season, teams, batsman, bowler, runs, extras, wickets etc.  
✅ Output: `data/processed/ipl_deliveries.parquet`