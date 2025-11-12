import pandas as pd

# ----------------------------------------------------------------
# ⚠️ UPDATE THIS PATH to point to your actual Parquet file
# ----------------------------------------------------------------
FILE_PATH = "data/processed/ipl_deliveries.parquet"

# Load the file
try:
    df = pd.read_parquet(FILE_PATH)
    print(f"✅ Successfully loaded {FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {FILE_PATH}")
    print("Please update the FILE_PATH variable in this script.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")

# --- 1. See all column names ---
# This is the most important command
print("\n--- 1. Column Names ---")
print(list(df.columns))

# --- 2. See column names AND their data types ---
# This will show you if 'innings' (or whatever it's called)
# is an integer, float, or string ('object')
print("\n--- 2. Column Info (Types) ---")
df.info()

# --- 3. Look at the first 5 rows of data ---
# This helps you visually confirm the 'innings' column
print("\n--- 3. Sample Data (Head) ---")
print(df.head())
