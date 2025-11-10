import pandas as pd

# Path to your processed IPL deliveries file
path = "data/processed/ipl_deliveries.parquet"

# Load only first few rows to avoid large memory use
df = pd.read_parquet(path)

print("\n📂 File:", path)
print("\n🧱 Total columns:", len(df.columns))
print("\n📋 Column names:\n")
for i, col in enumerate(df.columns, 1):
    print(f"{i:>2}. {col}")

print("\n🔎 Sample rows:")
print(df.head(3))
