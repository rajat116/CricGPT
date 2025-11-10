import pandas as pd

# Load your IPL data
df = pd.read_parquet("data/processed/ipl_deliveries.parquet")

# Extract unique entries
cities = sorted(df["city"].dropna().unique())
venues = sorted(df["venue"].dropna().unique())
teams = sorted(set(df["team_batting"].dropna()) | set(df["team_bowling"].dropna()))

# Print results
print("🏙️  Unique Cities:")
for c in cities:
    print("-", c)

print("\n🏟️  Unique Venues:")
for v in venues:
    print("-", v)

print("\n👕 Unique Teams:")
for t in teams:
    print("-", t)