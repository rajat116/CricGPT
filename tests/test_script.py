from cricket_tools.filters import load_dataset
df = load_dataset()
cols = [c for c in df.columns if "wicket" in c.lower() or "bowler" in c.lower()]
print("🔍 Columns possibly related to wickets:")
print(cols)