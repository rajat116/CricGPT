import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")
OUTPUT_PATH = Path("data/processed/player_names.csv")

def build_player_name_list():
    df = pd.read_parquet(DATA_PATH)

    batters = df["batsman"].dropna().unique().tolist()
    bowlers = df["bowler"].dropna().unique().tolist()

    all_players = sorted(set(batters + bowlers))
    print(f"✅ Found {len(all_players)} unique player names.")

    out_df = pd.DataFrame({"dataset_name": all_players})
    out_df["canonical_name"] = ""  # leave blank for manual editing
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved to {OUTPUT_PATH.absolute()}")

if __name__ == "__main__":
    build_player_name_list()
