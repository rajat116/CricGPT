# cricket_tools/check_data.py
import yaml
import pathlib
from pprint import pprint

# pick any YAML from IPL folder
sample = next(pathlib.Path("data/raw/ipl").glob("*.yaml"))
print(f"\n📂  Sample file: {sample}")

with open(sample, "r") as f:
    match = yaml.safe_load(f)

# ---------------------------
#  Basic info summary
# ---------------------------
info = match.get("info", {})
innings = match.get("innings", [])

print("\n=== MATCH INFO ===")
pprint({
    "venue": info.get("venue"),
    "city": info.get("city"),
    "teams": info.get("teams"),
    "date": info.get("dates"),
    "match_type": info.get("match_type"),
})

# ---------------------------
#  Verify innings + deliveries
# ---------------------------
if innings:
    first_innings = innings[0]
    innings_name = list(first_innings.keys())[0]  # e.g. "1st innings"
    team = first_innings[innings_name].get("team")
    deliveries = first_innings[innings_name].get("deliveries", [])

    print(f"\n=== INNINGS DETAILS ===")
    print(f"Innings name : {innings_name}")
    print(f"Batting team : {team}")
    print(f"Total deliveries : {len(deliveries)}")

    # print one sample delivery to confirm structure
    if deliveries:
        delivery = deliveries[0]
        ball_key = list(delivery.keys())[0]
        data = delivery[ball_key]
        print("\nSample delivery:")
        pprint(data)
else:
    print("⚠️ No innings data found in this match file.")
