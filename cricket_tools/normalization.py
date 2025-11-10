def normalize_entity(name, kind="city"):
    """
    Normalize entity names like city, venue, team to canonical forms.
    """
    if not name:
        return None

    name = name.strip().lower()

    # ---- Canonical City Normalization ----
    if kind == "city":
        city_map = {
            "banglore": "Bengaluru",
            "bangalore": "Bengaluru",
            "bengluru": "Bengaluru",
            "blr": "Bengaluru",
            "bengaluru": "Bengaluru",
            "new chandigarh": "New Chandigarh",
            "mohali": "Mohali",
            "navi mumbai": "Navi Mumbai",
            "mumbai": "Mumbai",
            "madras": "Chennai",
        }
        for k, v in city_map.items():
            if name == k:
                return v
        return name.title()

    # ---- Canonical Venue Normalization ----
    if kind == "venue":
        venue_map = {
            "chinnaswamy": "M Chinnaswamy Stadium",
            "chepuk": "MA Chidambaram Stadium",
            "ma chidambaram": "MA Chidambaram Stadium",
            "narendra modi": "Narendra Modi Stadium",
            "wankhede": "Wankhede Stadium",
            "eden gardens": "Eden Gardens",
            "dy patil": "Dr DY Patil Sports Academy",
            "brabourne": "Brabourne Stadium",
            "rajiv gandhi": "Rajiv Gandhi International Stadium",
        }
        for k, v in venue_map.items():
            if k in name:
                return v
        return name.title()

    # ---- Canonical Team Normalization ----
    if kind == "team":
        team_map = {
            "rcb": "Royal Challengers Bengaluru",
            "royal challengers bangalore": "Royal Challengers Bengaluru",
            "royal challengers bengaluru": "Royal Challengers Bengaluru",
            "mi": "Mumbai Indians",
            "csk": "Chennai Super Kings",
            "srh": "Sunrisers Hyderabad",
            "dc": "Delhi Capitals",
            "kkr": "Kolkata Knight Riders",
            "rr": "Rajasthan Royals",
            "kxip": "Kings XI Punjab",
            "pbks": "Punjab Kings",
            "gt": "Gujarat Titans",
            "lsg": "Lucknow Super Giants",
        }
        for k, v in team_map.items():
            if name == k:
                return v
        return name.title()

    return name.title()
