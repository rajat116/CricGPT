# cricket_tools/raw_summaries.py

def summarize_team_head_to_head(r):
    teamA = r["teamA"]
    teamB = r["teamB"]
    n = len(r["records"])
    winsA = sum(1 for x in r["records"] if x["win_ratio_teamA"] > 0.5)
    winsB = n - winsA

    return (
        f"📊 **Head-to-Head Summary — {teamA} vs {teamB}**\n\n"
        f"- Total matches analysed: **{n}**\n"
        f"- {teamA} wins: **{winsA}**\n"
        f"- {teamB} wins: **{winsB}**\n"
        f"- Recent trend: {teamA if r['records'][-1]['win_ratio_teamA'] > 0.5 else teamB} "
        f"won the latest match.\n\n"
        "Tip: Enable plots for season-wise breakdown."
    )


def summarize_team_momentum(r):
    tA, tB = r["teamA"], r["teamB"]
    # each series ~ list of momentum values per over
    a = r["momentum_teamA"]
    b = r["momentum_teamB"]

    peakA = max(a)
    peakB = max(b)

    domA = sum(1 for i in range(len(a)) if a[i] > b[i])
    domB = len(a) - domA

    return (
        f"📈 **Momentum Comparison — {tA} vs {tB}**\n\n"
        f"- Peak momentum: **{tA}: {peakA:.2f}**, **{tB}: {peakB:.2f}**\n"
        f"- Overs dominated: **{tA}: {domA}**, **{tB}: {domB}**\n"
        f"- Overall: {'slight edge to ' + tA if domA > domB else 'slight edge to ' + tB}\n\n"
        "Tip: Enable the plot for over-by-over visualization."
    )


def summarize_team_phase_dominance(r):
    tA, tB = r["teamA"], r["teamB"]
    p = r["phase_scores"]   # dict: {"powerplay": {...}, "middle": {...}, "death": {...}}

    def winner(phase):
        return tA if p[phase][tA] > p[phase][tB] else tB

    return (
        f"🟧 **Phase Dominance — {tA} vs {tB}**\n\n"
        f"- Powerplay: **{winner('powerplay')}** dominated\n"
        f"- Middle overs: **{winner('middle')}** dominated\n"
        f"- Death overs: **{winner('death')}** dominated\n\n"
        "Tip: Enable plots for scoring profile."
    )


def summarize_pressure_series(r):
    tA, tB = r["teamA"], r["teamB"]
    a = r["pressure_teamA"]
    b = r["pressure_teamB"]

    highA = max(a)
    highB = max(b)

    return (
        f"🔥 **Pressure Timeline — {tA} vs {tB}**\n\n"
        f"- Peak pressure: **{tA}: {highA:.2f}**, **{tB}: {highB:.2f}**\n"
        f"- Final overs favored: {'team ' + tA if a[-1] < b[-1] else tB}\n\n"
        "Tip: Enable plot for full pressure progression."
    )


def summarize_win_probability_series(r):
    tA, tB = r["teamA"], r["teamB"]
    wpa = r["wp_teamA"]     # win probability over time
    wpb = r["wp_teamB"]

    # final prediction
    pred = tA if wpa[-1] > wpb[-1] else tB

    # biggest swing overs
    swings = max(abs(wpa[i] - wpa[i-1]) for i in range(1, len(wpa)))

    return (
        f"📉 **Win Probability Summary — {tA} vs {tB}**\n\n"
        f"- Predicted winner (final): **{pred}**\n"
        f"- Largest probability swing: **{swings*100:.1f}%**\n"
        f"- Close contest: {'Yes' if swings < 0.12 else 'No'}\n\n"
        "Tip: Enable plot for ball-by-ball probability movement."
    )


def summarize_partnership_graph(r):
    t = r["team"]
    pairs = r["pairs"]   # list of dicts: [{"pair": "A+B", "runs": 42}, ...]

    # find top partnership
    best = max(pairs, key=lambda x: x["runs"])

    return (
        f"👬 **Partnership Summary — {t}**\n\n"
        f"- Total partnerships: **{len(pairs)}**\n"
        f"- Highest stand: **{best['pair']} — {best['runs']} runs**\n\n"
        "Tip: Enable plot for full partnership tree."
    )