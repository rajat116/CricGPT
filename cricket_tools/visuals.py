# visuals.py — Rendering layer (charts only)
# -----------------------------------------
# Depends on analytics.py for data. Saves plots to outputs/plots/*.png

from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cricket_tools import analytics


# -------------------------------
# Helpers
# -------------------------------
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# Fallback card (for LLM generic answers or missing data)
# ---------------------------------------------------------------------
def _plot_fallback_card(data: dict) -> str:
    plt.style.use("dark_background")
    outdir = _ensure_dir("outputs/plots")
    ts = _stamp()

    title = data.get("title") or "CricGPT LLM Fallback View"
    who = data.get("player") or data.get("team") or "Unknown"
    comment = data.get("comment", "")

    # pull numeric bits if present
    numeric_keys = ["runs", "avg", "strike_rate", "wickets", "economy", "centuries"]
    numeric_data = {k: v for k, v in data.items()
                    if k in numeric_keys and isinstance(v, (int, float)) and v is not None}

    if numeric_data:
        fig, ax = plt.subplots(figsize=(7, 3))
        keys, vals = list(numeric_data.keys()), list(numeric_data.values())
        bars = ax.bar(keys, vals)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, val, f"{val}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Value")
        ax.set_title(f"{who} — AI Fallback Summary")
        fname = f"{outdir}/fallback_{who.replace(' ', '_')}_{ts}.png"
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.65, who, ha="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.40, comment, ha="center", wrap=True, fontsize=10)
        ax.set_title(title)
        fname = f"{outdir}/fallback_{who.replace(' ', '_')}_{ts}.png"

    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname


def _adapt_llm_result_to_form_df(result: dict) -> pd.DataFrame | None:
    if not isinstance(result, dict):
        return None
    data = result.get("data") or result
    if not isinstance(data, dict):
        return None
    if "player" in data and any(k in data for k in ("runs", "avg", "strike_rate")):
        return pd.DataFrame([{
            "player": data.get("player", "Unknown"),
            "runs": data.get("runs", 0),
            "avg": data.get("avg", None),
            "strike_rate": data.get("strike_rate", None),
            "comment": data.get("comment", "AI-derived fallback result"),
        }])
    return None


# ---------------------------------------------------------------------
# Main entry: generate_plot_from_result
# ---------------------------------------------------------------------
def generate_plot_from_result(result: dict, act: str, plot_type: str):
    # Router for new analytics plots
    if plot_type == "top" and act == "get_top_players":
        data = result.get("data") or {}
        return _plot_top_players(data)

    if plot_type == "h2h-team":
        return plot_team_h2h_matrix(result)
    if plot_type == "momentum":
        return plot_team_momentum_worm(result)
    if plot_type == "phase-dominance":
        return plot_team_phase_dominance(result)

    if plot_type == "threat-map":
        return plot_team_threat_map(result)
    data = result.get("data", result)
    if plot_type in ("partnership", "partnership_graph"):
        return plot_partnership_graph(data)
    elif plot_type in ("pressure", "pressure_curve", "pressure_series"):
        return plot_pressure_series(data)
    elif plot_type in ("win-prob", "win_probability", "win_probability_series"):
        return plot_win_probability_series(data)

    # Pure fallback
    if act.startswith("llm_fallback"):
        data = result.get("data") or result
        return _plot_fallback_card(data)

    # LLM minimal numeric → tiny bar card
    llm_df = _adapt_llm_result_to_form_df(result)
    if llm_df is not None:
        d = llm_df.iloc[0]
        player = d["player"]
        runs = float(d.get("runs", 0) or 0)
        comment = d.get("comment", "AI-derived summary")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.bar(["Runs"], [runs])
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{player} — AI-Derived Summary")
        ax.set_ylabel("Runs (approx.)")
        ax.text(0.5, -0.25, comment, transform=ax.transAxes, fontsize=8, ha="center", va="top", wrap=True)
        ax.text(0.01, -0.15, "CricGPT LLM Fallback View", transform=ax.transAxes, fontsize=9, alpha=0.8)

        outdir = _ensure_dir("outputs/plots")
        fname = f"{outdir}/fallback_{player.replace(' ', '_')}_{_stamp()}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    # Structured single-player (form)
    outdir = _ensure_dir("outputs/plots")
    ts = _stamp()

    if act == "get_batter_stats" and "data" in result:
        d = result["data"]
        player = d.get("player", "Unknown")
        avg = float(d.get("avg", 0) or 0)
        sr = float(d.get("strike_rate", 0) or 0)
        runs = float(d.get("runs", 0) or 0)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["Runs", "Average", "Strike Rate"], [runs, avg, sr])
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom")
        ax.set_title(f"{player} — Batting Summary")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.text(0.01, -0.15, "CricGPT • get_batter_stats", transform=ax.transAxes, fontsize=9, alpha=0.8)

        fname = f"{outdir}/form_{player.replace(' ', '_')}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    elif act == "compare_players" and "data" in result:
        cmpd = result["data"].get("comparison", {})
        # keep insertion order where possible
        players = [k for k in cmpd.keys()]
        if len(players) < 2:
            return _plot_fallback_card({"comment": "Comparison requires two players."})
        pA, pB = players[0], players[1]
        avgA = float(cmpd[pA].get("avg", 0) or 0)
        avgB = float(cmpd[pB].get("avg", 0) or 0)
        srA  = float(cmpd[pA].get("sr",  0) or 0)
        srB  = float(cmpd[pB].get("sr",  0) or 0)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics = ["Average", "Strike Rate"]
        x = np.arange(len(metrics))
        width = 0.38
        ax.bar(x - width/2, [avgA, srA], width, label=pA)
        ax.bar(x + width/2, [avgB, srB], width, label=pB)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(f"{pA} vs {pB}")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.text(0.01, -0.15, "CricGPT • compare_players", transform=ax.transAxes, fontsize=9, alpha=0.8)

        fname = f"{outdir}/h2h_{pA.split()[0]}_vs_{pB.split()[0]}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    elif act == "get_team_stats" and "data" in result:
        d = result["data"]
        team = d.get("team", "Unknown")
        matches = int(d.get("matches", 0) or 0)
        wins = int(d.get("wins", 0) or 0)
        losses = max(0, matches - wins)
        ratio = float(d.get("win_ratio", 0) or 0)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([wins, losses], labels=[f"Wins ({wins})", f"Losses ({losses})"],
               autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{team} — Win Ratio ({ratio:.1f}%)")
        ax.text(0, -1.2, "CricGPT • get_team_stats", ha="center", fontsize=9, alpha=0.8)

        fname = f"{outdir}/venue_ratio_{team.replace(' ', '_')}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    # Top players via router
    elif act == "get_top_players" and "data" in result:
        return _plot_top_players(result["data"])

    return _plot_fallback_card({"comment": "Unsupported plot/action combo."})


def _plot_top_players(result_data: dict) -> str:
    if not isinstance(result_data, dict):
        return _plot_fallback_card({"comment": "Invalid data for Top-N plot."})

    metric = result_data.get("metric")
    top_list = result_data.get("top_players") or []
    season = result_data.get("season")
    venue = result_data.get("venue")
    city = result_data.get("city")

    if not metric or not top_list:
        return _plot_fallback_card({"comment": "Missing metric or rows for Top-N."})

    names, values = [], []
    for row in top_list:
        p = row.get("player", "Unknown")
        if metric in ("runs", "run", "score", "runs_batter"):
            val = row.get("runs_batter") if "runs_batter" in row else row.get("runs")
        elif metric in ("wicket", "wickets"):
            val = row.get("wickets")
        else:
            val = row.get(metric)
        if val is None:
            continue
        names.append(p)
        values.append(float(val))

    if not names:
        return _plot_fallback_card({"comment": "No valid rows for Top-N."})

    where_bits = []
    if season: where_bits.append(f"Season {season}")
    if city and not venue: where_bits.append(f"in {city}")
    if venue: where_bits.append(f"at {venue}")
    where_str = " • ".join(where_bits) if where_bits else ""

    metric_title = {
        "runs_batter": "Runs",
        "runs": "Runs",
        "wickets": "Wickets",
        "wicket": "Wickets",
        "strike_rate": "Strike Rate",
        "avg": "Average",
    }.get(metric, metric)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sorted_pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    names_sorted = [p for p, _ in sorted_pairs]
    vals_sorted = [v for _, v in sorted_pairs]

    bars = ax.bar(names_sorted, vals_sorted)
    ax.set_ylabel(metric_title)
    title = f"Top {len(names_sorted)} — {metric_title}"
    if where_str:
        title += f" ({where_str})"
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_xticklabels(names_sorted, rotation=20, ha="right")

    for b, v in zip(bars, vals_sorted):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/top_players_{metric}_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------
# Auto router (used by agent --plot auto)
# ---------------------------------------------------------------------
def auto_plot_from_result(result: dict, act: str) -> str:
    """
    Automatically determine the correct visualization type for a given tool result.
    Supports all Step-8 analytics tools for --plot auto.
    """
    action_to_plot = {
        # Basic stats tools
        "get_batter_stats": "form",
        "compare_players": "h2h",
        "get_team_stats": "venue-ratio",
        "get_top_players": "top",
        "llm_fallback": "fallback",
        # Advanced analytics tools
        "team_head_to_head": "h2h-team",
        "team_momentum": "momentum",
        "team_phase_dominance": "phase-dominance",
        "team_threat_map": "threat-map",
        "partnership_graph": "partnership",
        "pressure_series": "pressure",
        "win_probability_series": "win-prob",
    }

    # Handle fallback cases
    if act.startswith("llm_fallback"):
        plot_type = "fallback"
    else:
        plot_type = action_to_plot.get(act)

    if not plot_type:
        return _plot_fallback_card({"comment": f"No auto-plot available for '{act}'."})

    return generate_plot_from_result(result, act, plot_type)

# ---------------------------------------------------------------------
# New broadcast-style plots (use analytics layer)
# ---------------------------------------------------------------------
def plot_team_h2h_matrix(data: dict) -> str:
    teamA = data.get("teamA", "Mumbai Indians")
    teamB = data.get("teamB", "Chennai Super Kings")

    res = analytics.team_head_to_head(teamA, teamB,
                                      start=data.get("start"),
                                      end=data.get("end"),
                                      season=data.get("season"))
    recs = res.get("records", [])
    df = pd.DataFrame(recs)
    if df.empty:
        return _plot_fallback_card({"team": f"{teamA} vs {teamB}", "comment": "No H2H data found."})

    pivot = df.pivot_table(values="win_ratio_teamA", index="season", columns="venue")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot, aspect="auto")
    plt.colorbar(im, ax=ax, label=f"Win % ({res.get('teamA', 'Team A')})")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"Head-to-Head Win Matrix — {res.get('teamA')} vs {res.get('teamB')}")

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/team_h2h_matrix_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname


def plot_team_momentum_worm(data: dict | list) -> str:
    """
    Plot over-by-over cumulative run momentum ("worm") for two teams.

    Accepts either:
    - dict with keys {teamA, teamB, season}
    - list of dict rows with columns {team, over, runs}

    Returns path to saved PNG plot.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from . import analytics

    # --- Handle input shape ---
    if isinstance(data, list):
        # Convert list of per-over records into DataFrame
        df = pd.DataFrame(data)
        if df.empty or not {"team", "over", "runs"} <= set(df.columns):
            return _plot_fallback_card({"comment": "Momentum data unavailable."})

        teams = df["team"].unique().tolist()
        if len(teams) < 2:
            return _plot_fallback_card({"comment": "Momentum data incomplete."})
        teamA, teamB = teams[:2]
        season = df.get("season", [None])[0] if "season" in df else None

        # Compute cumulative runs per team
        df_group = df.groupby(["team", "over"])["runs"].sum().reset_index()
        df_group["cumulative_runs"] = df_group.groupby("team")["runs"].cumsum()

    else:
        # Expected dict-based structure
        teamA = data.get("teamA", "Mumbai Indians")
        teamB = data.get("teamB", "Chennai Super Kings")
        season = data.get("season", None)

        res = analytics.team_momentum(teamA, teamB, season=season)
        # res may already be list-like
        raw = res.get("data", res) if isinstance(res, dict) else res
        df = pd.DataFrame(raw)
        if df.empty or not {"team_batting", "over", "cumulative_runs"} <= set(df.columns):
            return _plot_fallback_card({"comment": "Momentum data unavailable."})
        df_group = df.rename(columns={"team_batting": "team"})

    # --- Plot ---
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))

    for team in df_group["team"].unique():
        sub = df_group[df_group["team"] == team]
        if sub.empty:
            continue
        ax.plot(sub["over"], sub["cumulative_runs"], label=team, linewidth=2)

    ax.set_xlabel("Over")
    ax.set_ylabel("Cumulative Runs")
    title = f"Momentum Worm — {teamA} vs {teamB}"
    if season:
        title += f" ({season})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/momentum_worm_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def plot_team_phase_dominance(data: dict) -> str:
    teamA = data.get("teamA", "Mumbai Indians")
    teamB = data.get("teamB", "Chennai Super Kings")
    season = data.get("season", None)

    res = analytics.team_phase_dominance(teamA, teamB, season=season)
    phases = pd.DataFrame(res.get("phases", []))
    if phases.empty or not {"team_batting", "phase", "rpo"}.issubset(phases.columns):
        return _plot_fallback_card({"comment": "Phase data unavailable."})

    # pivot to [phase] rows, [team] columns
    pv = phases.pivot_table(values="rpo", index="phase", columns="team_batting")
    pv = pv.reindex(["Powerplay", "Middle", "Death"])  # consistent order when present

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(len(pv.index))
    width = 0.35

    valsA = pv.get(teamA, pd.Series([0]*len(idx), index=pv.index)).values
    valsB = pv.get(teamB, pd.Series([0]*len(idx), index=pv.index)).values

    ax.bar(idx - width/2, valsA, width, label=teamA)
    ax.bar(idx + width/2, valsB, width, label=teamB)
    ax.set_xticks(idx)
    ax.set_xticklabels(pv.index)
    ax.set_ylabel("Runs per Over (RPO)")
    title = f"Phase Dominance — {teamA} vs {teamB}" + (f" ({season})" if season else "")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/phase_dominance_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

# ---------------------------------------------------------------------
# Advanced Analytics Visuals
# ---------------------------------------------------------------------

def plot_team_threat_map(data: dict) -> str:
    """
    Plot wickets taken by bowlers across Powerplay, Middle, and Death phases for a given team.
    """
    team = data.get("team", "Mumbai Indians")
    season = data.get("season", None)

    df = analytics.team_threat_map(team, season=season)
    if df.empty or not {"bowler", "phase", "wickets"} <= set(df.columns):
        return _plot_fallback_card({"team": team, "comment": "No threat map data available."})

    phases = ["Powerplay", "Middle", "Death"]
    pivot = df.pivot_table(values="wickets", index="bowler", columns="phase", fill_value=0)
    pivot = pivot.reindex(columns=[p for p in phases if p in pivot.columns])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(pivot))))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="plasma")
    ax.set_xlabel("Wickets")
    ax.set_ylabel("Bowler")
    ax.set_title(f"Bowling Threat Map — {team}" + (f" ({season})" if season else ""))
    ax.legend(title="Phase", loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/threat_map_{team.replace(' ', '_')}_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname


def plot_partnership_graph(data: dict, top_n: int = 10) -> str:
    """
    Simple, readable batting partnerships chart.
    - If innings_no info is present -> two stacked bar charts (1 per innings).
    - Else -> single bar chart of Top-N partnerships.
    """

    match_id = data.get("match_id")
    if not match_id:
        return _plot_fallback_card({"comment": "No match_id provided for partnership graph."})

    res = analytics.partnership_graph(match_id)
    edges = res.get("edges", [])
    if not edges:
        return _plot_fallback_card({"comment": "No partnership data available."})

    import pandas as pd
    import matplotlib.pyplot as plt

    # Build dataframe from edges
    rows = []
    for e in edges:
        a = e.get("source")
        b = e.get("target")
        w = e.get("weight", 0)
        if not a or not b:
            continue
        rows.append({
            "pair": f"{a} – {b}",
            "runs": float(w or 0),
            "innings_no": e.get("innings_no"),   # corrected
            "team": e.get("team_batting") or e.get("team")  # optional
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return _plot_fallback_card({"comment": "No valid partnership rows."})

    # Choose grouping (prefer innings_no, fallback to single)
    has_innings = "innings_no" in df.columns and df["innings_no"].notna().any()
    panels = []
    if has_innings:
        for inn in sorted(df["innings_no"].dropna().unique()):
            d = df[df["innings_no"] == inn].copy()
            if d.empty:
                continue
            d = d.sort_values("runs", ascending=False).head(top_n)
            panels.append((f"Innings {inn}", d))
    else:
        d = df.sort_values("runs", ascending=False).head(top_n)
        panels.append(("Top Partnerships", d))

    # --- Plot ---
    plt.style.use("dark_background")
    n_panels = len(panels)
    fig_h = max(3.5, 0.45 * sum(len(p[1]) for p in panels))  # auto height
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, fig_h), squeeze=False)
    axes = axes.ravel()

    for ax, (title, d) in zip(axes, panels):
        d = d.iloc[::-1]  # reverse for better label order
        ax.barh(d["pair"], d["runs"], color="#1f77b4", alpha=0.8)
        for y, val in enumerate(d["runs"]):
            ax.text(val + max(1.0, 0.02 * d["runs"].max()), y, f"{val:.0f}", va="center", fontsize=9)
        ax.set_xlabel("Partnership runs")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=12, pad=6)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)

    fig.suptitle(f"Top Partnerships — Match {match_id}", fontsize=14, weight="bold", y=0.995)
    fig.text(0.01, 0.01, "Bars = partnership runs", fontsize=9, alpha=0.9)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/partnership_top_{match_id}_{_stamp()}.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def plot_pressure_series(data: dict) -> str:
    """
    Plot current vs required run rate progression for a specific match.
    """
    match_id = data.get("match_id")
    if not match_id:
        return _plot_fallback_card({"comment": "No match_id provided for pressure series."})

    df = analytics.pressure_series(match_id)
    if df.empty or not {"over", "current_rr", "required_rr"} <= set(df.columns):
        return _plot_fallback_card({"comment": "Pressure data unavailable."})

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["over"], df["current_rr"], label="Current RR", linewidth=2)
    ax.plot(df["over"], df["required_rr"], label="Required RR", linewidth=2, linestyle="--")
    ax.set_xlabel("Over")
    ax.set_ylabel("Run Rate (RPO)")
    ax.set_title(f"Pressure Curve — Match {match_id}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/pressure_series_{match_id}_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname


def plot_win_probability_series(data: dict) -> str:
    """
    Plot evolving win probability curve over match progression.
    """
    match_id = data.get("match_id")
    if not match_id:
        return _plot_fallback_card({"comment": "No match_id provided for win probability series."})

    df = analytics.win_probability_series(match_id)
    if df.empty or not {"over", "ball_in_over", "win_prob"} <= set(df.columns):
        return _plot_fallback_card({"comment": "Win probability data unavailable."})

    df["ball_num"] = df["over"] * 6 + df["ball_in_over"]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["ball_num"], df["win_prob"] * 100, color="lime", linewidth=2)
    ax.set_xlabel("Ball Number")
    ax.set_ylabel("Win Probability (%)")
    ax.set_title(f"Win Probability (Chasing Team) — Match {match_id}")
    ax.grid(True, linestyle="--", alpha=0.3)

    outdir = _ensure_dir("outputs/plots")
    fname = f"{outdir}/win_probability_{match_id}_{_stamp()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname