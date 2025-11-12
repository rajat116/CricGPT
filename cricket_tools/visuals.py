# ---------------------------------------------------------------------
# 🧠 Helper — adapt LLM fallback or hybrid data into DataFrame form
# ---------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def _plot_fallback_card(data: dict) -> str:
    """
    Create a simple bar or card-style visualization from fallback data.
    """
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt

    # ✅ Fix: support both player and team cases
    player = data.get("player") or data.get("team") or "Unknown"

    os.makedirs("outputs/plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    comment = data.get("comment", "")
    numeric_keys = ["runs", "avg", "strike_rate", "wickets", "economy", "centuries"]
    numeric_data = {
        k: v for k, v in data.items()
        if k in numeric_keys and isinstance(v, (int, float)) and v is not None
    }

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 3))
    
    if numeric_data:
        keys, vals = list(numeric_data.keys()), list(numeric_data.values())
        bars = ax.bar(keys, vals, color="#00b7ff")
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, val + 0.5, f"{val}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Value")
        ax.set_title(f"{player} — AI Fallback Summary")
    else:
        # no numeric data: just show comment card
        ax.axis("off")
        ax.text(0.5, 0.6, player, ha="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.4, comment, ha="center", wrap=True, fontsize=10)
        ax.set_title("CricGPT LLM Fallback View")

    fname = f"outputs/plots/fallback_{player.replace(' ', '_')}_{ts}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

def _adapt_llm_result_to_form_df(result):
    """
    Convert a fallback-style LLM result into a DataFrame that mimics
    the usual player form summary. Returns None if not applicable.
    """
    if not isinstance(result, dict):
        return None

    # LLM fallback may store structured info under "data" or top-level
    data = result.get("data") or result
    if not isinstance(data, dict):
        return None

    # Detect basic structure (player + runs, avg, sr, etc.)
    if "player" in data and any(k in data for k in ("runs", "avg", "strike_rate")):
        df = pd.DataFrame([{
            "player": data.get("player", "Unknown"),
            "runs": data.get("runs", 0),
            "avg": data.get("avg", None),
            "strike_rate": data.get("strike_rate", None),
            "comment": data.get("comment", "AI-derived fallback result"),
        }])
        return df

    return None

# ---------------------------------------------------------------------
# 🧠 Step-8: LLM-driven visualization entry point
# ---------------------------------------------------------------------
def generate_plot_from_result(result: dict, act: str, plot_type: str):
    """
    Create broadcast-style plots directly from LLM/agent tool results.
    This assumes `result` is the structured dict returned by cricket_query().

    Supported cases:
      - get_batter_stats   → player form / summary
      - compare_players    → head-to-head comparison
      - get_team_stats     → team performance / win ratio

    The chart is saved automatically to outputs/plots/.
    """
    # -- Top-N players (router) -----------------------------------------
    if plot_type == "top" and act == "get_top_players":
        data = result.get("data") or {}
        return _plot_top_players(data)

    # ------------------------------------------------------------------
    # 🧠 Handle pure LLM fallback (action == 'llm_fallback')
    # ------------------------------------------------------------------
    if act == "llm_fallback":
        data = result.get("data") or result
        return _plot_fallback_card(data)

    # ------------------------------------------------------------------
    # ✅ Handle LLM fallback (Gemini / OpenAI JSON answer with 'comment')
    # ------------------------------------------------------------------
    from datetime import datetime
    import os
    import matplotlib.pyplot as plt

    llm_df = _adapt_llm_result_to_form_df(result)
    if llm_df is not None:
        d = llm_df.iloc[0]
        player = d["player"]
        runs = float(d.get("runs", 0) or 0)
        comment = d.get("comment", "AI-derived summary")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.bar(["Runs"], [runs], color="#00b7ff")
        ax.set_title(f"{player} — AI-Derived Summary")
        ax.set_ylabel("Runs (approx.)")

        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

        # Add the AI comment below plot area
        ax.text(0.5, -0.25, comment, transform=ax.transAxes,
                fontsize=8, ha="center", va="top", wrap=True)
        ax.text(0.01, -0.15, "CricGPT LLM Fallback View",
                transform=ax.transAxes, fontsize=9, alpha=0.8)

        outdir = os.path.join("outputs", "plots")
        os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{outdir}/fallback_{player.replace(' ', '_')}_{ts}.png"

        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    import matplotlib.pyplot as plt
    import os, datetime

    def _ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _stamp() -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    outdir = _ensure_dir("outputs/plots")
    ts = _stamp()

    # --------------------------------------------------------------
    # Case 1: Player stats (form)
    # --------------------------------------------------------------
    if act == "get_batter_stats" and "data" in result:
        d = result["data"]
        player = d.get("player", "Unknown")
        avg = float(d.get("avg", 0) or 0)
        sr = float(d.get("strike_rate", 0) or 0)
        runs = float(d.get("runs", 0) or 0)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["Runs", "Average", "Strike Rate"], [runs, avg, sr], color=["#00b7ff", "#ffaa00", "#00ff88"])
        ax.set_title(f"{player} — Batting Summary")
        ax.set_ylabel("Value")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", va="bottom")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.text(0.01, -0.15, "CricGPT LLM View • get_batter_stats", transform=ax.transAxes, fontsize=9, alpha=0.8)

        fname = f"{outdir}/form_{player.replace(' ', '_')}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    # --------------------------------------------------------------
    # Case 2: Player comparison
    # --------------------------------------------------------------
    elif act == "compare_players" and "data" in result:
        cmp = result["data"].get("comparison", {})
        if len(cmp) < 2:
            raise ValueError("Comparison result incomplete; need two players.")

        players = list(cmp.keys())
        pA, pB = players[0], players[1]
        avgA = float(cmp[pA].get("avg", 0) or 0)
        avgB = float(cmp[pB].get("avg", 0) or 0)
        srA = float(cmp[pA].get("sr", 0) or 0)
        srB = float(cmp[pB].get("sr", 0) or 0)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics = ["Average", "Strike Rate"]
        x = range(len(metrics))
        width = 0.38
        ax.bar([i - width/2 for i in x], [avgA, srA], width, label=pA, color="#00b7ff")
        ax.bar([i + width/2 for i in x], [avgB, srB], width, label=pB, color="#ffaa00")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics)
        ax.set_title(f"{pA} vs {pB}")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.text(0.01, -0.15, "CricGPT LLM View • compare_players", transform=ax.transAxes, fontsize=9, alpha=0.8)

        fname = f"{outdir}/h2h_{pA.split()[0]}_vs_{pB.split()[0]}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    # --------------------------------------------------------------
    # Case 3: Team stats (win ratio)
    # --------------------------------------------------------------
    elif act == "get_team_stats" and "data" in result:
        d = result["data"]
        team = d.get("team", "Unknown")
        matches = int(d.get("matches", 0))
        wins = int(d.get("wins", 0))
        ratio = float(d.get("win_ratio", 0))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(5, 5))
        parts = [wins, max(0, matches - wins)]
        labels = [f"Wins ({wins})", f"Losses ({max(0, matches - wins)})"]
        colors = ["#00ff88", "#ff4444"]
        ax.pie(parts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        ax.set_title(f"{team} — Win Ratio ({ratio:.1f}%)")
        ax.text(0, -1.2, "CricGPT LLM View • get_team_stats", ha="center", fontsize=9, alpha=0.8)

        fname = f"{outdir}/venue_ratio_{team.replace(' ', '_')}_{ts}.png"
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fname

    # --------------------------------------------------------------
    # Case 4: Top players (bar chart)
    # --------------------------------------------------------------
    elif act == "get_top_players" and "data" in result:
        return _plot_top_players(result["data"])

    else:
        raise ValueError(f"Unsupported action or invalid result for plotting: {act}")

def _plot_top_players(result_data: dict) -> str:
    """
    Plot Top-N players for a given metric, season, and optional venue/city,
    using the structure returned by get_top_players.
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Basic validation
    if not isinstance(result_data, dict):
        raise ValueError("Invalid result data for top players.")
    metric = result_data.get("metric")  # e.g. "runs_batter" or "wicket"
    top_list = result_data.get("top_players") or []
    season = result_data.get("season")
    venue = result_data.get("venue")
    city = result_data.get("city")

    if not metric or not top_list:
        raise ValueError("Top players result missing 'metric' or 'top_players'.")

    # Extract names/values
    names = []
    values = []
    for row in top_list:
        p = row.get("player", "Unknown")
        # 🩹 normalize metric aliases
        val = (
            row.get(metric)
            or row.get("runs_batter") if metric in ("runs", "run", "score") else
            row.get("wickets") if metric in ("wicket", "wickets") else
            None
        )
        if val is None:
            continue
        names.append(p)
        values.append(float(val))

        if val is None:
            # be strict: if the metric is missing, skip (keeps math honest)
            continue
        names.append(p)
        values.append(float(val))

    if not names:
        raise ValueError("No valid rows found for the requested metric.")

    # Title pieces
    where_bits = []
    if season: where_bits.append(f"Season {season}")
    if city and not venue: where_bits.append(f"in {city}")
    if venue: where_bits.append(f"at {venue}")

    where_str = " • ".join(where_bits) if where_bits else ""
    metric_title = {
        "runs_batter": "Runs",
        "wicket": "Wickets",
        "wickets": "Wickets",
        "strike_rate": "Strike Rate",
        "avg": "Average",
    }.get(metric, metric)

    # Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Bar chart sorted by value (descending) to look more “broadcast”
    # (Do not change arithmetic—only order visually)
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

    # Annotate values on bars (no math changes)
    for b, v in zip(bars, vals_sorted):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    outdir = os.path.join("outputs", "plots")
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{outdir}/top_players_{metric}_{ts}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fname

# ---------------------------------------------------------------------
# 🤖 Auto router: pick the best plot for a given tool action
# ---------------------------------------------------------------------
def auto_plot_from_result(result: dict, act: str) -> str:
    """
    Route 'result' to the most sensible plot based on tool action.
    Only uses plot types that are already implemented in Step-8.
    """
    # Map only to the plot types we already support today
    action_to_plot = {
        "get_batter_stats": "form",
        "compare_players": "h2h",
        "get_team_stats": "venue-ratio",
        "get_top_players": "top",
        "llm_fallback": "fallback", 
        # NOTE: Still not mapping 'get_bowler_stats' (structure not shown yet)
    }

    # 🩹 make it resilient to llm_fallback_generic / llm_fallback_team / etc.
    if act.startswith("llm_fallback"):
        plot_type = "fallback"
    else:
        plot_type = action_to_plot.get(act)

    if not plot_type:
        raise ValueError(
            f"No auto-plot available for action '{act}'. "
            f"Implemented actions: {list(action_to_plot.keys())}"
        )

    return generate_plot_from_result(result, act, plot_type)