#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cricket Chat Agent
==================

This module provides the main entry point for the Cricket Chat Agent.
"""

from __future__ import annotations
import json, sys, os, re, argparse, logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Literal
from datetime import datetime, timezone  # ✅ added for time parsing
from .memory import merge_with_memory, update_memory, clear_memory
from .llm_fallback import llm_fallback_answer
from .config import get_llm_client
import pandas as pd 
from cricket_tools import analytics

# ---------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    from openai import OpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    HAVE_SEMANTIC = True
except ImportError:
    HAVE_SEMANTIC = False

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
from .core import cricket_query
from . import visuals  # 🆕 Step-8 integrated visuals (LLM-driven)

# ---------------------------------------------------------------------
# 🔧 Helper: interpret natural or explicit time ranges
# ---------------------------------------------------------------------
def interpret_time_range(text: str):
    """Return (start, end) ISO date strings or (None, None)."""
    text_low = text.lower()
    today = datetime.now(timezone.utc).date()
    this_year = today.year

    years = re.findall(r"\b(20\d{2})\b", text)
    if years:
        years = sorted(set(int(y) for y in years))
        if len(years) == 1:
            y = years[0]
            return f"{y}-01-01", f"{y}-12-31"
        else:
            return f"{years[0]}-01-01", f"{years[-1]}-12-31"

    if any(k in text_low for k in ["recent", "recently", "these days", "current form", "lately", "past year"]):
        from datetime import timedelta
        start = (today - timedelta(days=365)).isoformat()
        end = today.isoformat()
        return start, end
    if "this year" in text_low:
        start = f"{this_year}-01-01"; end = today.isoformat(); return start, end
    if "last year" in text_low:
        start = f"{this_year-1}-01-01"; end = f"{this_year-1}-12-31"; return start, end
    return None, None

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("CricketAgent")

ToolPlan = Dict[str, Any]

# ---------------------------------------------------------------------
# Tool registry (safe to accept extra args)
# ---------------------------------------------------------------------
def tool_get_batter_stats(player, start=None, end=None, **kwargs):
    return cricket_query(player, role="batter", start=start, end=end)

def tool_get_bowler_stats(player, start=None, end=None, **kwargs):
    return cricket_query(player, role="bowler", start=start, end=end)

def tool_predict_performance(player, start=None, end=None, **kwargs):
    return cricket_query(player, role="predict", start=start, end=end)

# ---------------------------------------------------------------------
# 🆕 Step-6: new tool wrappers for team / compare / top analytics
# ---------------------------------------------------------------------
def tool_compare_players(playerA, playerB, start=None, end=None, team=None, venue=None, city=None):
    return cricket_query("", role="compare",
                         playerA=playerA, playerB=playerB,
                         start=start, end=end, team=team, venue=venue, city=city)

def tool_get_team_stats(team, start=None, end=None, season=None, venue=None, city=None, **kwargs):
    return cricket_query("", role="team", team=team, start=start, end=end, season=season, venue=venue, city=city)

def tool_get_top_players(metric="runs_batter", season=None, n=5, venue=None, city=None):
    return cricket_query("", role="top", metric=metric, season=season, n=n, venue=venue, city=city)

def safe_call(tool_fn, **kwargs):
    import inspect
    valid_args = {k: v for k, v in kwargs.items() if k in inspect.signature(tool_fn).parameters}
    return tool_fn(**valid_args)

def tool_team_head_to_head(teamA, teamB, start=None, end=None, season=None):
    return analytics.team_head_to_head(teamA, teamB, start, end, season)

def tool_team_momentum(teamA, teamB, season=None):
    return analytics.team_momentum(teamA, teamB, season)

def tool_team_phase_dominance(teamA, teamB, season=None):
    return analytics.team_phase_dominance(teamA, teamB, season)

def tool_team_threat_map(team, season=None):
    return analytics.team_threat_map(team, season)

'''def tool_partnership_graph(teamA=None, teamB=None, season=None, match_id=None):
    if not match_id and teamA and teamB:
        match_id = analytics._find_match_id(teamA, teamB, season)
    if not match_id:
        return {"error": "No match found for given teams/season."}
    res = analytics.partnership_graph(match_id)
    res["match_id"] = match_id  # 🩹 propagate to visuals
    return res'''

def tool_partnership_graph(teamA=None, teamB=None, season=None, match_id=None):
    print(f"[DEBUG] >>> called tool_partnership_graph(teamA={teamA}, teamB={teamB}, season={season}, match_id={match_id})")

    if not match_id and teamA and teamB:
        match_id = analytics._find_match_id(teamA, teamB, season)
        print(f"[DEBUG] _find_match_id() returned: {match_id}")

    if not match_id:
        print("[DEBUG] No match_id found — returning early.")
        return {"error": f"No match found for {teamA} vs {teamB} ({season})."}

    try:
        res = analytics.partnership_graph(match_id)
        print(f"[DEBUG] analytics.partnership_graph() returned type={type(res)}, keys={list(res.keys()) if isinstance(res, dict) else 'N/A'}")
    except Exception as e:
        print(f"[DEBUG] Exception while calling partnership_graph: {e}")
        res = {"error": str(e)}

    try:
        res["match_id"] = match_id
    except Exception as e:
        print(f"[DEBUG] Exception while setting match_id: {e}")
        res = {"error": f"Failed to set match_id: {e}"}

    print(f"[DEBUG] >>> final res keys: {list(res.keys()) if isinstance(res, dict) else 'N/A'}")
    return res

'''def tool_pressure_series(teamA=None, teamB=None, season=None, match_id=None):
    if not match_id and teamA and teamB:
        match_id = analytics._find_match_id(teamA, teamB, season)
    if not match_id:
        return {"error": "No match found for given teams/season."}
    res = analytics.pressure_series(match_id)
    if isinstance(res, pd.DataFrame):
        res = res.to_dict(orient="list")
    res["match_id"] = match_id
    return res'''

# In agent.py
def tool_pressure_series(teamA=None, teamB=None, season=None, match_id=None):
    if not match_id and teamA and teamB:
        match_id = analytics._find_match_id(teamA, teamB, season)
    if not match_id:
        return {"error": "No match found for given teams/season."}
    
    print(f"[DEBUG agent.py] Calling analytics.pressure_series({match_id})") # <-- ADD
    res = analytics.pressure_series(match_id)
    print(f"[DEBUG agent.py] analytics.pressure_series returned: {type(res)}") # <-- ADD
    
    if isinstance(res, pd.DataFrame):
        print(f"[DEBUG agent.py] DataFrame is empty: {res.empty}") # <-- ADD
        if not res.empty:
             print(res.head()) # <-- ADD
        res = res.to_dict(orient="list")
        
    res["match_id"] = match_id
    print(f"[DEBUG agent.py] Final tool result dict: {res}") # <-- ADD
    return res

def tool_win_probability_series(teamA=None, teamB=None, season=None, match_id=None):
    if not match_id and teamA and teamB:
        match_id = analytics._find_match_id(teamA, teamB, season)
    if not match_id:
        return {"error": "No match found for given teams/season."}
    res = analytics.win_probability_series(match_id)
    if isinstance(res, pd.DataFrame):
        res = res.to_dict(orient="list")
    res["match_id"] = match_id
    return res

# ---------------------------------------------------------------------
# Combine registry
# ---------------------------------------------------------------------
TOOL_REGISTRY = {
    "get_batter_stats": tool_get_batter_stats,
    "get_bowler_stats": tool_get_bowler_stats,
    "predict_performance": tool_predict_performance,
    # 🆕 Step-6 additions
    "compare_players": tool_compare_players,
    "get_team_stats": tool_get_team_stats,
    "get_top_players": tool_get_top_players,
    # Step-8.5 analytics tools
    "team_head_to_head": tool_team_head_to_head,
    "team_momentum": tool_team_momentum,
    "team_phase_dominance": tool_team_phase_dominance,
    "team_threat_map": tool_team_threat_map,
    "partnership_graph": tool_partnership_graph,
    "pressure_series": tool_pressure_series,
    "win_probability_series": tool_win_probability_series,
}

# ---------------------------------------------------------------------
# Base planner (unchanged)
# ---------------------------------------------------------------------
class BasePlanner(ABC):
    @abstractmethod
    def decide_tool(self, question: str, trace: List[Dict[str, Any]]) -> ToolPlan:
        pass

# ---------------------------------------------------------------------
# Mock planner (expanded patterns for Step-6)
# ---------------------------------------------------------------------
class MockPlanner(BasePlanner):
    def __init__(self):
        player_regex = r"(?P<player>(?:[A-Z]\.\s?)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        compare_regex = r"(?P<playerA>[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:and|vs\.?|versus)\s+(?P<playerB>[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        team_regex = r"(?P<team>[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        self.QUERY_TEMPLATES = [
            ("predict_performance", rf"(?i)(predict|forecast|form for)\s+{player_regex}"),
            ("predict_performance", rf"(?i){player_regex}\s+(next match|future performance)"),
            ("get_bowler_stats", rf"(?i)(bowling stats|wickets for|economy for)\s+{player_regex}"),
            ("get_batter_stats", rf"(?i)(batting stats|average for|sr for|runs for|stats for)\s+{player_regex}"),
            # 🆕 Step-6 new intents
            ("compare_players", rf"(?i)compare\s+{compare_regex}"),
            ("get_team_stats", rf"(?i)(how did|show)\s+{team_regex}\s+(perform|do|play)"),
            ("get_top_players", rf"(?i)top\s+(\d+)?\s*(run|wicket|score)[a-z]*\s+(players|takers)?"),
            ("get_batter_stats", rf"^\s*{player_regex}\s*$"),
        ]
        self.ALIAS_MAP = {
            "rohit": "Rohit Sharma", "kohli": "Virat Kohli", "bumrah": "Jasprit Bumrah",
            "jasprit": "Jasprit Bumrah", "rahul": "KL Rahul", "kl": "KL Rahul",
            "gill": "Shubman Gill", "dhoni": "MS Dhoni",
        }

    def _clean_player(self, s: str) -> str:
        s = re.sub(r"(?i)\b(show|tell|get|give|find|predict|forecast|how|what|compare|versus|vs)\b\s*", "", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    def decide_tool(self, question: str, trace: List[Dict[str, Any]]) -> ToolPlan:
        q = question.strip()
        for tool_name, pattern in self.QUERY_TEMPLATES:
            m = re.search(pattern, q)
            if not m:
                continue
            gd = m.groupdict()
            start, end = interpret_time_range(q)
            if tool_name == "compare_players" and gd.get("playerA") and gd.get("playerB"):
                return {"action": tool_name,
                        "args": {"playerA": gd["playerA"], "playerB": gd["playerB"], "start": start, "end": end},
                        "thought": f"Matched compare for {gd['playerA']} vs {gd['playerB']}"}
            if tool_name == "get_team_stats" and gd.get("team"):
                return {"action": tool_name,
                        "args": {"team": gd["team"], "start": start, "end": end},
                        "thought": f"Matched team stats for {gd['team']}"}
            if tool_name == "get_top_players":
                n = int(m.group(1)) if m.group(1) else 5
                metric = "runs_batter" if "run" in q.lower() else "wicket"
                return {"action": tool_name,
                        "args": {"metric": metric, "n": n},
                        "thought": f"Matched top {n} {metric} query"}
            # Default single-player path
            player = gd.get("player", "")
            for alias, full in self.ALIAS_MAP.items():
                if player.lower() == alias:
                    player = full; break
            return {"action": tool_name, "args": {"player": player, "start": start, "end": end},
                    "thought": f"Matched {tool_name} for {player}"}
        return {"action": "final_answer", "answer": "Sorry, could not parse the question."}

# ---------------------------------------------------------------------
# Semantic planner
# ---------------------------------------------------------------------
class SemanticPlanner(BasePlanner):
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    CONFIDENCE_THRESHOLD = 0.70
    _ALIAS_MAP = {
        "rohit": "Rohit Sharma", "kohli": "Virat Kohli", "virat": "Virat Kohli",
        "bumrah": "Jasprit Bumrah", "rahul": "KL Rahul", "dhoni": "MS Dhoni", "gill": "Shubman Gill"
    }

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not HAVE_SEMANTIC:
            raise ImportError("Please install sentence-transformers.")
        self.model = SentenceTransformer(model_name)

        # -------------------------------------------------------
        # 🧩 Step-6: Expanded intent library
        # -------------------------------------------------------
        self.intent_examples = {
            "get_batter_stats": [
                "batting stats for <PLAYER>", "show me <PLAYER> runs", "runs scored by <PLAYER>",
                "what is <PLAYER>'s batting average", "<PLAYER> strike rate",
                "how did <PLAYER> bat", "<PLAYER> career batting stats", "<PLAYER> runs in <YEAR>"
            ],
            "get_bowler_stats": [
                "bowling stats for <PLAYER>", "how many wickets for <PLAYER>", "<PLAYER> wickets",
                "economy rate of <PLAYER>", "what is <PLAYER>'s bowling average", "how did <PLAYER> bowl"
            ],
            "predict_performance": [
                "predict <PLAYER> next match", "what is <PLAYER>'s form", "forecast <PLAYER> performance",
                "expected runs for <PLAYER>", "how will <PLAYER> do", "<PLAYER> form next match"
            ],
            # 🆕 Step-6 new intents
            "compare_players": [
                "compare <PLAYER> and <PLAYER>", "compare performance of <PLAYER> and <PLAYER>",
                "who performed better between <PLAYER> and <PLAYER>", "<PLAYER> vs <PLAYER> stats"
            ],
            "get_team_stats": [
                "team performance of <TEAM>", "how did <TEAM> perform", "show <TEAM> stats",
                "<TEAM> results this season", "how many wins for <TEAM>"
            ],
            "get_top_players": [
                "top <N> run scorers", "top <N> wicket takers", "best batsmen this year",
                "top bowlers in <YEAR>", "who scored the most runs"
            ],
        }

        # Encode all intent examples
        names, sentences = [], []
        for k, v in self.intent_examples.items():
            for s in v:
                names.append(k)
                sentences.append(s)
        self.intent_names = names
        self.intent_embeds = self.model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)

    # -------------------------------------------------------
    # Entity extraction
    # -------------------------------------------------------
    def _extract_entities(self, text):
        # detect numeric n (like "top 5")
        n_match = re.search(r"top\s+(\d+)", text, re.IGNORECASE)
        n = int(n_match.group(1)) if n_match else None

        # detect city/venue ("in Mumbai", "at Wankhede")
        venue, city = None, None
        m_venue = re.search(r"at\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        m_city = re.search(r"in\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        if m_venue: venue = m_venue.group(1)
        if m_city: city = m_city.group(1)

        # simple capitalized player/team extraction
        ents = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text)
        start, end = interpret_time_range(text)
        masked = re.sub("|".join(ents) if ents else "", "<ENTITY>", text, count=1)
        return ents, start, end, n, venue, city, masked.lower()

    # -------------------------------------------------------
    # Simple rule-based fallback (fast classification)
    # -------------------------------------------------------
    def _rule_intent(self, q_lower: str) -> Optional[str]:
        if any(k in q_lower for k in ["compare", "versus", "vs"]): return "compare_players"
        if "top" in q_lower or "most" in q_lower: return "get_top_players"
        if any(k in q_lower for k in ["team", "how did", "perform", "wins", "match result"]): return "get_team_stats"
        if any(k in q_lower for k in ["predict", "forecast", "form"]): return "predict_performance"
        if any(k in q_lower for k in ["economy", "wicket", "bowling", "bowler"]): return "get_bowler_stats"
        if any(k in q_lower for k in ["batting", "average", "strike rate", "runs", "score"]): return "get_batter_stats"
        return None

    # -------------------------------------------------------
    # Main decision function
    # -------------------------------------------------------
    def decide_tool(self, question, trace):
        ents, start, end, n, venue, city, masked = self._extract_entities(question)
        q_lower = question.lower()

        forced = self._rule_intent(q_lower)
        if forced:
            thought = f"[semantic] heuristic → {forced}"
            intent = forced
        else:
            q_emb = self.model.encode(masked, convert_to_tensor=True, normalize_embeddings=True)
            sims = util.cos_sim(q_emb, self.intent_embeds)[0]
            idx = sims.argmax().item(); score = sims[idx].item()
            if score < self.CONFIDENCE_THRESHOLD:
                return {"action": "final_answer", "answer": "Low confidence semantic match."}
            intent = self.intent_names[idx]
            thought = f"[semantic] {intent} ({score:.2f})"

        args = {}
        if intent == "compare_players" and len(ents) >= 2:
            args = {"playerA": ents[0], "playerB": ents[1], "start": start, "end": end}
        elif intent == "get_team_stats" and ents:
            args = {"team": ents[0], "start": start, "end": end, "venue": venue, "city": city}
        elif intent == "get_top_players":
            metric = "runs_batter" if "run" in q_lower else "wicket"
            args = {"metric": metric, "season": start[:4] if start else None, "n": n or 5, "venue": venue, "city": city}
        else:
            if ents:
                args = {"player": ents[0], "start": start, "end": end}

        return {"action": intent, "args": args, "thought": thought}

class OpenAIPlanner(BasePlanner):
    def __init__(self, model="gpt-4o-mini"):
        if not HAVE_OPENAI:
            raise ImportError("OpenAI SDK not installed.")

        # ✅ Use unified LLM config (OpenAI, Gemini, or Ollama)
        provider, model_name, client = get_llm_client()

        self.provider = provider
        self.model = model_name
        self.client = client

    def decide_tool(self, question, trace):
        prompt = f"""
You are a precise cricket query planner.

Decide the correct *action* and structured arguments for the query below.

Available tools:
- get_batter_stats(player, start, end)
- get_bowler_stats(player, start, end)
- predict_performance(player, start, end)
- compare_players(playerA, playerB, start, end)
- get_team_stats(team, start, end, venue, city)
- get_top_players(metric, season, n, venue, city)
- team_head_to_head(teamA, teamB, season)
- team_momentum(teamA, teamB, season)
- team_phase_dominance(teamA, teamB, season)
- team_threat_map(team, season)
- partnership_graph(teamA, teamB, season)
- pressure_series(teamA, teamB, season)
- win_probability_series(teamA, teamB, season)

Guidelines:
- "compare", "vs", or "versus" → compare_players
- "how did <TEAM>" or "performance of <TEAM>" → get_team_stats
- "top" or "most runs/wickets" → get_top_players
- "bat", "run", "average", "strike" → get_batter_stats
- "bowl", "wicket", "economy" → get_bowler_stats
- "predict", "forecast", "form" → predict_performance
- Include numeric 'n' if user says "top 5", "top 10", etc.
- Detect venue/city words like "in Chennai", "at Wankhede" if present.

Return *only* a valid JSON object:
{{
  "action": "<tool_name>",
  "player": null,
  "playerA": null,
  "playerB": null,
  "team": null,
  "metric": null,
  "season": null,
  "n": null,
  "venue": null,
  "city": null,
  "start": null,
  "end": null
}}

Question: "{question}"
"""

        # --- Branch depending on provider ---
        if self.provider == "gemini":
            # ✅ Gemini API requires creating a GenerativeModel instance
            model_obj = self.client.GenerativeModel(self.model)
            response = model_obj.generate_content(prompt)
            text = response.text.strip()
        else:
            # Default: OpenAI-compatible
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()

        # --- Parse JSON output (robust across LLMs) ---
        clean_text = text.strip()

        # ✅ Gemini and some LLMs wrap responses in markdown blocks
        clean_text = re.sub(r"^```(?:json)?", "", clean_text, flags=re.IGNORECASE).strip()
        clean_text = re.sub(r"```$", "", clean_text).strip()

        # ✅ Sometimes Gemini adds multiple code fences or extra newlines
        clean_text = clean_text.strip("` \n")

        try:
            j = json.loads(clean_text)
        except Exception as e:
            log.warning(f"LLM reply not valid JSON, returning raw text. Error: {e}\n{text}")
            return {"action": "final_answer", "answer": clean_text}

        # Extract fields
        act = j.get("action", "")
        start, end = j.get("start"), j.get("end")
        player = j.get("player")
        playerA, playerB = j.get("playerA"), j.get("playerB")
        team = j.get("team")
        venue, city = j.get("venue"), j.get("city")
        metric, season, n = j.get("metric"), j.get("season"), j.get("n")

        # Apply fallback date interpreter
        s2, e2 = interpret_time_range(question)
        if s2 or e2:
            start, end = s2, e2

        # Fallback heuristics if model misses action
        q_low = question.lower()
        if not act:
            if any(k in q_low for k in ["compare", "vs", "versus"]):
                act = "compare_players"
            elif "top" in q_low:
                act = "get_top_players"
            elif any(k in q_low for k in ["how did", "perform", "team"]):
                act = "get_team_stats"
            elif any(k in q_low for k in ["predict", "forecast", "form"]):
                act = "predict_performance"
            elif any(k in q_low for k in ["bowl", "wicket", "economy"]):
                act = "get_bowler_stats"
            else:
                act = "get_batter_stats"

        args = {
            "player": player,
            "playerA": playerA,
            "playerB": playerB,
            "team": team,
            "metric": metric,
            "season": season,
            "n": n,
            "venue": venue,
            "city": city,
            "start": start,
            "end": end,
        }

        return {
            "action": act,
            "args": {k: v for k, v in args.items() if v is not None},
            "thought": f"[{self.provider}] {act} for {player or playerA or team or metric}",
        }

# ---------------------------------------------------------------------
# 🧠 Reasoning LLM Planner (Hybrid IPL Reasoning + Retry + Fallback)
# ---------------------------------------------------------------------
class ReasoningLLMPlanner(BasePlanner):
    MAX_RETRIES = 3

    def __init__(self, model="gpt-4o-mini"):
        provider, model_name, client = get_llm_client()
        self.provider = provider
        self.model = model_name
        self.client = client

    # -----------------------------------------------------------------
    def _try_reason_once(self, question: str, prompt_suffix: str = "") -> dict | None:
        """Perform one reasoning attempt and return parsed JSON or None."""
        prompt = f"""
You are a cricket analytics reasoning engine specialized *only* for the Indian Premier League (IPL) dataset.

Your goal: decide which analysis tool to use. You have **two options only**:
1.  llm_fallback → for general cricket knowledge or any non-IPL tournament.
2.  IPL structured tools → for IPL-specific data such as player or team performance.

---

### RULE 1 — Non-IPL Check (HIGHEST PRIORITY)
First, scan the query for explicit **non-IPL tournament keywords**.

Non-IPL keywords:
World Cup, ODI, Test, Asia Cup, T20I, T20 World Cup, ICC, bilateral series, against England, against Australia, against Pakistan

If you find **any** of these keywords, you must immediately choose:
action = "llm_fallback"

Do **not** use structured tools if these words appear, even if player names or years are mentioned.

**Examples of RULE 1**
- Query: "How did Rohit Sharma perform in 2016 ODI World Cup"
  → Contains "ODI World Cup" → action = "llm_fallback"
- Query: "Top wicket takers in ICC World Cup 2019"
  → Contains "ICC World Cup" → action = "llm_fallback"
- Query: "Best batsmen in bilateral series against England"
  → Contains "bilateral series" → action = "llm_fallback"

---

### RULE 2 — IPL Tool Mapping (Only if Rule 1 does NOT apply)
If and only if the query does **not** contain any non-IPL keyword from Rule 1, assume it refers to the IPL.

Then map it to the correct structured tool:

Available structured tools:
- get_batter_stats(player, start, end)
- get_bowler_stats(player, start, end)
- predict_performance(player, start, end)
- compare_players(playerA, playerB, start, end)
- get_team_stats(team, start, end, venue, city)
- get_top_players(metric, season, n, venue, city)
- team_head_to_head(teamA, teamB, season)
- team_momentum(teamA, teamB, season)
- team_phase_dominance(teamA, teamB, season)
- team_threat_map(team, season)
- partnership_graph(teamA, teamB, season)
- pressure_series(teamA, teamB, season)
- win_probability_series(teamA, teamB, season)

Mapping hints:
- Phrases with "top", "most", "highest runs", or "wickets" → get_top_players
- "compare", "vs", or "between" → compare_players
- "how did <TEAM> perform", "team stats" → get_team_stats
- "bat", "average", "runs", "form" → get_batter_stats
- "bowl", "wickets", "economy" → get_bowler_stats
- "head-to-head", "vs", "versus" + two team names → team_head_to_head
- "momentum worm", "worm chart", "over-by-over" → team_momentum
- "phase comparison", "phase dominance", "powerplay vs death" → team_phase_dominance
- "threat map", "bowling threat", "phase wickets", "bowler impact" → team_threat_map
- "partnership graph", "batting partnership", "pair runs", "links between batsmen" → partnership_graph
  (Requires two team names and optional season; automatically resolves match_id.)
- "pressure curve", "required run rate", "current run rate", "rpo comparison" → pressure_series
  (Works for any IPL match between two teams in a season.)
- "win probability", "chance to win", "probability curve", "match win chances" → win_probability_series
  (Uses the same inferred match context automatically.)
- Always include the detected year as "season" or as ("start","end").

---

### OUTPUT FORMAT
Return only a valid JSON object (no markdown, no code fences):

{{
  "action": "<tool_name_or_llm_fallback>",
  "args": {{
    "player": null,
    "playerA": null,
    "playerB": null,
    "team": null,
    "teamA": null,
    "teamB": null,
    "start": null,
    "end": null,
    "venue": null,
    "city": null,
    "season": null,
    "metric": null,
    "n": null,
    "topic": null
  }},
  "reasoning": "<short explanation of why you chose this action>"
}}

Query: "{question}"
{prompt_suffix}
"""
        try:
            if self.provider == "gemini":
                model_obj = self.client.GenerativeModel(self.model)
                response = model_obj.generate_content(prompt)
                text = response.text.strip()
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,
                )
                text = resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Reasoning LLM call failed: {e}")
            return None

        raw = text.strip()
        clean = re.sub(r"^```(?:json)?", "", raw).strip()
        clean = re.sub(r"```$", "", clean).strip("` \n")

        try:
            return json.loads(clean)
        except Exception as e:
            log.warning(f"Invalid JSON in reasoning output: {e}\n{text}")
            return None

    # -----------------------------------------------------------------
    def decide_tool(self, question, trace):
        trace_local = []
        result = None

        for i in range(self.MAX_RETRIES):
            suffix = ""
            if i == 1:
                suffix = "\nRetry 2: Rephrase internally if needed so it fits IPL seasons, venues, or players."
            elif i == 2:
                suffix = "\nRetry 3: Final attempt — assume the user meant IPL context unless the question explicitly says World Cup, Asia Cup, ODI, or T20I."

            j = self._try_reason_once(question, suffix)
            if j:
                trace_local.append({
                    "attempt": i + 1,
                    "action": j.get("action"),
                    "reasoning": j.get("reasoning")
                })
                if j.get("action"):
                    result = j
                    # ✅ stop immediately if the model already decided fallback
                    if j["action"] == "llm_fallback" or j["action"].startswith("llm_fallback"):
                        break
                    # otherwise keep only when it's a structured IPL tool
                    if j["action"] != "llm_fallback":
                        break
            else:
                trace_local.append({"attempt": i + 1, "error": "no valid JSON"})

        if not result:
            result = {
                "action": "llm_fallback",
                "args": {"topic": question},
                "reasoning": "All 3 reasoning attempts failed to produce valid plan."
            }

        args = result.get("args", {}) or {}

        s2, e2 = interpret_time_range(question)
        if s2 or e2:
            args["start"], args["end"] = s2, e2

        log.info(f"🧩 Reasoning result → {result.get('action')} | args={args}")

        return {
            "action": result.get("action", "get_batter_stats"),
            "args": args,
            "thought": f"[reasoning] {result.get('reasoning', 'No reasoning given')}",
            "retries": trace_local
        }

# ---------------------------------------------------------------------
# Agent + CLI (unchanged runtime)
# ---------------------------------------------------------------------
BackendChoice = Literal["auto", "mock", "semantic", "llm"]

class CricketAgent:
    def __init__(self, backend="auto", model="gpt-4o-mini", semantic_model="all-MiniLM-L6-v2"):
        self.backend = self._resolve_backend(backend)
        if self.backend == "llm":
            self.planner = OpenAIPlanner(model=model)
            log.info(f"🔗 Using LLM backend ({self.planner.provider}, {self.planner.model}).")
        elif self.backend == "llm_reasoning":
            self.planner = ReasoningLLMPlanner(model=model)
            log.info(f"🧠 Using Reasoning LLM backend ({self.planner.provider}, {self.planner.model}).")
        elif self.backend == "semantic":
            self.planner = SemanticPlanner(model_name=semantic_model); log.info(f"🧠 Using Semantic backend ({semantic_model}).")
        else:
            self.planner = MockPlanner(); log.info("⚙️ Using Mock backend.")

    def _resolve_backend(self, backend):
        if backend != "auto":
            return backend
        # Auto-select based on available environment
        if os.getenv("LLM_PROVIDER"):
            return "llm"
        if HAVE_SEMANTIC:
            return "semantic"
        return "mock"

    # ✅ Ensure this is indented under the class
    def run(self, question, max_iter=3):
        import inspect
        trace = []
        # --- STEP-7 Memory merge before planning ---
        # We don’t yet know entities, but after planner we can merge remembered context.
        plan = self.planner.decide_tool(question, trace)
        act = plan.get("action")

        # --- STEP-7 Memory merge before planning ---
        args_from_plan = plan.get("args", {}) or {}

        # Pull memory but DO NOT overwrite explicit plan args
        mem_args = merge_with_memory(args_from_plan) or {}

        # 1) Plan args must win over memory
        merged_args = {**mem_args, **args_from_plan}

        # 2) If the current plan didn't mention venue/city, strip them (avoid stale filters)
        if "venue" not in args_from_plan:
            merged_args.pop("venue", None)
        if "city" not in args_from_plan:
            merged_args.pop("city", None)

        # 3) Never drop explicit season/start/end
        for k in ("season", "start", "end", "team"):
            if k in args_from_plan and args_from_plan[k] is not None:
                merged_args[k] = args_from_plan[k]

        # (optional) log what memory actually contributed
        mem_used = {k: v for k, v in merged_args.items()
                    if k not in args_from_plan and v is not None}
        if mem_used:
            log.info(f"🧠 Using memory (filtered): {mem_used}")

        plan["args"] = merged_args

        # 🧠 If memory contributed something, show it once
        mem_used = {k: v for k, v in merged_args.items() if k not in args_from_plan and v}
        if mem_used:
            log.info(f"🧠 Using memory: {mem_used}")

        plan["args"] = merged_args

        # 🩹 Patch: if start/end belong to same year, auto-fill season
        args = plan.get("args", {}) or {}
        if args.get("start") and args.get("end"):
            try:
                y1 = str(args["start"])[:4]
                y2 = str(args["end"])[:4]
                if y1 == y2:
                    args["season"] = y1
                    print(f"[DEBUG] Auto-inferred season={y1} from start/end")
            except Exception:
                pass
        plan["args"] = args  # update plan after patch

        if act == "final_answer":
            return {"status": "final", "answer": plan.get("answer"), "trace": trace}

        # --- 🧩 Direct handling of reasoning-based LLM fallback ---
        if act == "llm_fallback":
            fb = llm_fallback_answer(question)
            fb["reasoning"] = plan.get("thought", "Triggered LLM fallback reasoning.")
            trace.append({
                "action": "llm_fallback",
                "args": plan.get("args", {}),
                "result": fb,
                "thought": fb.get("reasoning", "")
            })
            return fb

        tool = TOOL_REGISTRY.get(act)
        if not tool:
            return {"status": "error", "message": f"Unknown action {act}", "trace": trace}

        args = plan.get("args", {}) or {}
        # 🩹 carry over season if auto-inferred earlier
        if "season" not in args and plan.get("args", {}).get("season"):
            args["season"] = plan["args"]["season"]
        valid_params = inspect.signature(tool).parameters
        clean_args = {k: v for k, v in args.items() if k in valid_params}

        try:
            result = tool(**clean_args)
        except Exception as e:
            result = {"error": str(e), "args": clean_args}

        trace.append({
            "action": act,
            "args": clean_args,
            "result": result,
            "thought": plan.get("thought")
        })

        # --- STEP-7 Update memory after successful tool run ---
        update_memory(clean_args)

        # --- 🔍 STOP EARLY IF RESULT IS AMBIGUOUS ---
        if isinstance(result, dict) and result.get("status") == "ambiguous":
            return {
                "status": "ambiguous",
                "options": result.get("options", []),
                "hint": result.get("hint", ""),
                "query": clean_args.get("player")
            }

        # --- 🧠 Fallback: if tool failed or returned no data ---
        #use_fallback = getattr(self, "fallback_enabled", False) or \
        #               os.getenv("ENABLE_LLM_FALLBACK", "false").lower() == "true"
        use_fallback = True

        # ✅ DEFINE a list of tools that return raw data (dicts/DataFrames)
        #    and do NOT follow the {"data": ...} schema.
        RAW_DATA_TOOLS = [
            "team_head_to_head",
            "team_momentum",
            "team_phase_dominance",
            "team_threat_map",
            "partnership_graph",
            "pressure_series",
            "win_probability_series"
        ]

        if use_fallback and (isinstance(result, dict) or isinstance(result, pd.DataFrame)):
            
            # ✅ Smarter empty detection
            empty_or_error = False # Start with False

            if isinstance(result, dict) and "error" in result:
                empty_or_error = True
            elif isinstance(result, dict) and result.get("status") == "error":
                empty_or_error = True
            elif act not in RAW_DATA_TOOLS:
                # --- This is a STATS tool, it MUST have a 'data' key ---
                if isinstance(result, dict):
                    empty_or_error = (
                        (result.get("data") in (None, {}, []))
                        or (isinstance(result.get("data"), dict) and "note" in result["data"]
                            and "no data" in str(result["data"]["note"]).lower())
                    )
                else:
                    # Not a dict, must be an error
                    empty_or_error = True
            
            elif act in RAW_DATA_TOOLS:
                # --- This is a RAW tool, check for its specific empty state ---
                if isinstance(result, pd.DataFrame) and result.empty:
                    empty_or_error = True
                elif isinstance(result, dict):
                    if act == "partnership_graph" and not result.get("nodes"):
                        empty_or_error = True
                    elif act == "team_momentum" and not result.get("data"):
                        empty_or_error = True
                    elif act == "team_phase_dominance" and not result.get("phases"):
                        empty_or_error = True
                    elif act == "pressure_series" and not result.get("over"):
                        empty_or_error = True
                    elif act == "win_probability_series" and not result.get("over"): # Check for 'over' or 'win_prob'
                        empty_or_error = True
                    elif act == "team_head_to_head" and not result.get("records"):
                        empty_or_error = True
                    elif act == "team_threat_map" and isinstance(result, pd.DataFrame) and result.empty:
                        empty_or_error = True # This one returns a DF, covered above, but good to be explicit
            
            # (End of new logic)

            if empty_or_error:
                fb = llm_fallback_answer(question)
                fb["answer"] = f"📊 No structured data found for this query — using AI reasoning:\n\n{fb['answer']}"
                
                # Append the fallback step to the trace
                trace.append({
                    "action": "llm_fallback",
                    "args": {},
                    "result": fb,
                    "thought": "Triggered fallback due to empty/error tool result"
                })
                
                # --- ⬇️ FIX ⬇️ ---
                # Return the FULL trace and complete response
                final_response = {
                    "status": "final", 
                    "answer": fb["answer"], # Use the fallback answer
                    "trace": trace,         # Return the full trace
                    
                    # Also merge the fallback 'data' and 'action' for consistency
                    "action": fb.get("action", "llm_fallback_generic"),
                    "args": fb.get("args", {}),
                    "data": fb.get("data", {}),
                    "reasoning": fb.get("reasoning", "Structured fallback")
                }
                return final_response

        # --------------------------------------------------------
        # Unified final result for formatter
        # --------------------------------------------------------
        final_res = {
            "status": "final",
            "action": act,
            "args": clean_args,
            "data": result,
            "trace": trace,
        }

        # Natural-language mode
        if getattr(self, "natural_language", False):
            from .llm_formatter import format_natural_answer
            final_res["answer"] = format_natural_answer(question, final_res)
            return final_res

        # Default JSON mode
        return final_res


def print_clean_result(q, res):
    print(f"\n❓ Query: {q}\n" + "-"*70)
    if res.get("trace"):
        for step in res["trace"]:
            args = step.get("args", {})
            if any(k in args for k in ("team", "venue", "city")):
                print(f"🧭 Resolved entities: { {k: args[k] for k in ('team','venue','city') if k in args} }")
    print(json.dumps(res, indent=2, ensure_ascii=False))

def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--backend", default="llm_reasoning", choices=["auto", "mock", "semantic", "llm", "llm_reasoning"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--semantic_model", default="all-MiniLM-L6-v2")
    p.add_argument("--clear-memory", action="store_true", help="Clear previous session context")
    p.add_argument("--fallback", action="store_true", help="Use LLM fallback for unknown or empty responses")
    # --- 🆕 Step-8 Visualization flags ---
    p.add_argument("--plot", choices=["auto", "form", "h2h", "top", "venue-ratio", "h2h-team", "momentum", "phase-dominance", "threat-map", "partnership", "pressure", "win-prob"],
                   help="Generate broadcast-style or analytics plot (auto | form | top | h2h | venue-ratio | h2h-team | momentum | phase-dominance | threat-map | partnership | pressure | win-prob)")
    p.add_argument("--players", nargs="+",
                   help="Player(s) for plotting (one for form; two for h2h)")
    p.add_argument("--team",
                   help="Team name (required for venue-ratio plots)")
    p.add_argument("--nl", action="store_true",
                   help="Return natural-language answer instead of JSON")

    args = p.parse_args()
    q = " ".join(args.question).strip()
    # --- 🧩 Step-8: LLM-driven visualization routing ---
    if args.plot:
        print("🎨 Generating visualization through LLM agent…")
        agent = CricketAgent(args.backend, args.model, args.semantic_model)
        agent.natural_language = args.nl
        # ✅ Always allow fallback during plotting so we can visualize even when IPL data is missing
        agent.fallback_enabled = True
        # Step 1: Run LLM/semantic pipeline to get structured result
        res = agent.run(q)

        # Step 2: Find last executed tool (if any)
        if not res.get("trace"):
            print("❌ No structured result found for visualization.")
            sys.exit(1)

        last_step = next((s for s in reversed(res["trace"]) if "result" in s), None)
        if not last_step or "result" not in last_step:
            print("❌ No tool result available for plotting.")
            sys.exit(1)

        result_data = last_step["result"]
        act = last_step.get("action")

        # Step 3: Generate visualization from tool result
        if args.plot == "auto":
            saved_path = visuals.auto_plot_from_result(result_data, act)
        else:
            saved_path = visuals.generate_plot_from_result(result_data, act, plot_type=args.plot)
        print(f"✅ Plot saved at: {saved_path}")
        return

    if not q:
        print("Please provide a question."); sys.exit(1)
    agent = CricketAgent(args.backend, args.model, args.semantic_model)
    agent.fallback_enabled = args.fallback
    if args.clear_memory:
        clear_memory()
        print("🧹 Cleared session memory.")

    r = agent.run(q)

    # -----------------------------------------------------------
    # PHASE-2 Unified Natural-Language Output via run_query()
    # -----------------------------------------------------------
    if args.nl:
        from cricket_tools.runner import run_query

        resp = run_query(
            q,
            backend=args.backend,
            plot=(args.plot == "auto"),
            fallback=args.fallback,
            session_id=None,
        )

        print("\n" + resp["reply"] + "\n")
        if resp.get("plot_path"):
            print(f"📊 Plot saved at: {resp['plot_path']}")
        return

    # -----------------------------------------------
    # Default JSON output
    # -----------------------------------------------
    print_clean_result(q, r)

if __name__ == "__main__":
    main_cli()