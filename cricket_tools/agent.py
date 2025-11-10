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

def tool_compare_players(playerA, playerB, start=None, end=None, team=None, venue=None, city=None, season=None, **kwargs):
    return cricket_query("", role="compare",
                         playerA=playerA, playerB=playerB,
                         start=start, end=end, team=team, venue=venue, city=city, season=season)

def tool_get_team_stats(team, start=None, end=None, venue=None, city=None, **kwargs):
    return cricket_query("", role="team", team=team, start=start, end=end, venue=venue, city=city)

def tool_get_top_players(metric="runs_batter", season=None, n=5, venue=None, city=None, **kwargs):
    return cricket_query("", role="top", metric=metric, season=season, n=n, venue=venue, city=city)

# ---------------------------------------------------------------------
# 🆕 Step-6: new tool wrappers for team / compare / top analytics
# ---------------------------------------------------------------------
def tool_compare_players(playerA, playerB, start=None, end=None, team=None, venue=None, city=None):
    return cricket_query("", role="compare",
                         playerA=playerA, playerB=playerB,
                         start=start, end=end, team=team, venue=venue, city=city)

def tool_get_team_stats(team, start=None, end=None, venue=None, city=None):
    return cricket_query("", role="team", team=team, start=start, end=end, venue=venue, city=city)

def tool_get_top_players(metric="runs_batter", season=None, n=5, venue=None, city=None):
    return cricket_query("", role="top", metric=metric, season=season, n=n, venue=venue, city=city)

def safe_call(tool_fn, **kwargs):
    import inspect
    valid_args = {k: v for k, v in kwargs.items() if k in inspect.signature(tool_fn).parameters}
    return tool_fn(**valid_args)

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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            j = json.loads(text)
        except Exception:
            log.warning(f"OpenAI reply not valid JSON: {text}")
            return {"action": "final_answer", "answer": text}

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

        # Construct args for cricket_query()
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
            "thought": f"[openai] {act} for {player or playerA or team or metric}",
        }

# ---------------------------------------------------------------------
# Agent + CLI (unchanged runtime)
# ---------------------------------------------------------------------
BackendChoice = Literal["auto", "mock", "semantic", "openai"]

class CricketAgent:
    def __init__(self, backend="auto", model="gpt-4o-mini", semantic_model="all-MiniLM-L6-v2"):
        self.backend = self._resolve_backend(backend)
        if self.backend == "openai":
            self.planner = OpenAIPlanner(model=model); log.info(f"🔗 Using OpenAI backend ({model}).")
        elif self.backend == "semantic":
            self.planner = SemanticPlanner(model_name=semantic_model); log.info(f"🧠 Using Semantic backend ({semantic_model}).")
        else:
            self.planner = MockPlanner(); log.info("⚙️ Using Mock backend.")

    def _resolve_backend(self, backend):
        if backend != "auto": return backend
        if HAVE_OPENAI and os.getenv("OPENAI_API_KEY"): return "openai"
        if HAVE_SEMANTIC: return "semantic"
        return "mock"

    # ✅ Ensure this is indented under the class
    def run(self, question, max_iter=3):
        import inspect
        trace = []
        plan = self.planner.decide_tool(question, trace)
        act = plan.get("action")

        if act == "final_answer":
            return {"status": "final", "answer": plan.get("answer"), "trace": trace}

        tool = TOOL_REGISTRY.get(act)
        if not tool:
            return {"status": "error", "message": f"Unknown action {act}", "trace": trace}

        args = plan.get("args", {}) or {}
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
        return {"status": "final", "answer": "Tool executed", "trace": trace}

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
    p.add_argument("--backend", default="auto", choices=["auto", "mock", "semantic", "openai"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--semantic_model", default="all-MiniLM-L6-v2")
    args = p.parse_args()
    q = " ".join(args.question).strip()
    if not q:
        print("Please provide a question."); sys.exit(1)
    agent = CricketAgent(args.backend, args.model, args.semantic_model)
    r = agent.run(q); print_clean_result(q, r)

if __name__ == "__main__":
    main_cli()