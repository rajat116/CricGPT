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
    """
    Returns (start, end) ISO date strings or (None, None).
    Handles explicit years (2022, 2021-2023) and relative words
    like 'recently', 'this year', 'last year'.
    """
    text_low = text.lower()
    today = datetime.now(timezone.utc).date()
    this_year = today.year

    # explicit year(s)
    years = re.findall(r"\b(20\d{2})\b", text)
    if years:
        years = sorted(set(int(y) for y in years))
        if len(years) == 1:
            y = years[0]
            return f"{y}-01-01", f"{y}-12-31"
        else:
            return f"{years[0]}-01-01", f"{years[-1]}-12-31"

    # relative keywords
    if any(k in text_low for k in ["recent", "recently", "these days", "current form", "lately", "past year"]):
        start = (today - timedelta(days=365)).isoformat()
        end = today.isoformat()
        return start, end

    if "this year" in text_low:
        start = f"{this_year}-01-01"
        end = today.isoformat()
        return start, end

    if "last year" in text_low:
        start = f"{this_year-1}-01-01"
        end = f"{this_year-1}-12-31"
        return start, end

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
# Tool registry
# ---------------------------------------------------------------------
def tool_get_batter_stats(player, start=None, end=None):
    return cricket_query(player, role="batter", start=start, end=end)

def tool_get_bowler_stats(player, start=None, end=None):
    return cricket_query(player, role="bowler", start=start, end=end)

def tool_predict_performance(player, start=None, end=None):
    return cricket_query(player, role="predict", start=start, end=end)

TOOL_REGISTRY = {
    "get_batter_stats": tool_get_batter_stats,
    "get_bowler_stats": tool_get_bowler_stats,
    "predict_performance": tool_predict_performance,
}

# ---------------------------------------------------------------------
# Base planner
# ---------------------------------------------------------------------
class BasePlanner(ABC):
    @abstractmethod
    def decide_tool(self, question: str, trace: List[Dict[str, Any]]) -> ToolPlan:
        pass

# ---------------------------------------------------------------------
# Mock planner
# ---------------------------------------------------------------------
class MockPlanner(BasePlanner):
    def __init__(self):
        player_regex = r"(?P<player>(?:[A-Z]\.\s?)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        self.QUERY_TEMPLATES = [
            ("predict_performance", rf"(?i)(predict|forecast|form for)\s+{player_regex}"),
            ("predict_performance", rf"(?i){player_regex}\s+(next match|future performance)"),
            ("get_bowler_stats", rf"(?i)(bowling stats|wickets for|economy for)\s+{player_regex}"),
            ("get_bowler_stats", rf"(?i){player_regex}\s+(bowling stats|wickets)"),
            ("get_batter_stats", rf"(?i)(batting stats|average for|sr for|runs for|stats for)\s+{player_regex}"),
            ("get_batter_stats", rf"(?i){player_regex}\s+(batting stats|stats|record|average|sr|runs)"),
            ("get_bowler_stats", rf"(?i)(?:how did|how was)\s+{player_regex}\s+(?:bowl|bowling).*"),
            ("get_batter_stats", rf"(?i)(?:how did|how was)\s+{player_regex}\s+(?:bat|batting).*"),
            ("get_batter_stats", rf"^\s*{player_regex}\s*$"),
        ]
        self.ALIAS_MAP = {
            "rohit": "Rohit Sharma", "kohli": "Virat Kohli", "bumrah": "Jasprit Bumrah",
            "jasprit": "Jasprit Bumrah", "rahul": "KL Rahul", "kl": "KL Rahul",
            "gill": "Shubman Gill", "dhoni": "MS Dhoni",
        }

    def _clean_player(self, s: str) -> str:
        s = re.sub(r"(?i)\b(show|tell|get|give|find|predict|forecast|how|what)\b\s*", "", s)
        s = re.sub(r"(?i)\b(batting|bowling|stats?|record|average|sr|runs|wickets|economy|performance|form|match|next)\b", "", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    def decide_tool(self, question: str, trace: List[Dict[str, Any]]) -> ToolPlan:
        q = question.strip()
        for tool_name, pattern in self.QUERY_TEMPLATES:
            m = re.search(pattern, q)
            if not m:
                continue
            player = m.groupdict().get("player", "").strip()
            if not player:
                continue
            player = self._clean_player(player)
            for alias, full in self.ALIAS_MAP.items():
                if player.lower() == alias:
                    player = full
                    break
            # ✅ use new time interpreter
            start, end = interpret_time_range(q)
            return {"action": tool_name, "args": {"player": player, "start": start, "end": end},
                    "thought": f"Matched {tool_name} for {player}"}
        return {"action": "final_answer", "answer": "Sorry, could not parse the question."}

# ---------------------------------------------------------------------
# Semantic planner
# ---------------------------------------------------------------------
class SemanticPlanner(BasePlanner):
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    CONFIDENCE_THRESHOLD = 0.70
    _ALIAS_MAP = {"rohit": "Rohit Sharma","kohli": "Virat Kohli","virat": "Virat Kohli",
                  "bumrah": "Jasprit Bumrah","rahul": "KL Rahul","dhoni": "MS Dhoni","gill": "Shubman Gill"}

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not HAVE_SEMANTIC:
            raise ImportError("Please install sentence-transformers.")
        self.model = SentenceTransformer(model_name)
        self.intent_examples = {
            "get_batter_stats": [
                "batting stats for <PLAYER>", "show me <PLAYER> runs", "runs scored by <PLAYER>",
                "what is <PLAYER>'s batting average", "<PLAYER> batting average", "<PLAYER> strike rate",
                "how did <PLAYER> bat", "<PLAYER> career batting stats", "<PLAYER> runs in <YEAR>",
                "<PLAYER> batting stats between <YEAR> and <YEAR>",
            ],
            "get_bowler_stats": [
                "bowling stats for <PLAYER>", "how many wickets for <PLAYER>", "<PLAYER> wickets",
                "economy rate of <PLAYER>", "what is <PLAYER>'s bowling average", "how did <PLAYER> bowl",
                "<PLAYER> bowling stats between <YEAR> and <YEAR>", "<PLAYER> economy in <YEAR>",
            ],
            "predict_performance": [
                "predict <PLAYER> next match", "what is <PLAYER>'s form", "forecast <PLAYER> performance",
                "expected runs for <PLAYER>", "how will <PLAYER> do", "<PLAYER> form next match",
                "predict performance of <PLAYER>",
            ],
        }
        names, sentences = [], []
        for k, v in self.intent_examples.items():
            for s in v:
                names.append(k); sentences.append(s)
        self.intent_names = names
        self.intent_embeds = self.model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)

    def _extract_entities(self, text):
        # simple capitalized span
        player = None
        hit = re.search(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text)
        if hit:
            player = hit.group().strip()
        start, end = interpret_time_range(text)  # ✅ replaced manual year parsing
        masked = re.sub(player if player else "", "<PLAYER>", text, count=1)
        return player, start, end, masked.lower()

    def _rule_intent(self, q_lower: str) -> Optional[str]:
        if any(k in q_lower for k in ["predict", "forecast", "next match", "form"]): return "predict_performance"
        if any(k in q_lower for k in ["economy", "wicket", "bowling", "bowler"]): return "get_bowler_stats"
        if any(k in q_lower for k in ["batting", "average", "strike rate", "runs", "score", "scored"]): return "get_batter_stats"
        return None

    def decide_tool(self, question, trace):
        player, start, end, masked = self._extract_entities(question)
        if not player:
            log.warning(f"SemanticPlanner: No player name found in '{question}'")
            return {"action": "final_answer", "answer": "No player name found."}
        forced = self._rule_intent(question.lower())
        if forced is None:
            q_emb = self.model.encode(masked, convert_to_tensor=True, normalize_embeddings=True)
            sims = util.cos_sim(q_emb, self.intent_embeds)[0]
            idx = sims.argmax().item(); score = sims[idx].item()
            if score < self.CONFIDENCE_THRESHOLD:
                log.warning(f"SemanticPlanner: Low confidence ({score:.2f}) for '{masked}'")
                return {"action": "final_answer", "answer": "Low confidence intent match."}
            intent = self.intent_names[idx]
            thought = f"[semantic] {intent} ({score:.2f}) for {player}"
        else:
            intent = forced; thought = f"[semantic] heuristic → {intent} for {player}"
        return {"action": intent, "args": {"player": player, "start": start, "end": end}, "thought": thought}

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

You must select exactly one of these tools and include it in the JSON output:
- get_batter_stats  → when the question is about batting, runs, averages, strike rate, etc.
- get_bowler_stats  → when the question is about wickets, bowling, economy, etc.
- predict_performance → when the question asks to predict or forecast.

Extract:
1. The player's full name.
2. The time period if mentioned (e.g. 2020–2023, or single year like 2022).
3. If no year is mentioned, leave both start and end null.
4. If the user says "recent", "current", or "last few matches", set to the latest season (2025).

Return a JSON object only, formatted exactly as:
{{"action": "...", "player": "...", "start": null, "end": null}}

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

        act = j.get("action", "")
        player = j.get("player", "")
        start, end = j.get("start"), j.get("end")

        # ✅ Apply our unified date interpreter
        s2, e2 = interpret_time_range(question)
        if s2 or e2:
            start, end = s2, e2
        elif isinstance(start, (int, str)) or isinstance(end, (int, str)):
            s3, e3 = interpret_time_range(f"{start or ''} {end or ''}")
            if s3 or e3:
                start, end = s3, e3

        # if model failed to provide action, infer simple fallback
        if not act:
            q_lower = question.lower()
            if any(k in q_lower for k in ["bat", "run", "score", "average", "strike"]):
                act = "get_batter_stats"
            elif any(k in q_lower for k in ["bowl", "wicket", "economy"]):
                act = "get_bowler_stats"
            elif "predict" in q_lower or "form" in q_lower:
                act = "predict_performance"

        return {
            "action": act,
            "args": {"player": player, "start": start, "end": end},
            "thought": f"[openai] {act or 'unknown'} for {player}",
        }

# ---------------------------------------------------------------------
# Agent + CLI (unchanged)
# ---------------------------------------------------------------------
BackendChoice = Literal["auto", "mock", "semantic", "openai"]

class CricketAgent:
    def __init__(self, backend="auto", model="gpt-4o-mini", semantic_model=SemanticPlanner.DEFAULT_MODEL):
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

    def run(self, question, max_iter=3):
        trace = []; plan = self.planner.decide_tool(question, trace)
        act = plan.get("action")
        if act == "final_answer":
            return {"status": "final", "answer": plan.get("answer"), "trace": trace}
        tool = TOOL_REGISTRY.get(act)
        if not tool:
            return {"status": "error", "message": f"Unknown action {act}", "trace": trace}
        args = plan.get("args", {}); result = tool(**args)
        trace.append({"action": act, "args": args, "result": result, "thought": plan.get("thought")})
        return {"status": "final", "answer": "Tool executed", "trace": trace}

def print_clean_result(q, res):
    print(f"\n❓ Query: {q}\n" + "-"*70)
    print(json.dumps(res, indent=2, ensure_ascii=False))

def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--backend", default="auto", choices=["auto", "mock", "semantic", "openai"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--semantic_model", default=SemanticPlanner.DEFAULT_MODEL)
    args = p.parse_args()
    q = " ".join(args.question).strip()
    if not q:
        print("Please provide a question."); sys.exit(1)
    agent = CricketAgent(args.backend, args.model, args.semantic_model)
    r = agent.run(q); print_clean_result(q, r)

if __name__ == "__main__":
    main_cli()