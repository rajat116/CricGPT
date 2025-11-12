#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entity_matcher.py — Intelligent entity resolver for CricGPT
Automatically recognizes city, venue, and team names in free-text queries.
"""

import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_GLOBAL_MATCHER = None  # ✅ cache global matcher instance

# ---------------------------------------------------------------------
# Common aliases (expanded for teams, cities, venues)
# ---------------------------------------------------------------------
ALIASES = {
    # --- Teams ---
    "rcb": "Royal Challengers Bengaluru",
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "srh": "Sunrisers Hyderabad",
    "kkr": "Kolkata Knight Riders",
    "rr": "Rajasthan Royals",
    "gt": "Gujarat Titans",
    "dc": "Delhi Capitals",
    "dd": "Delhi Daredevils",
    "kxip": "Kings XI Punjab",
    "pbks": "Punjab Kings",
    "lsg": "Lucknow Super Giants",
    "rps": "Rising Pune Supergiant",
    "gl": "Gujarat Lions",
    "pw": "Pune Warriors",
    "ktk": "Kochi Tuskers Kerala",

    # --- City canonicalization ---
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "blr": "Bengaluru",

    # --- Venue aliases ---
    "chepuk": "M. A. Chidambaram Stadium",
    "chepuk stadium": "M. A. Chidambaram Stadium",
    "chepauk": "M. A. Chidambaram Stadium",
    "chepauk stadium": "M. A. Chidambaram Stadium",
    "chinnaswamy": "M. Chinnaswamy Stadium",
    "wankhede": "Wankhede Stadium",
    "wankhade": "Wankhede Stadium",
    "eden": "Eden Gardens",

    # --- Narendra Modi stadium variants ---
    "narendra modi": "Narendra Modi Stadium",
    "narenda modi": "Narendra Modi Stadium",
    "namo stadium": "Narendra Modi Stadium"
}


# ---------------------------------------------------------------------
# Entity Matcher Class
# ---------------------------------------------------------------------
class EntityMatcher:
    def __init__(self, df: pd.DataFrame):
        self.model = MODEL
        self.aliases = {k.lower(): v for k, v in ALIASES.items()}

        self.entities = {
            "city": sorted(df["city"].dropna().unique().tolist()),
            "venue": sorted(df["venue"].dropna().unique().tolist()),
            "team": sorted(set(df["team_batting"]).union(df["team_bowling"]))
        }

        # Precompute embeddings for semantic similarity
        self.embeddings = {
            k: self.model.encode(v, convert_to_tensor=True, normalize_embeddings=True)
            for k, v in self.entities.items()
        }

    # --------------------------------------------------
    def resolve(self, query: str, etype: str | None = None, top_k: int = 3):
        """Resolve query to canonical entity name(s)."""
        q = query.strip().lower()

        # ✅ 1. Alias direct match (handles RCB, MI, CSK, etc.)
        if q in self.aliases:
            return [{"name": self.aliases[q], "score": 1.0, "reason": "alias"}]

        # ✅ 2. Auto-detect type if not given
        if etype is None:
            etype = self._infer_type(q)

        candidates = self.entities.get(etype, [])
        if not candidates:
            return [{"error": f"No known entities of type '{etype}'"}]

        # ✅ 3. Fuzzy similarity
        fuzzy_scores = {c: fuzz.token_sort_ratio(q, c.lower()) / 100.0 for c in candidates}

        # ✅ 4. Semantic similarity
        q_emb = self.model.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        sim_scores = util.cos_sim(q_emb, self.embeddings[etype])[0].cpu().tolist()

        # ✅ 5. Combine hybrid scores (fixed order alignment)
        results = []
        for cand, ss in zip(candidates, sim_scores):
            fs = fuzzy_scores[cand]
            hybrid = 0.6 * fs + 0.4 * ss
            reason = "hybrid"
            if fs > 0.95:
                reason = "exact"
            results.append({"name": cand, "score": round(hybrid, 4), "reason": reason})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # --------------------------------------------------
    def _infer_type(self, q: str) -> str:
        """Infer entity type based on keywords."""
        q = q.lower()
        if any(x in q for x in ["stadium", "ground", "venue", "chepuk", "wankhede", "chinnaswamy"]):
            return "venue"
        if any(x in q for x in ["indians", "knight", "royals", "capitals", "sunrisers", "titans", "rcb", "csk"]):
            return "team"
        return "city"

    # --------------------------------------------------
    @classmethod
    def from_dataset(cls, path: Path = DATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_parquet(path)
        return cls(df)


# ---------------------------------------------------------------------
# Simple Normalizer for core / agent
# ---------------------------------------------------------------------
def normalize_entity(name: str, kind: str | None = None) -> str:
    """Return canonical entity name using alias or fuzzy matching,
    handling dual IPL team names like 'Bengaluru' vs 'Bangalore'."""
    if not name:
        return name

    global _GLOBAL_MATCHER  # ✅ ensure we reuse one instance
    try:
        if _GLOBAL_MATCHER is None:
            _GLOBAL_MATCHER = EntityMatcher.from_dataset()
        matcher = _GLOBAL_MATCHER
        results = matcher.resolve(name, etype=kind, top_k=1)
        if results and "name" in results[0]:
            name = results[0]["name"]
    except Exception:
        pass

    # --- Dual/historical names ---
    TEAM_EQUIVALENTS = {
        "Royal Challengers Bangalore": ["Royal Challengers Bengaluru"],
        "Royal Challengers Bengaluru": ["Royal Challengers Bangalore"],
        "Delhi Daredevils": ["Delhi Capitals"],
        "Kings XI Punjab": ["Punjab Kings"],
        "Rising Pune Supergiants": ["Rising Pune Supergiant"],
    }

    # ✅ keep both directions for your dataset (both exist)
    CITY_SYNONYMS = {
        "Bangalore": "Bengaluru",
        "Bengaluru": "Bangalore",
        "Banglore": "Bangalore",
    }

    # ✅ Dataset-aware check for team name equivalence
    if kind == "team" and name in TEAM_EQUIVALENTS:
        try:
            from cricket_tools.filters import load_dataset
            df = load_dataset()
            variants = [name] + TEAM_EQUIVALENTS[name]
            for v in variants:
                if v in df["team_batting"].unique() or v in df["team_bowling"].unique():
                    return v
            return variants[0]
        except Exception:
            return TEAM_EQUIVALENTS.get(name, [name])[0]

    # ✅ City normalization
    if kind == "city" and name in CITY_SYNONYMS:
        return CITY_SYNONYMS[name]

    return name


# ---------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    matcher = EntityMatcher.from_dataset()
    tests = ["RCB", "Banglore", "Chepuk", "MI", "Narendra Modi"]
    for t in tests:
        print(f"\n🧩 Query: {t}")
        for r in matcher.resolve(t):
            print("  →", r)