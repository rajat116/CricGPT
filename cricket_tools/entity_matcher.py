#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entity_matcher.py — Intelligent entity resolver for CricGPT
Automatically recognizes city, venue, and team names in free-text queries.
"""

import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import torch
import json

DATA_PATH = Path("data/processed/ipl_deliveries.parquet")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Common aliases
ALIASES = {
    # --- Teams ---
    "rcb": "Royal Challengers Bengaluru",
    "mi": "Mumbai Indians",
    "csk": "Chennai Super Kings",
    "srh": "Sunrisers Hyderabad",
    "kkr": "Kolkata Knight Riders",
    "rr": "Rajasthan Royals",
    "gt": "Gujarat Titans",
    "delhi daredevils": "Delhi Capitals",

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


class EntityMatcher:
    def __init__(self, df: pd.DataFrame):
        self.model = MODEL
        self.aliases = {k.lower(): v for k, v in ALIASES.items()}

        # Extract unique entity lists
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
        # --- 1. Alias check
        if q in self.aliases:
            return [{"name": self.aliases[q], "score": 1.0, "reason": "alias"}]

        # --- 2. Auto-detect type if not given
        if etype is None:
            etype = self._infer_type(q)

        candidates = self.entities.get(etype, [])
        if not candidates:
            return [{"error": f"No known entities of type '{etype}'"}]

        # --- 3. Fuzzy match
        fuzzy_scores = {c: fuzz.token_sort_ratio(q, c.lower()) / 100.0 for c in candidates}

        # --- 4. Semantic similarity
        q_emb = self.model.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        sim_scores = util.cos_sim(q_emb, self.embeddings[etype])[0].cpu().tolist()

        # --- 5. Combine hybrid scores
        results = []
        for cand, fs, ss in zip(candidates, fuzzy_scores.values(), sim_scores):
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


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":
    matcher = EntityMatcher.from_dataset()
    tests = ["Banglore", "Chepuk", "RCB", "Narendra Modi", "Mumbai"]
    for t in tests:
        print(f"\n🧩 Query: {t}")
        for r in matcher.resolve(t):
            print("  →", r)