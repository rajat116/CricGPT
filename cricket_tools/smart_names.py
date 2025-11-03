from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from rapidfuzz import process, fuzz
import jellyfish
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------
# Paths
CSV_PATH = Path("data/processed/player_names.csv")
PARQUET_PATH = Path("data/processed/ipl_deliveries.parquet")

# ---------------------------------------------------------------------
# Global semantic thresholds (balanced)
_SEM_HIGH = 0.92      # strong, auto-OK
_SEM_MID = 0.87       # good, confirm
_SEM_MARGIN = 0.05    # min lead over #2 for confident OK

# ---------------------------------------------------------------------
# Data structures
@dataclass
class Candidate:
    dataset_name: str
    canonical_name: str
    score: float
    reasons: List[str]

@dataclass
class ResolveResult:
    status: str               # ok | confirm | ambiguous | not_found
    method: str
    best: Optional[Candidate]
    candidates: List[Candidate]
    ask_hint: Optional[str]

# ---------------------------------------------------------------------
# Utility helpers
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().replace(".", "").replace("_", " "))

def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]

def _initials(full_name: str) -> str:
    ts = _tokens(full_name)
    if not ts:
        return ""
    given = ts[:-1] if len(ts) >= 2 else ts
    return "".join(t[0].upper() for t in given)

def _surname(full_name: str) -> str:
    ts = _tokens(full_name)
    return ts[-1] if ts else ""

def _metaphone(word: str) -> Tuple[str, str]:
    w = (word or "").strip()
    if not w:
        return ("", "")
    try:
        from metaphone import doublemetaphone
        return doublemetaphone(w)
    except ImportError:
        if hasattr(jellyfish, "metaphone"):
            return (jellyfish.metaphone(w), "")
        elif hasattr(jellyfish, "soundex"):
            return (jellyfish.soundex(w), "")
        else:
            return (w.lower(), "")

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# ---------------------------------------------------------------------
# Semantic reranker
_embed_model = SentenceTransformer("thenlper/gte-small")

def semantic_rerank(query: str, candidates: list[str], top_k: int = 5):
    """Return candidates ranked by semantic similarity to query text."""
    if not candidates:
        return []
    query_emb = _embed_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cand_emb = _embed_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, cand_emb)[0].tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ---------------------------------------------------------------------
# Resolver
class SmartNameResolver:
    """Deterministic resolver with semantic confirmation & rescue."""

    def __init__(self, csv_path: Path = CSV_PATH, parquet_path: Path = PARQUET_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["canonical_name"] = self.df["canonical_name"].astype(str).str.strip()
        self.df["dataset_name"]   = self.df["dataset_name"].astype(str).str.strip()
        self.df["norm_canon"]     = self.df["canonical_name"].map(_norm)
        self.df["canon_tokens"]   = self.df["canonical_name"].map(_tokens)
        self.df["canon_surname"]  = self.df["canonical_name"].map(_surname)
        self.df["canon_init"]     = self.df["canonical_name"].map(_initials)
        self.df["canon_surn_mp"]  = self.df["canon_surname"].map(lambda s: _metaphone(s)[0])

        # optional frequency (for tie-breaks)
        self.freq: Dict[str, int] = {}
        try:
            pdf = pd.read_parquet(parquet_path)
            counts = pd.concat([
                pdf["batsman"].dropna().astype(str),
                pdf["bowler"].dropna().astype(str),
            ]).value_counts()
            short2canon = dict(zip(self.df["dataset_name"], self.df["canonical_name"]))
            for short, c in counts.items():
                can = short2canon.get(short)
                if can:
                    self.freq[can] = int(c)
        except Exception:
            self.freq = {}

        self._canon_names = self.df["canonical_name"].tolist()
        self._exact_map = {n: ds for n, ds in zip(self.df["norm_canon"], self.df["dataset_name"])}

    # ----------------------------------------------------------
    def _score(self, query: str, row: pd.Series) -> Tuple[float, List[str]]:
        q_norm = _norm(query)
        q_tokens = _tokens(query)
        q_surname = q_tokens[-1] if q_tokens else ""
        q_init = _initials(query)
        reasons, score = [], 0.0

        if q_norm == row["norm_canon"]:
            return 100.0, ["exact"]

        if q_norm in row["norm_canon"]:
            score += 20; reasons.append("substring")

        if q_surname and q_init and row["canon_surname"] == q_surname:
            score += 25; reasons.append("surname_eq")
            if row["canon_init"].startswith(q_init.upper()):
                score += 25; reasons.append("init_prefix")

        jac = _jaccard(q_tokens, row["canon_tokens"])
        score += 20 * jac; reasons.append(f"jaccard={jac:.2f}")

        if q_tokens and any(row["norm_canon"].startswith(t) for t in q_tokens):
            score += 5; reasons.append("startswith")

        qmp = _metaphone(q_surname)[0] if q_surname else ""
        if qmp and qmp == row["canon_surn_mp"]:
            score += 12; reasons.append("surname_phonetic")

        fr = fuzz.WRatio(q_norm, row["norm_canon"])
        score += fr * 0.4; reasons.append(f"fuzzy={fr}")

        freq = self.freq.get(row["canonical_name"], 0)
        if freq:
            bump = min(10.0, 2.0 + (freq ** 0.5) / 50.0)
            score += bump; reasons.append(f"freq+{bump:.1f}")

        return score, reasons

    # ----------------------------------------------------------
    def resolve(self, query: str) -> ResolveResult:
        q = query or ""
        q_norm, q_tokens = _norm(q), _tokens(q)
        if not q_tokens:
            return ResolveResult("not_found", "rule", None, [], "Please provide a player name.")

        # --- 1️⃣ Exact match ---
        exact_ds = self._exact_map.get(q_norm)
        if exact_ds:
            row = self.df.loc[self.df["dataset_name"] == exact_ds].iloc[0]
            cand = Candidate(exact_ds, row["canonical_name"], 100.0, ["exact"])
            return ResolveResult("ok", "exact", cand, [cand], None)

        # --- 2️⃣ Single-token queries ---
        if len(q_tokens) == 1:
            token = q_tokens[0]
            mask = self.df["canon_tokens"].apply(lambda ts: token in ts or any(t.startswith(token) for t in ts))
            matches = self.df[mask]

            if len(matches) == 1:
                row = matches.iloc[0]
                sc, rs = self._score(query, row)
                best = Candidate(row["dataset_name"], row["canonical_name"], sc, rs)
                return ResolveResult("ok", "token", best, [best], None)

            if len(matches) > 1:
                ranked = self._rank(query, matches)
                cand_names = [c.canonical_name for c in ranked[:10]]
                sem = semantic_rerank(query, cand_names)
                print(f"[Semantic disambiguation #2] {query=} → {sem[:5]}")
                best_name, best_sim = sem[0]
                alt_names = [n for n, _ in sem[1:4]]
                ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
                best = Candidate(ds_name, best_name, best_sim * 100, ["semantic_disambig"])

                # Single token: always confirm, never auto-OK unless difference huge
                if (best_sim >= _SEM_HIGH) and (best_sim - sem[1][1] > 0.10):
                    return ResolveResult("ok", "semantic", best, [best], None)
                hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                return ResolveResult("confirm", "semantic", best, [best], hint)

            # semantic rescue for typos like "Rohitt"
            sem = semantic_rerank(query, self.df["canonical_name"].tolist())
            if sem:
                best_name, best_sim = sem[0]
                alt_names = [n for n, _ in sem[1:4]]
                if best_sim > _SEM_MID:
                    ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
                    best = Candidate(ds_name, best_name, best_sim * 100, ["semantic_rescue"])
                    if (best_sim >= _SEM_HIGH) and (best_sim - sem[1][1] > _SEM_MARGIN):
                        return ResolveResult("ok", "semantic", best, [best], None)
                    hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                    return ResolveResult("confirm", "semantic", best, [best], hint)

        # --- 3️⃣ Initials + surname ---
        if len(q_tokens) >= 2:
            qs = q_tokens[-1]
            mask_surn = (self.df["canon_surname"] == qs) | (self.df["canon_surn_mp"] == _metaphone(qs)[0])
            cand = self.df[mask_surn]
            if len(cand) == 1:
                row = cand.iloc[0]
                sc, rs = self._score(query, row)
                best = Candidate(row["dataset_name"], row["canonical_name"], sc, rs)
                return ResolveResult("ok", "initials", best, [best], None)

            if len(cand) > 1:
                ranked = self._rank(query, cand)
                cand_names = [c.canonical_name for c in ranked[:10]]
                sem = semantic_rerank(query, cand_names)
                print(f"[Semantic rerank #3] {query=} → {sem[:5]}")
                best_name, best_sim = sem[0]
                alt_names = [n for n, _ in sem[1:4]]
                ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
                best = Candidate(ds_name, best_name, best_sim * 100, ["semantic"])
                if (best_sim >= _SEM_HIGH) and (best_sim - sem[1][1] > _SEM_MARGIN):
                    return ResolveResult("ok", "semantic", best, [best], None)
                if best_sim >= _SEM_MID:
                    hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                    return ResolveResult("confirm", "semantic", best, [best], hint)
                hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                return ResolveResult("ambiguous", "semantic", None, ranked[:10], hint)

        # --- 4️⃣ Token overlap ---
        tok_mask = self.df["canon_tokens"].apply(lambda ts: bool(set(q_tokens) & set(ts)))
        tok_cand = self.df[tok_mask]
        if not tok_cand.empty:
            ranked = self._rank(query, tok_cand)
            cand_names = [c.canonical_name for c in ranked[:10]]
            sem = semantic_rerank(query, cand_names)
            print(f"[Semantic rerank #4] {query=} → {sem[:5]}")
            best_name, best_sim = sem[0]
            alt_names = [n for n, _ in sem[1:4]]
            ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
            best = Candidate(ds_name, best_name, best_sim * 100, ["semantic"])
            if (best_sim >= _SEM_HIGH) and (best_sim - sem[1][1] > _SEM_MARGIN):
                return ResolveResult("ok", "semantic", best, [best], None)
            if best_sim >= _SEM_MID:
                hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                return ResolveResult("confirm", "semantic", best, [best], hint)
            hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
            return ResolveResult("ambiguous", "semantic", None, ranked[:10], hint)

        # --- 5️⃣ Fuzzy fallback ---
        try:
            best_name, fr, _ = process.extractOne(q_norm, self._canon_names, scorer=fuzz.WRatio)
        except Exception:
            best_name, fr = None, 0
        if best_name and fr >= 75:
            row = self.df.loc[self.df["canonical_name"] == best_name].iloc[0]
            sc, rs = self._score(query, row)
            best = Candidate(row["dataset_name"], row["canonical_name"], sc, rs)
            return ResolveResult("ok", "fuzzy", best, [best], None)

        # --- 6️⃣ Global semantic rescue ---
        sem = semantic_rerank(query, self.df["canonical_name"].tolist())
        if sem:
            best_name, best_sim = sem[0]
            alt_names = [n for n, _ in sem[1:4]]
            if best_sim > _SEM_MID:
                ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
                best = Candidate(ds_name, best_name, best_sim * 100, ["semantic_global"])
                if (best_sim >= _SEM_HIGH) and (best_sim - sem[1][1] > _SEM_MARGIN):
                    return ResolveResult("ok", "semantic", best, [best], None)
                hint = f"Did you mean **{best_name}**? If not, maybe: {', '.join(alt_names)}."
                return ResolveResult("confirm", "semantic", best, [best], hint)

        return ResolveResult("not_found", "fuzzy", None, [], "No player matched. Please provide full name or team.")

    # ----------------------------------------------------------
    def _rank(self, query: str, frame: pd.DataFrame) -> List[Candidate]:
        scored = []
        for _, row in frame.iterrows():
            sc, rs = self._score(query, row)
            scored.append(Candidate(row["dataset_name"], row["canonical_name"], sc, rs))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

# ---------------------------------------------------------------------
# Module-level instance
_resolver = SmartNameResolver()

def resolve_player_smart(query: str):
    """Compact wrapper for CLI/agent use."""
    res = _resolver.resolve(query)
    if res.status == "ok":
        return (res.best.dataset_name, "ok")
    if res.status == "confirm":
        return (res.best.canonical_name, "confirm", res.ask_hint)
    if res.status == "ambiguous":
        return ([c.canonical_name for c in res.candidates], "ambiguous", res.ask_hint)
    return (None, "not_found", res.ask_hint)