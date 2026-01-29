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
#CSV_PATH = Path("data/processed/player_names.csv")
#PARQUET_PATH = Path("data/processed/ipl_deliveries.parquet")

# ---------------------------------------------------------------------
# Paths (resolved relative to the package → works on Streamlit & locally)
BASE_DIR = Path(__file__).resolve().parent      # cricket_tools/
ROOT_DIR = BASE_DIR.parent                      # project root

CSV_PATH = ROOT_DIR / "data" / "processed" / "player_names.csv"
PARQUET_PATH = ROOT_DIR / "data" / "processed" / "ipl_deliveries.parquet"

# ---------------------------------------------------------------------
# Global semantic thresholds (balanced)
_SEM_HIGH = 0.92      # strong, auto-OK
_SEM_MID = 0.87       # good, confirm
_SEM_MARGIN = 0.05    # min lead over #2 for confident OK
_FUZZY_MARGIN = 10.0  # min lead for fuzzy score

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
# Utility helpers (no changes)
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
# Semantic reranker (no changes)
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
    # _score method (no changes)
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
    # _rank method (no changes)
    def _rank(self, query: str, frame: pd.DataFrame) -> List[Candidate]:
        scored = []
        for _, row in frame.iterrows():
            sc, rs = self._score(query, row)
            scored.append(Candidate(row["dataset_name"], row["canonical_name"], sc, rs))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored
    
    # ----------------------------------------------------------
    # NEW: Centralized ambiguity "brain"
    # ----------------------------------------------------------
    def _process_semantic_results(
        self, 
        query: str, 
        sem_list: list[tuple[str, float]], 
        rule_candidates: list[Candidate],
        method: str = "semantic"
    ) -> ResolveResult:
        """
        Processes a list of semantic scores to check for ambiguity.
        This is the "full proof" logic.
        """
        if not sem_list:
            # Semantic rerank failed, fall back to rule-based ambiguity
            hint = "Multiple players found. Can you be more specific?"
            return ResolveResult("ambiguous", "rule", None, rule_candidates[:5], hint)

        best_name, best_sim = sem_list[0]

        # --- This is the key "full proof" logic ---
        if len(sem_list) > 1:
            second_sim = sem_list[1][1]
            sim_diff = best_sim - second_sim

            # If the top 2 (or more) are too close, it's ambiguous
            if sim_diff < _SEM_MARGIN:
                close_options = [n for n, s in sem_list[:3] if (best_sim - s) < _SEM_MARGIN]
                hint = f"Your query '{query}' is ambiguous. Did you mean: {', '.join(f'**{n}**' for n in close_options)}?"
                
                ambig_candidates = []
                for name in close_options:
                    ds = self.df.loc[self.df["canonical_name"] == name, "dataset_name"].values[0]
                    score = next(s for n, s in sem_list if n == name) * 100
                    ambig_candidates.append(Candidate(ds, name, score, [f"{method}_ambiguous"]))
                
                # Return AMBIGUOUS, not confirm
                return ResolveResult("ambiguous", method, None, ambig_candidates, hint)
        # --- End of key logic ---

        # If we're here, the #1 result is a clear winner
        # We still return "confirm" because it's not an exact match.
        ds_name = self.df.loc[self.df["canonical_name"] == best_name, "dataset_name"].values[0]
        best = Candidate(ds_name, best_name, best_sim * 100, [method])
        alt_names = [n for n, _ in sem_list[1:4] if n != best_name]
        #hint = f"Did you mean **{best_name}**?{f' Or maybe: {", ".join(alt_names)}' if alt_names else ''}."
        alt_part = f" Or maybe: {', '.join(alt_names)}" if alt_names else ""
        hint = f"Did you mean **{best_name}**?{alt_part}."

        return ResolveResult("confirm", method, best, [best], hint)

    # ----------------------------------------------------------
    # MODIFIED: Main resolve method
    # ----------------------------------------------------------
    def resolve(self, query: str) -> ResolveResult:
        q = query or ""
        q_norm, q_tokens = _norm(q), _tokens(q)
        if not q_tokens:
            return ResolveResult("not_found", "rule", None, [], "Please provide a player name.")

        # --- 1️⃣ Exact match ---
        # This is the *only* case that returns "ok"
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
                hint = f"Did you mean **{best.canonical_name}**?"
                return ResolveResult("confirm", "token", best, [best], hint)

            if len(matches) > 1:
                ranked = self._rank(query, matches)
                cand_names = [c.canonical_name for c in ranked[:10]]
                sem = semantic_rerank(query, cand_names)
                
                # MODIFIED: Call the new centralized "brain"
                sem_result = self._process_semantic_results(query, sem, ranked)
                
                # Special case for "Sharma" (too many rule-based matches)
                # If it wasn't already ambiguous, make it ambiguous due to count
                if len(matches) > 5 and sem_result.status != "ambiguous":
                    hint = f"Found {len(matches)} players matching '{query}'. Can you provide a first name or initials? (e.g., {', '.join([c.canonical_name for c in ranked[:3]])}, ...)"
                    return ResolveResult("ambiguous", "rule_count", sem_result.best, ranked[:5], hint)
                    
                return sem_result # This will be the "ambiguous" or "confirm" result

            # semantic rescue for typos like "Rohitt"
            # MODIFIED: Use the ambiguity-aware logic here too
            sem = semantic_rerank(query, self._canon_names)
            if sem and sem[0][1] > _SEM_MID: # Check if best score is decent
                # Pass an empty list for rule_candidates
                return self._process_semantic_results(query, sem, [], method="semantic_rescue")

        # --- 3️⃣ Initials + surname ---
        if len(q_tokens) >= 2:
            qs = q_tokens[-1]
            mask_surn = (self.df["canon_surname"] == qs) | (self.df["canon_surn_mp"] == _metaphone(qs)[0])
            cand = self.df[mask_surn]
            
            if len(cand) == 1:
                row = cand.iloc[0]
                sc, rs = self._score(query, row)
                best = Candidate(row["dataset_name"], row["canonical_name"], sc, rs)
                hint = f"Did you mean **{best.canonical_name}**?"
                return ResolveResult("confirm", "initials", best, [best], hint)

            if len(cand) > 1:
                ranked = self._rank(query, cand)
                # ... (surname-priority filter logic is good) ...
                if len(q_tokens) >= 2:
                    surname = q_tokens[-1]
                    firstname = q_tokens[0]
                    same_surn = [c for c in ranked if surname.lower() in _norm(c.canonical_name)]
                    if same_surn:
                        same_surn_first = [c for c in same_surn if firstname.lower() in _norm(c.canonical_name)]
                        if same_surn_first:
                            ranked = same_surn_first
                        else:
                            ranked = same_surn
                
                cand_names = [c.canonical_name for c in ranked[:10]]
                sem = semantic_rerank(query, cand_names)
                print(f"[Semantic rerank #3] {query=} → {sem[:5]}")
                
                # MODIFIED: Call the new centralized "brain"
                return self._process_semantic_results(query, sem, ranked)

        # --- 4️⃣ Token overlap ---
        tok_mask = self.df["canon_tokens"].apply(lambda ts: bool(set(q_tokens) & set(ts)))
        tok_cand = self.df[tok_mask]
        if not tok_cand.empty:
            ranked = self._rank(query, tok_cand)
            # ... (surname-priority filter logic is good) ...
            if len(q_tokens) >= 2:
                surname = q_tokens[-1]
                firstname = q_tokens[0]
                same_surn = [c for c in ranked if surname.lower() in _norm(c.canonical_name)]
                if same_surn:
                    same_surn_first = [c for c in same_surn if firstname.lower() in _norm(c.canonical_name)]
                    if same_surn_first:
                        ranked = same_surn_first
                    else:
                        ranked = same_surn
            
            cand_names = [c.canonical_name for c in ranked[:10]]
            sem = semantic_rerank(query, cand_names)
            print(f"[Semantic rerank #4] {query=} → {sem[:5]}")
            
            # MODIFIED: Call the new centralized "brain"
            return self._process_semantic_results(query, sem, ranked)

        # --- 5️⃣ Fuzzy fallback ---
        # MODIFIED: Made this step ambiguity-aware
        try:
            # Get top 2 candidates
            fuzzy_matches = process.extract(q_norm, self._canon_names, scorer=fuzz.WRatio, limit=2)
        except Exception:
            fuzzy_matches = []
            
        if fuzzy_matches:
            best_name, fr_best, _ = fuzzy_matches[0]
            if fr_best >= 75: # Only proceed if the best match is decent
                
                # --- AMBIGUITY CHECK ---
                if len(fuzzy_matches) > 1:
                    second_name, fr_second, _ = fuzzy_matches[1]
                    # If fuzzy scores are very close
                    if (fr_best - fr_second) < _FUZZY_MARGIN:
                        close_options = [best_name, second_name]
                        hint = f"Your query '{query}' is ambiguous (fuzzy match). Did you mean: {', '.join(f'**{n}**' for n in close_options)}?"
                        ambig_candidates = []
                        for name in close_options:
                            ds = self.df.loc[self.df["canonical_name"] == name, "dataset_name"].values[0]
                            score = next(s for n, s, _ in fuzzy_matches if n == name)
                            ambig_candidates.append(Candidate(ds, name, score, ["fuzzy_ambiguous"]))
                        return ResolveResult("ambiguous", "fuzzy", None, ambig_candidates, hint)
                # --- END AMBIGUITY CHECK ---
                
                # If we're here, it's a clear fuzzy winner
                row = self.df.loc[self.df["canonical_name"] == best_name].iloc[0]
                best = Candidate(row["dataset_name"], row["canonical_name"], fr_best, ["fuzzy"])
                hint = f"Did you mean **{best.canonical_name}**? (Matched with {fr_best:.0f}% similarity)"
                return ResolveResult("confirm", "fuzzy", best, [best], hint)

        # --- 6️⃣ Global semantic rescue ---
        # MODIFIED: Use the ambiguity-aware logic here too
        sem = semantic_rerank(query, self._canon_names)
        if sem and sem[0][1] > _SEM_MID: # Check if best score is decent
            return self._process_semantic_results(query, sem, [], method="semantic_global")

        return ResolveResult("not_found", "fuzzy", None, [], "No player matched. Please provide full name or team.")

# ---------------- Module-level instance + wrapper ----------------
# This part remains the same
_resolver = SmartNameResolver()

def resolve_player_smart(query: str):
    """Return (dataset_name, canonical_name, status, hint)"""
    res = _resolver.resolve(query)

    if res.status == "ok" and res.best:
        return (res.best.dataset_name, res.best.canonical_name, "ok", None)

    if res.status == "confirm" and res.best:
        return (res.best.dataset_name, res.best.canonical_name, "confirm", res.ask_hint)

    if res.status == "ambiguous":
        # Return the *list* of options in the canonical_name slot
        options = [c.canonical_name for c in res.candidates] if res.candidates else []
        if res.best: # Should be None, but just in case
            options = [res.best.canonical_name] + [o for o in options if o != res.best.canonical_name]
        return (None, options, "ambiguous", res.ask_hint)

    return (None, None, "not_found", res.ask_hint)