import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import jellyfish
import numpy as np

# --- Load your player list
df = pd.read_csv("data/processed/player_names.csv")
names = df["canonical_name"].tolist()

# --- Model for embedding tests
model = SentenceTransformer("thenlper/gte-small")
embs = model.encode(names, normalize_embeddings=True)

# --- Helper 1: RapidFuzz
def match_fuzzy(query, top_k=5):
    scores = process.extract(query, names, scorer=fuzz.WRatio, limit=top_k)
    return [(n, round(s, 3)) for n, s, _ in scores]

# --- Helper 2: Embedding cosine
def match_embed(query, top_k=5):
    q_emb = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, embs)[0].cpu().numpy()
    idx = np.argsort(scores)[::-1][:top_k]
    return [(names[i], round(float(scores[i]), 3)) for i in idx]

# --- Helper 3: Hybrid (weighted)
def match_hybrid(query, top_k=5, w_embed=0.6):
    q_emb = model.encode(query, normalize_embeddings=True)
    scores_emb = util.cos_sim(q_emb, embs)[0].cpu().numpy()
    scores_fuz = np.array([fuzz.WRatio(query, n)/100 for n in names])
    combined = w_embed * scores_emb + (1 - w_embed) * scores_fuz
    idx = np.argsort(combined)[::-1][:top_k]
    return [(names[i], round(float(combined[i]), 3)) for i in idx]

# --- Helper 4: Phonetic
def match_phonetic(query, top_k=5):
    q_code = jellyfish.metaphone(query.split()[-1])  # use surname sound
    sims = []
    for n in names:
        code = jellyfish.metaphone(n.split()[-1])
        sims.append(1.0 if code == q_code else 0.0)
    idx = np.argsort(sims)[::-1][:top_k]
    return [(names[i], sims[i]) for i in idx if sims[i] > 0]

# --- Test queries
queries = ["Rohittt Sharma", "Rashid Khan", "Ashwin", "Hardik Pandya", "Rohitt Sharm", "Abhishek", "Dhoni", "Rohit", "Rashidkhan", "Ashwin","Kuldeep"]

for q in queries:
    print(f"\n🧩 Query: {q}")
    print("Fuzzy:", match_fuzzy(q))
    print("Embed:", match_embed(q))
    print("Hybrid:", match_hybrid(q))
    print("Phonetic:", match_phonetic(q))