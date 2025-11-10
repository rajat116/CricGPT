# 🏏 CricGPT — Intelligent IPL Analytics & Chat Agent

**CricGPT** is a modular cricket analytics framework focused on the **Indian Premier League (IPL)**.
It combines structured IPL statistics, machine-learning performance models, and (optional) LLM-powered natural-language reasoning.

---

### ⚡ Key Capabilities

* Parse and analyze **IPL datasets** from Cricsheet
* Predict **player performance** using trained ML models
* Answer **natural-language IPL questions** like
  *“How did Mumbai Indians perform at Wankhede in 2021?”*
* Maintain **contextual memory** across queries
* Use **LLM fallback** only when structured IPL data is missing (`--fallback`)

> 🧠 CricGPT focuses exclusively on IPL data.
> The optional fallback LLM (OpenAI / Gemini) can answer general questions,
> but it’s outside the structured IPL scope.

---

## 🧩 Step 0 — Setup & Data Verification

✅ Environment and dependencies installed
✅ Cricsheet IPL YAML datasets downloaded
✅ Verified schema and match integrity
✅ Confirmed canonical `deliveries` schema

**Output**

```
data/processed/ipl_deliveries.parquet
```

---

## 📂 Step 1 — Data Parser & Cache

Converts all IPL YAML scorecards into one clean Parquet file.

**Features**

* Merges every season automatically
* Columns: `match_id`, `season`, `teams`, `batsman`, `bowler`, `runs`, `extras`, `wickets`, etc.
* Uses `@lru_cache` for fast repeated reads

```bash
python cricket_tools/parse_data.py
```

Result → `data/processed/ipl_deliveries.parquet`

---

## 🧠 Step 2 — Smart Name Resolver (`smart_names.py`)

Understands **typos**, **initials**, and **ambiguous** names.

**Techniques**

* Phonetic (DMetaphone)
* Fuzzy matching (Levenshtein)
* Semantic embeddings (Sentence-Transformer)
* Confidence-weighted auto-confirmation

```python
from cricket_tools.smart_names import resolve_player_smart
resolve_player_smart("Rohitt Sharm")
```

Returns

```text
('Rohit Gurunath Sharma', 'confirm',
 'Did you mean **Rohit Gurunath Sharma**? If not maybe: Mohit Mahipal Sharma, Rahul Sharma.')
```

Statuses: `ok | confirm | ambiguous | not_found`

---

## 📊 Step 3 — Stats & Filter Layer (`stats.py`, `filters.py`)

All analytical logic is isolated here.

| File         | Responsibility                                                      |
| ------------ | ------------------------------------------------------------------- |
| `stats.py`   | Computes player/team aggregates (runs, SR, avg, wickets, economy …) |
| `filters.py` | Applies filters (`season`, `team`, `venue`, `city`, `start`, `end`) |

Functions:

```python
get_player_stats()
get_bowler_stats()
get_team_stats()
get_top_players()
```

Includes canonical normalization (e.g. “Banglore” → “Bengaluru”, “Chepuk” → “M. A. Chidambaram Stadium”).

---

## 🤖 Step 4 — ML Performance Prediction

Machine-learning layer predicts **expected performance** (runs / wickets per match).

### Pipeline

1. `ml_build.py` → build batting features
2. `ml_build_bowl.py` → build bowling features
3. `ml_model.py` → train RandomForestRegressors

   * `performance_model_bat.pkl`
   * `performance_model_bowl.pkl`

Run:

```bash
python cricket_tools/ml_build.py
python cricket_tools/ml_build_bowl.py
python cricket_tools/ml_model.py
```

Each model saves a `*_meta.json` (features + metrics + timestamp).

### Unified Prediction

`predict_future_performance()` in `ml_model.py`:

* Automatically loads both models
* Handles batters, bowlers, or all-rounders
* Always returns structured JSON

---

## 🧭 Step 5 — Unified Agent Interface (`agent.py`)

Introduces the **natural-language agent** that understands queries and routes them to the right tool.

### Backends

| Backend          | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `mock`           | Offline pattern-based planner                             |
| `semantic`       | Embedding-based planner (Sentence-Transformer)            |
| `openai` / `llm` | LLM planner (OpenAI / Gemini / Ollama via unified config) |

Default → `openai` if `OPENAI_API_KEY` exists, else `semantic`, else `mock`.

### Example Queries

```bash
python -m cricket_tools.agent "Forecast Bumrah's form"
python -m cricket_tools.agent "Show Kohli stats 2021" --backend semantic
python -m cricket_tools.agent "Top wicket takers in Bengaluru" --backend llm
```

Supports dual-role players (bat + bowl) and prediction integration.

---

## 🧩 Step 6 — Context-Aware Knowledge & Multi-Player Reasoning

Extends the agent beyond single-player queries to team- and comparison-level analytics.

### Goals

* Understand queries like
  *“Who scored most runs for Mumbai in 2023?”*
  *“Compare Rohit and Virat this season.”*
* Add team-level aggregations and multi-filter logic
* Handle multiple player names and venues
* Enable caching for faster repeated queries

### Implementation Summary

| Area             | Change                                                       |
| ---------------- | ------------------------------------------------------------ |
| `filters.py`     | Added season/venue/city/team filters + canonical mapping     |
| `stats.py`       | Added `get_team_stats`, `compare_players`, `get_top_players` |
| `core.py`        | Unified routing for team and comparison intents              |
| `agent.py`       | Multi-entity extraction and intent detection                 |
| `smart_names.py` | Continued handling ambiguous players                         |

### ✅ Verified Capabilities

| Feature             | Example Query                                          |
| ------------------- | ------------------------------------------------------ |
| Player Stats        | “Show Rohit Sharma batting stats in 2023”              |
| Bowler Stats        | “Bowling stats for Bumrah last year”                   |
| Team Performance    | “How did Chennai Super Kings perform in 2020?”         |
| Player Comparison   | “Compare Rohit and Virat in Chepuk 2023”               |
| Top Players         | “Top 5 run scorers in Mumbai 2021”                     |
| Venue Normalization | “Banglore → Bengaluru”, “Chepuk → Chidambaram Stadium” |
| Ambiguity Handling  | “Virat” → clarification prompt                         |
| Multi-Filter Logic  | “RCB in Bengaluru 2019”                                |

### 🔬 Test All Features

```bash
bash tests/run_agent_tests.sh llm
```

Generates logs in `tests/test_results_<timestamp>.log`.

---

## 🧠 Step 7 — Conversational Memory & LLM Fallback

This step gives **context carry-over** between user queries and adds **LLM fallback** for missing data.

### 🧩 Files Added / Updated

| File              | Purpose                                                               |
| ----------------- | --------------------------------------------------------------------- |
| `memory.py`       | Implements session-level context cache (`.cache/session_memory.json`) |
| `agent.py`        | Merges memory before each query + updates after tool execution        |
| `config.py`       | Unified LLM provider (OpenAI / Gemini / Ollama) configuration         |
| `llm_fallback.py` | Handles fallback answers when structured data is empty                |

### 💾 Memory Features

* Stores entities like `team`, `venue`, `city`, `player`, etc.
* 2-hour TTL (automatically expires)
* Merge new inputs with previous context
* Clear with `--clear-memory`

Example:

```bash
python -m cricket_tools.agent "Show MI performance at Wankhede" --backend llm
python -m cricket_tools.agent "Show at Chepuk" --backend llm
```

➡ Second query remembers `team = MI`.

### 🧩 LLM Fallback

If structured tools return “no data”, use:

```bash
python -m cricket_tools.agent "Who was top scorer 2023" --backend llm --fallback
```

➡ Fallback LLM (OpenAI / Gemini) answers directly
while structured queries remain unaffected.

---

## 🚀 Upcoming Roadmap

### **Step 8 — Visualization & Interactive Dashboards**

**Goal:** Let users *see* the stats.
**Planned Features**

* Integrate **Matplotlib / Plotly** for:

  * Player form over time
  * Head-to-head bar charts
  * Team win ratios by venue
* Auto-export plots to `outputs/plots/`
* CLI option:

  ```bash
  python -m cricket_tools.agent "Compare Rohit and Virat 2023" --plot
  ```
* Optional **Streamlit UI** for interactive queries

---

### **Step 9 — Knowledge Augmentation & Data Expansion**

**Goal:** Enrich datasets and context.

* Integrate match summaries & player metadata (age, team history)
* Hybrid queries like:
  *“Who was best finisher in Chennai 2023 with SR > 140?”*
* Metadata embeddings for smarter entity linking
* Optional support for ODI / T20I datasets with auto selection

---

### **Step 10 — Deployment & Showcase**

**Goal:** Make CricGPT production-ready.

* Package agent as CLI + FastAPI microservice (`serve_agent.py`)
* Provide Dockerfile + `requirements.txt` for reproducible setup
* Host demo via Streamlit Cloud or Codespaces
* Add evaluation notebook (`notebooks/evaluation.ipynb`)
* Finalize README with badges & project banner

---

### ✅ Optional Future Extensions

* Multi-player (>2) comparison table & radar charts
* Natural-language explanations via mini-LLM (gpt-4o-mini local)
* Persistent user profiles for custom recommendations
* Integration with CricAPI / live feeds for real-time stats

---

**🧩 Test Everything So Far**

```bash
bash tests/run_agent_tests.sh llm
python -m tests.test_memory_chain
python -m tests.test_memory_quick
python -m cricket_tools.llm_fallback
```

---

**© 2025 Rajat Gupta**  ·  CricGPT Project ·  All Rights Reserved
Data Scientist & Researcher · University of Pittsburgh / CERN