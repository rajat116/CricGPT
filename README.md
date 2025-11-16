# 🏏 OverPrompt — LLM-Powered IPL Analytics & Chat Agent

**OverPrompt** is an IPL-focused cricket assistant that combines:

* **Ball-by-ball IPL data from Cricsheet** (stored as a clean Parquet dataset)
* A **deterministic analytics engine** (Pandas / Python, no guessed numbers)
* A thin **LLM layer** (OpenAI / Gemini / etc.) for natural-language understanding and nice explanations
* Optional **visualizations** (momentum worms, phase dominance, top scorers, H2H, etc.)

🔗 **Live app:** [https://overprompt.streamlit.app/](https://overprompt.streamlit.app/)

OverPrompt is designed as a **data-first cricket brain**: every stat and chart is computed from structured IPL data; the LLM is only used to understand your question and phrase answers.

---

## 💡 What OverPrompt Is (in plain English)

Think of OverPrompt as:

> “A serious IPL stats engine with a chat interface — not just another prompt around ChatGPT.”

You can ask questions like:

* *“Show Rohit Sharma’s batting stats in IPL 2020.”*
* *“Compare Virat Kohli and Rohit Sharma in 2016.”*
* *“Top 5 run scorers at Wankhede in 2019.”*
* *“How did CSK perform in 2018?”*
* *“Show the momentum worm for MI vs CSK 2019.”*

OverPrompt will:

1. Parse your question in natural language.
2. Resolve player/team/venue names robustly (even with typos).
3. Run **structured Pandas queries** on ball-by-ball IPL data.
4. Optionally generate a **plot**.
5. Use the LLM only to turn that structured result into a clean explanation.

---

## 🆚 How OverPrompt is Different

### 1. Grounded in **ball-by-ball IPL data**, not vague web knowledge

Many “CricGPT”-style projects and generic ChatGPT/Gemini agents:

* Use **summary tables** or scraped scorecards.
* Or rely only on the LLM’s internal knowledge (which can be wrong / outdated).

OverPrompt instead:

* Ingests full **Cricsheet IPL YAML**,
* Normalises it into a **single Parquet dataset** (`ipl_deliveries.parquet`),
* Works at the **ball, over, innings, and match** level with precise filters.

If OverPrompt shows a number, it comes from that dataset — not from internet guessing.

---

### 2. **Data-first design** – LLM is a helper, not the boss

OverPrompt is deliberately built as:

> **Deterministic stats engine → then LLM for language.**

* All stats, aggregations, rankings, and filters are computed in **Python/Pandas**.
* The LLM:

  * Interprets messy questions,
  * Decides which tool / stat function to call,
  * And formats the final explanation.

This means:

* No “hallucinated” scores or fake matches.
* Easy to reproduce and test — you can inspect the underlying data anytime.

---

### 3. Robust **entity & name resolver** (players, teams, venues, cities)

Instead of hoping the LLM spells names right, OverPrompt uses a dedicated resolver that combines:

* **Phonetic matching** (e.g. DMetaphone via `jellyfish`),
* **Fuzzy string matching** (via `rapidfuzz`),
* **Semantic embeddings** (via `sentence-transformers`).

It can handle:

* Typos: `“Rohitt Sharm”` → **Rohit Sharma**
* Ambiguity: `“Rahul”` → asks whether you mean KL Rahul / Rahul Chahar / Rahul Sharma
* Canonicalization:

  * `"Banglore"` → **Bengaluru**
  * `"Chepuk"` → **M. A. Chidambaram Stadium**
  * `"RCB"` → **Royal Challengers Bengaluru**

This entity layer is **explicit and explainable**, not hidden inside the LLM.

---

### 4. IPL-aware analytics & plots, not just text

Because OverPrompt has ball-by-ball data, it can generate cricket-specific analytics and charts, such as:

* **Player form cards** (runs, average, strike rate)
* **Head-to-head comparisons** (Rohit vs Kohli, etc.)
* **Top-N batters / bowlers** with filters (season, venue, city)
* **Momentum worms** (over-by-over cumulative runs)
* **Phase dominance** (powerplay / middle / death RPO)
* **Bowling threat maps** (wickets by phase and bowler)
* **Partnership graphs**
* **Pressure / required run rate curves**
* **Win-probability series** (ball-by-ball)

These are not generic “LLM charts”; they’re computed from your IPL dataset with standard cricket logic.

---

### 5. Safer than generic ChatGPT / Gemini for stats

ChatGPT or Gemini alone:

* Might answer IPL questions using **training data only**,
* Can hallucinate, especially on niche or very specific match stats.

OverPrompt:

* Always tries structured IPL tools first.
* If data doesn’t exist (e.g. non-IPL query), it can:

  * Explain that the data isn’t in the IPL dataset, or
  * Use a **clearly marked fallback** mode.

The emphasis is on **honest, data-faithful answers**, not storytelling.

---

## 🖥️ Try OverPrompt Online

You don’t need to install anything to see it in action.

1. Open: **[https://overprompt.streamlit.app/](https://overprompt.streamlit.app/)**
2. Wait for the app to load.
3. You’ll see:

   * A chat-style input box,
   * A main results area,
   * A sidebar with settings (e.g. auto-plot toggle, backend selection if exposed).

### Example questions to try

Copy-paste any of these into the app:

* `Show Rohit Sharma batting stats in IPL 2020`
* `Compare Virat Kohli and Rohit Sharma in 2016`
* `Top 5 run scorers at Wankhede in 2019`
* `How did Chennai Super Kings perform in 2018?`
* `Momentum worm for MI vs CSK 2019`
* `Phase dominance MI vs CSK 2019`
* `Partnership graph for MI vs KKR 2017`

If plotting is enabled, you’ll get **both** a written summary and a plot.

---

## 🛠️ Local Setup (Developer Quickstart)

If you want to run or modify OverPrompt locally:

```bash
# 1. Clone the repo
git clone https://github.com/rajat116/CricGPT.git
cd CricGPT   # repo name can stay; app name is OverPrompt

# 2. Create & activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### IPL data

OverPrompt expects a processed IPL dataset like:

```text
data/processed/ipl_deliveries.parquet
```

If it’s not already present, use the parsing script in `cricket_tools/` (for example):

```bash
python cricket_tools/parse_data.py
```

This script reads Cricsheet IPL YAML files (see comments/config inside the script) and builds the unified Parquet file.

### Environment / API keys

If you use the LLM features (reasoning + nice explanations), you’ll need provider settings, typically via `.env` or a settings TOML, e.g.:

```env
LLM_PROVIDER=openai        # or gemini / ollama
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=...
GEMINI_MODEL=models/gemini-2.5-flash
```

(Adjust according to how your repo is currently wired; the core logic works fine in **non-LLM / semantic** mode too.)

### Run the Streamlit app locally

```bash
streamlit run streamlit_app.py
# or whatever the Streamlit entrypoint file is named in your repo
```

Then open the link Streamlit prints (usually `http://localhost:8501`).

---

## 🧱 High-Level Architecture

At a high level, OverPrompt is structured like this:

```text
User question
   ↓
LLM / semantic planner (intent + entities)
   ↓
Cricket tools (Pandas over Parquet IPL data)
   ↓
Optional: plotting layer (matplotlib)
   ↓
LLM formatter for natural-language answer
   ↓
Streamlit UI (or CLI / API)
```

---

## 📂 Repository Structure (conceptual)

The exact filenames may differ slightly, but the core layout is:

```text
cricket_tools/
  ├── parse_data.py        # Cricsheet YAML → Parquet (ipl_deliveries.parquet)
  ├── smart_names.py       # Phonetic + fuzzy + semantic name resolver
  ├── entity_matcher.py    # Player/team/venue/city resolution, CLI utilities
  ├── filters.py           # Common filters (season, venue, city, team, date)
  ├── stats.py             # Player, team, and team-level stats
  ├── ml_model.py          # (Optional) ML models for performance prediction
  ├── agent.py             # High-level planner / orchestrator
  ├── runner.py            # Unified `run_query(...)` backend function
  ├── visuals.py           # Plotting & visualization router
  └── config/...           # LLM + provider configuration

streamlit_app.py (or app.py)
  # Streamlit UI that calls `cricket_tools.runner.run_query`

data/
  └── processed/
        └── ipl_deliveries.parquet   # Ball-by-ball IPL dataset (generated)

outputs/
  └── plots/
        └── ...png                   # Saved charts (momentum, H2H, etc.)

tests/                               # Optional tests & scripts
```

This separation makes it easy to:

* Reuse the core engine in **FastAPI**, **CLI**, or other UIs.
* Test `cricket_tools` without running Streamlit.
* Swap LLM providers without touching the stats logic.

---

## 🧰 Tech Stack & Tools

### Core Data & Analytics

- **Python 3.x**
- **Pandas**, **NumPy** – feature engineering, aggregations, match/over/ball-level stats
- **PyArrow / Parquet** – columnar IPL dataset (`ipl_deliveries.parquet`)
- **Matplotlib** – momentum worms, phase-dominance, H2H, partnership, pressure & win-probability plots

### Entity Resolution & Semantics

- **rapidfuzz** – fuzzy string matching for noisy player/team/venue names
- **jellyfish** – DMetaphone-based phonetic matching
- **sentence-transformers** – semantic embeddings for hybrid name/entity resolution

### LLM & Reasoning Layer

- **OpenAI / Gemini / Ollama** – pluggable LLM backends (configured via `LLM_PROVIDER` + API keys)
- **Custom reasoning agent** (`cricket_tools.agent`) that:
  - Parses natural-language questions
  - Detects intent (player stats, team stats, H2H, analytics, top-N, etc.)
  - Calls the right **structured IPL tools** (no hallucinated numbers)
- **LLM formatter** (`llm_formatter`) that:
  - Takes **JSON stats** and returns controlled natural-language explanations
  - Enforces “no guessing / no hallucinations” for IPL data
- **Fallback LLM layer** for out-of-scope questions (non-IPL / missing data), clearly separated from structured answers

### App & Infra

- **Streamlit** – public web app UI: https://overprompt.streamlit.app/
- **(Optional) FastAPI / `runner.py`** – unified backend entrypoint (`run_query(...)`) ready for API deployment
- **dotenv / TOML config** – provider selection, model selection, and feature flags
- **GitHub + Codespaces / local venv** – development & deployment workflow

---

## 📌 Status & Roadmap (short)

**Current status**

* ✅ Live Streamlit app running with OverPrompt branding
* ✅ Robust IPL data pipeline (ball-by-ball, Parquet)
* ✅ Entity resolver for players, teams, venues, cities
* ✅ Structured stats engine + analytics
* ✅ LLM-backed understanding + explanation
* ✅ Visualization layer for key cricket analytics

**Possible next steps**

* Add non-IPL leagues (WPL, BBL, etc.)
* Add more advanced derived metrics (clutch index, consistency score, intent vs execution, etc.)
* FastAPI backend for production-grade API deployment
* More tests & benchmarks against other cricket agents / pure LLM baselines
