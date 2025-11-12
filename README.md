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

## 🎨 Step 8 — Visualization & Interactive Dashboards

This step adds a **unified plotting layer** that turns any tool result into clean, shareable charts saved under `outputs/plots/`. It uses a **dark theme**, timestamped filenames, and a robust **auto-router** so callers don’t need to choose chart types manually.

---

### What’s included

* **Router:** `auto_plot_from_result(result, act)` → returns a PNG path under `outputs/plots/`.
* **Direct renderer:** `generate_plot_from_result(result, act, plot_type)` (for explicit control).
* **Graceful fallbacks:** If data is missing/insufficient, a readable **fallback card** is saved instead of raising errors.
* **Consistent schema:** Upstream agent attaches `plot_path` to `{act, data, text}` for CLI/API/UI display.

---

### File layout

```
cricket_tools/
  visuals.py           # all charts + router
  analytics.py         # data providers used by visuals (H2H, momentum, phases, etc.)
outputs/
  plots/               # saved charts (PNG, timestamped)
```

---

### Actions → Plots (auto-mapped)

`auto_plot_from_result(result, act)` supports these actions:

| Action (`act`)                | Plot type (auto)  | Description                                                 |
| ----------------------------- | ----------------- | ----------------------------------------------------------- |
| `get_batter_stats`            | `form`            | Compact bar card: Runs, Average, Strike Rate                |
| `compare_players`             | `h2h`             | Side-by-side bars (Average, SR) for Player A vs B           |
| `get_team_stats`              | `venue-ratio`     | Pie (Wins vs Losses) + overall win %                        |
| `get_top_players`             | `top`             | Sorted Top-N bar chart (runs or wickets)                    |
| `team_head_to_head`           | `h2h-team`        | Heat-matrix (Season × Venue) of Team-A win %                |
| `team_momentum`               | `momentum`        | Over-by-over cumulative runs (“worm”) for both teams        |
| `team_phase_dominance`        | `phase-dominance` | Powerplay/Middle/Death RPO bars per team                    |
| `team_threat_map`             | `threat-map`      | Stacked horizontal bars: wickets by phase per bowler        |
| `partnership_graph`           | `partnership`     | Top partnerships; **uses `innings_no` panels when present** |
| `pressure_series`             | `pressure`        | Current vs Required run-rate lines                          |
| `win_probability_series`      | `win-prob`        | Ball-by-ball win-probability line                           |
| `llm_fallback*` (any variant) | `fallback`        | Readable fallback card (or tiny numeric bar)                |

If `act` is unknown or the data is empty → a **fallback card** is saved automatically.

---

### Output & theming

* **Save path:** `outputs/plots/<kind>_<context>_<YYYYMMDD_HHMMSS>.png`
* **Theme:** `matplotlib` dark background, readable grids/labels, value annotations on bars.
* **Sizing:** Auto-sized figures; multi-panel partnerships when `innings_no` is available.

---

### Typical usage (from agent / API)

1. Run tool → get `{ act, data, text }`.
2. Call: `plot_path = auto_plot_from_result(result, act)`.
3. Return/emit: `{ act, data, text, plot_path }` to CLI/API/UI.
4. UI (or CLI) renders both the **text** and the **image**.

> You rarely need to choose a plot type manually. The router handles it.

---

### Implemented charts (highlights)

* **Player Form Card**: Bars for `Runs`, `Average`, `Strike Rate`.
* **H2H (Players)**: Two-bar groups for `Average` and `SR`; legend = player names.
* **Team Win Ratio**: Pie of wins/losses + title shows win %.
* **Top Players**: Sorted bars; supports `runs_batter` or `wickets`, optional `season`/`venue`/`city` qualifiers in title.
* **H2H Matrix**: Season × Venue matrix of Team-A win %.
* **Momentum Worm**: Over-by-over cumulative runs; optional season tag.
* **Phase Dominance**: Powerplay/Middle/Death RPO comparison.
* **Threat Map**: Stacked wickets by phase per bowler (horizontal bars).
* **Partnerships**: Panels per `innings_no` when present; otherwise Top-N list.
* **Pressure Series**: Current vs Required RPO across overs.
* **Win Probability**: `over*6 + ball_in_over` indexing; % line plot.
* **Fallback Card**: Always produces a readable image explaining missing/ambiguous data.

---

### Smoke-test checklist

* [ ] `compare_players` → bar chart saved to `outputs/plots/…h2h…png`
* [ ] `get_top_players` (`runs` & `wickets`) → sorted bars with correct qualifiers
* [ ] `get_team_stats` → pie with correct win % and counts
* [ ] `partnership_graph` → **separate panels per `innings_no`**
* [ ] `pressure_series` / `win_probability_series` → lines render with labeled axes
* [ ] Unknown/empty inputs → **fallback card** (no exceptions)
* [ ] Fresh repo + `pip install -r requirements.txt` → plots save without errors

---

### Troubleshooting

* **No file saved / blank plot?** Check that the upstream tool returned a proper `{ act, data }` payload.
* **Wrong chart?** Ensure `act` matches one of the mapped actions above.
* **Unreadable labels on Top-N?** Reduce `n` or widen the window; labels are rotated to avoid overlaps.
* **Ambiguous player/team?** The resolver prompts upstream. Visuals draw only once data is concrete.

---

### Notes for the upcoming web UI

* The returned `plot_path` is **ready to embed** in a chat-style interface.
* Keep response schema stable: `{ act, data, text, plot_path }`.
* (Optional later) Serve images via a static `/plots/<file>` route in FastAPI; current PNG files already suit Streamlit/CLI.


Below is a **clean, professional, production-ready Step-8 README section** that fits **seamlessly after your Step-7**.
Copy-paste directly into your repo README.

---

# 🎨 Step 8 — Visualization & Interactive Dashboards

Step 8 adds a complete **visual analytics layer** on top of the structured IPL engine.
Users can now generate **clean, informative charts** directly from natural-language queries using:

```
--plot
--plot auto
```

All plots are saved to:

```
outputs/plots/<plot_name>_<timestamp>.png
```

The visualization layer is fully modular and works headlessly — ideal for both CLI usage and later **FastAPI/Streamlit UI integration**.

---

## 📌 8.1 — Visualization Architecture

The visualization system lives entirely in:

```
cricket_tools/visuals.py
```

It contains:

| Component                     | Role                                                   |
| ----------------------------- | ------------------------------------------------------ |
| `auto_plot_from_result()`     | Maps an agent action (`act`) to the correct chart type |
| `generate_plot_from_result()` | Renders the actual plot from structured data           |
| Individual plotters           | `plot_team_h2h_matrix`, `plot_partnership_graph`, etc. |
| `_plot_fallback_card()`       | Graceful fallback when data is missing                 |

This means **no plot logic* is in the agent or stats layers.
Everything goes through this clean, isolated API.

---

## 📊 8.2 — Available Visualizations

All plots use a dark theme, labelled axes, and value annotations for readability.

| Intent / Action          | Plot Generated            | Description                        |
| ------------------------ | ------------------------- | ---------------------------------- |
| `get_batter_stats`       | **Form Card**             | Runs, Avg, SR bar chart            |
| `compare_players`        | **Head-to-Head Bars**     | Avg/SR comparison                  |
| `get_team_stats`         | **Win Ratio Pie**         | Wins vs Losses + %                 |
| `get_top_players`        | **Top-N Bar Chart**       | Top runs or wickets                |
| `team_head_to_head`      | **H2H Matrix**            | Venue × Season win %               |
| `team_momentum`          | **Momentum Worm**         | Over-by-over cumulative runs       |
| `team_phase_dominance`   | **Phase Dominance Bars**  | PP/Middle/Death RPO                |
| `team_threat_map`        | **Bowling Threat Map**    | Wickets by phase × bowler          |
| `partnership_graph`      | **Partnership Bars**      | Top partnerships (per innings)     |
| `pressure_series`        | **RR Pressure Curve**     | Current vs Required RR             |
| `win_probability_series` | **Win Probability Curve** | % vs ball number                   |
| fallback                 | **Fallback Card**         | Shown when structured data missing |

All plots accept multiple filters (season, venue, city, date range, team).

---

## ▶️ 8.3 — How to Run (CLI Examples)

### A) Player-level plots

```bash
python -m cricket_tools.agent "Show Rohit Sharma batting stats in 2023" --plot
```

### B) Player comparison

```bash
python -m cricket_tools.agent "Compare Rohit Sharma and Virat Kohli in 2023" --plot
```

### C) Team metrics

```bash
python -m cricket_tools.agent "How did Chennai Super Kings perform in 2020?" --plot
```

### D) Top-N plots

```bash
python -m cricket_tools.agent "Top 5 run scorers in Mumbai 2021" --plot
```

### E) Advanced analytics

```bash
python -m cricket_tools.agent "Show momentum worm MI vs CSK 2019" --plot
python -m cricket_tools.agent "Show phase dominance MI vs CSK 2019" --plot
python -m cricket_tools.agent "Bowling threat map for Mumbai Indians 2020" --plot
python -m cricket_tools.agent "Partnership graph for match 335982" --plot
python -m cricket_tools.agent "Pressure curve for match 335982" --plot
python -m cricket_tools.agent "Win probability for match 335982" --plot
```

Each command prints structured text + saves a chart to:

```
outputs/plots/<name>_<timestamp>.png
```

---

## 🧪 8.4 — Direct Module Testing (Advanced)

You can bypass the agent and test the visuals layer directly:

```bash
python - <<'PY'
from cricket_tools.visuals import auto_plot_from_result
from cricket_tools.stats import get_player_stats

res = {
    "act": "get_batter_stats",
    "data": get_player_stats("Rohit Sharma", start="2023-01-01", end="2023-12-31")
}
print(auto_plot_from_result(res, "get_batter_stats"))
PY
```

---

## 🧠 8.5 — Smarter Fallback & Hybrid Logic (Completed)

Step 8 also finalizes the hybrid **Reasoning LLM → Structured Data → Plot** pipeline:

1. Agent tries structured IPL tools first.
2. If tools return *empty* results → reasoning LLM confirms whether answer exists.
3. If still no result → fallback LLM answer + fallback card plot.

This prevents hallucinations and guarantees safe output every time.

---

## 📦 8.6 — Output Format & Conventions

* All plots use consistent dark theme.
* Filenames are timestamped:

  ```
  <plot>_YYYYMMDD_HHMMSS.png
  ```
* Legal deliveries only (no wides/noballs) for balls/overs.
* Bowler-credited dismissals only (`bowled`, `caught`, `lbw`, `stumped`, `hit wicket`).
* City/venue/team normalization applied automatically:

  * “Banglore” → “Bengaluru”
  * “Chepuk” → “M. A. Chidambaram Stadium”
  * “RCB” → “Royal Challengers Bengaluru”

---

## 🛠️ 8.7 — Troubleshooting

| Issue                 | Cause                       | Fix                                  |
| --------------------- | --------------------------- | ------------------------------------ |
| No PNG output         | Empty structured data       | Relax filters or check season        |
| Ambiguous player name | Resolver triggered          | Use full name (e.g. "Rohit Sharma")  |
| Fallback card shown   | No structured rows          | Dataset truly lacks data for filters |
| Slow first command    | SentenceTransformer loading | Cached after first use               |

---

## 🎯 Step-8 Summary

Step-8 completes the entire **visual analytics layer** for CricGPT:

✔ Full suite of IPL-specific charts
✔ Automatic plot routing using `--plot`
✔ Hybrid LLM fallback → safe outputs
✔ All visuals ready for web/Streamlit UI (Step-10)

CricGPT now supports **end-to-end IPL insights with visual explanations**, making it ready for deployment as a chatbot-style web application.

---

If you want, I can now generate

### ✅ Step-9 (removed)

or

### 🚀 Step-10 — Deployment Planning (FastAPI + Streamlit)

just say **"give me Step-10 README"**


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