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

Got it — here is a **clean, professional, minimal, and correct README section for Phase-1**, written exactly how you should put it in your repo.

No extra fluff
No hallucinations
Only what we *actually implemented*.

---

# ✅ **Step 8 — Natural-Language Response Layer (Phase-1 Completion)**

This step introduces a **universal natural-language formatter** that converts the agent’s structured JSON tool outputs into clear, friendly text responses.
It also handles ambiguous names and LLM fallback in a consistent way.

---

## 🎯 **What This Step Achieves**

### ✔️ 1. Natural Language for ALL Tool Outputs

Every structured result (player stats, team stats, comparisons, top lists, advanced analytics, partnership graphs, win-probability curves, etc.) is now converted into clean natural text.

### ✔️ 2. Ambiguous Names → LLM Clarification

If a query refers to an ambiguous player name:

```
stats for rahul
```

The formatter detects:

```json
{"status": "ambiguous", "options": ["Rahul Sharma", "KL Rahul", "Rahul Chahar"]}
```

And returns a friendly message like:

> “Which **Rahul** are you referring to?
> Rahul Sharma, KL Rahul, or Rahul Chahar?”

### ✔️ 3. Fallback → Uses LLM Natural Answer

If structured IPL data is missing, the fallback response already includes a natural-language explanation.
The formatter simply returns that.

### ✔️ 4. Strict No-Hallucination Mode for IPL Data

When structured stats are present, the LLM is instructed to:

* use **only** the numbers in JSON
* never guess or hallucinate
* describe missing fields as “unavailable”

### ✔️ 5. Fully Integrated into `agent.py` via `--nl`

You can now run:

```bash
python -m cricket_tools.agent "show stats for rohit sharma 2020" --nl
```

And the output is natural text instead of raw JSON.

---

# 📁 **Files Added / Modified in This Step**

### **Added**

```
cricket_tools/llm_formatter.py
```

A single unified module that converts structured results into natural language.

### **Updated**

```
cricket_tools/agent.py
```

* Calls the formatter whenever `--nl` is active
* Passes the full structured result into formatter
* Supports NL mode for all tools & plot workflows

---

# 🧪 **How To Test**

### 1) Ambiguous player name

```bash
python -m cricket_tools.agent "stats for rahul" --nl
```

### 2) Player stats with year

```bash
python -m cricket_tools.agent "show stats for rohit sharma 2020" --nl
```

### 3) Compare two players

```bash
python -m cricket_tools.agent "compare kohli and rohit" --nl
```

### 4) Top 5 run scorers in a season

```bash
python -m cricket_tools.agent "top 5 run scorers 2018" --nl
```

### 5) Team performance

```bash
python -m cricket_tools.agent "how did mumbai indians perform in 2019" --nl
```

### 6) Analytics + natural text

```bash
python -m cricket_tools.agent "win probability for mi vs csk 2019" --nl --plot auto
```

### 7) Partnership graph

```bash
python -m cricket_tools.agent "partnership graph for mi vs kkr 2020" --nl
```

---

# 🏁 **Outcome**

Step 8 completes **Phase-1** of the chatbot conversion:

* Agent backend stays unchanged
* All structured results now become user-friendly text
* Ambiguity and fallback handled cleanly
* Plot workflows supported
* Zero hallucination for IPL data

This prepares the project for **Phase-2 (run_query API)** and then **Phase-3 (FastAPI)** + **Phase-4 (Streamlit CricGPT chat UI)**.

---

Let me know when you finish pushing changes, then we begin **Phase-2**.

---

Here is the **clean, complete, correct README for Phase-2**, fully consistent with your project, your architecture, and everything we implemented.

---

# 🚀 **Phase-2 — Unified Backend Layer (`run_query`)**

Phase-2 introduces a **single universal backend function** for CricGPT that allows:

* CLI
* FastAPI
* Streamlit UI
* Jupyter notebooks
* Unit tests
* External integrations

to all use **the same orchestration pipeline**.

This is the step that turns your project from *"CLI-only tool"* into a proper **API-driven system** ready for deployment.

---

# 🎯 **Goal of Phase-2**

Until Phase-1, natural language formatting was done inside **agent.py** only in CLI mode using `--nl`.

CLI workflow looked like this:

```
User Query → Planner → Tool → Structured Output → LLM Formatter → Print
```

But:

* There was **no API function** to call CricGPT directly.
* Plot generation logic lived inside CLI.
* Memory merging happened only inside agent.
* Each environment (CLI, UI, API) would have required custom logic.
* Hard to embed CricGPT into a website or app.

**Phase-2 fixes all this.**

---

# 💡 What Phase-2 Adds

A new file:

```
cricket_tools/runner.py
```

with one main function:

```python
run_query(
    message: str,
    backend="llm_reasoning",
    plot=False,
    fallback=True,
    session_id=None
)
```

This function becomes the **single entry point** for the entire CricGPT system.

---

# 🧩 **What `run_query()` Does**

It performs *all* orchestration steps in one place:

### 1️⃣ Create a `CricketAgent`

Configures backend and fallback settings.

### 2️⃣ Execute the planning + tool pipeline

Same logic that CLI used internally.

### 3️⃣ Apply natural-language formatter

Uses the universal `llm_formatter.py` to generate smooth, non-hallucinating prose.

### 4️⃣ Generate optional plots

Automatically selects correct plotting function.

### 5️⃣ Manage session ID + memory

Returns a stable session token (UUID) if not provided.

### 6️⃣ Returns a **clean, predictable JSON** object:

```json
{
  "reply": "... natural language output ...",
  "act": "get_batter_stats",
  "result": { ... structured IPL data ... },
  "plot_path": "outputs/plots/xxxx.png",
  "meta": {},
  "trace": [...],
  "session_id": "...."
}
```

This becomes the base interface for:

* CLI
* FastAPI API
* Streamlit chat UI
* Tests
* Notebooks
* Later: mobile app or Electron app

---

# 🧱 **File Added in this Phase**

### ✔️ `cricket_tools/runner.py`

Contains the unified orchestrator.

### ✔️ Small update in `agent.py`

When CLI is called with `--nl`, instead of using the internal formatter, it now calls:

```python
resp = run_query(...)
```

This activates the unified backend.

---

# 🧪 How to Test Phase-2

### **1. Natural-language mode via CLI uses the unified backend**

```bash
python -m cricket_tools.agent "rohit sharma 2020" --nl
```

Expected:

* Uses reasoning planner
* Executes structured tool
* Applies natural language formatter
* (If plot auto: generates visuals)

### **2. Test plot generation via unified backend**

```bash
python -m cricket_tools.agent "win probability mi vs csk 2020" --nl --plot auto
```

Expected:

* JSON pipeline + visual generation
* Natural language output describing win-probability curve
* File saved under `outputs/`

### **3. Test ambiguous names**

```bash
python -m cricket_tools.agent "stats for rahul" --nl
```

Expected:

* Ambiguous → uses llm_formatter → friendly clarification

### **4. Directly test `runner.py` from Python**

```python
from cricket_tools.runner import run_query

resp = run_query("show kohli vs rohit", backend="llm_reasoning", plot=True)
print(resp["reply"])
```

---

# 🏆 Outcome of Phase-2

### ✔️ CricGPT now has a **professionally designed backend**

Everything—CLI/UI/API—will use **one** function.

### ✔️ Natural-language answers come from a single module

(Phase-1 formatter).

### ✔️ Ready for Phase-3 (FastAPI)

The API will be unbelievably simple:

```python
@app.post("/ask")
def ask(q: str):
    return run_query(q, backend="llm_reasoning", plot=False)
```

### ✔️ Ready for Phase-4 (Streamlit Chat UI)

The chat UI will call only `run_query`.

### ✔️ Consistent output everywhere

Every client gets structured + natural output + plot path.

---

# 🔮 What’s Next?

Here is the official sequence:

### **Phase-3 — FastAPI Inference Server**

Serve CricGPT via `/ask` endpoints.

### **Phase-4 — Streamlit Web UI**

Interactive CricGPT chatbot interface.

### **Phase-5 (Optional) — Vector Search Engine**

Embedding-based semantic search for richer queries.

---

# ✔️ Phase-2 is 100% Complete

You have:

* `runner.py`
* `agent.py` modifications
* Full unified backend
* Verified with `--nl` and normal CLI queries

---

# 🚀 Phase-3 — FastAPI Production API Layer

Phase-3 exposes the entire CricGPT engine (Phase-2 `run_query()`) as a clean, stable, production-ready **FastAPI server**.

This allows:

* Web UI / Streamlit frontends
* Integration with external apps
* Mobile clients
* CI/CD + deployment
* Test automation

All agent logic remains inside `cricket_tools/`.
FastAPI only orchestrates requests through a single API endpoint.

---

## 📁 Project Structure (Phase-3)

```
app/
   ├── main.py              # FastAPI app entrypoint
   ├── routers/
   │      └── query.py      # /query endpoint
   ├── schemas/
   │      └── query.py      # Pydantic request+response models
   └── __init__.py

cricket_tools/
   └── runner.py            # Phase-2 unified backend
```

---

## 📌 `/query` Endpoint

**POST /query**

### Request Body

```json
{
  "query": "rohit sharma 2020",
  "backend": "llm_reasoning",
  "plot": false,
  "fallback": true,
  "session_id": null
}
```

### Response

Returns a unified dictionary from `run_query()`:

```json
{
  "reply": "...natural language answer...",
  "act": "get_batter_stats",
  "result": {...structured stats...},
  "plot_path": null,
  "trace": [...],
  "session_id": "uuid"
}
```

---

## ▶️ Running the API

From the project root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)
* Health check → [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Quick Test (curl)

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "rohit sharma 2020"}'
```

---

## 🧪 Quick Test (Python)

```python
import requests
r = requests.post("http://localhost:8000/query", json={"query": "rohit sharma 2020"})
print(r.json())
```

---

## 🎯 Summary

Phase-3 adds a **stable API layer** on top of your unified agent backend:

✔ No duplication
✔ Same logic as CLI
✔ Natural-language included
✔ Plotting supported (plot: true)
✔ Ready for deployment and dockerization

---

# 🎉 Phase-3 complete

## Phase 4 — Streamlit Chat UI & Dashboard

In Phase 4 we wrapped the core **CricGPT agent** in a simple, interactive **Streamlit web app** so you can chat with the system, see structured answers, and optionally auto-generate plots without touching the CLI.

---

### 🎯 Objectives

- Provide a **single-page web UI** for CricGPT.
- Support **free-text chat** with the same reasoning pipeline as the CLI.
- Display **natural-language answers** and structured traces.
- Allow **optional auto-plotting** (momentum, H2H, partnerships, pressure, etc.).
- Keep everything **stateless per request** except for a lightweight chat history.

---

### 📂 Key Files

- `streamlit_app.py`  
  Main Streamlit entry point. Handles:
  - Text input box + “Send” button.
  - Display of model reply (natural language).
  - Optional display of plot path / embedded image.
  - Sidebar controls for backend & plotting.

- `cricket_tools/runner.py`  
  High-level entry (`run_query(...)`) used by both CLI and Streamlit:
  - Routes user message to `CricketAgent`.
  - Enables **natural-language mode** (`agent.natural_language = True`).
  - Optionally requests **auto-plotting**.
  - Returns a unified response dict for the UI:
    - `reply`: natural-language answer  
    - `plot_path`: optional plot file path  
    - `raw`: raw agent result (for debugging, if needed)

---

### 🧱 What Phase 4 Adds

1. **Chat-style interface**
   - Single text input box for the user’s query.
   - Press `Enter` or click **Send** to submit.
   - Response appears as formatted markdown (player stats, comparisons, explanations).

2. **Auto-Generate Plots toggle**
   - Sidebar checkbox: **“Auto-generate plots (when available)”**.
   - When enabled:
     - `run_query(..., plot=True)` is called.
     - Analytics tools (H2H, momentum, phase dominance, threat map, partnership, pressure, win-prob) can produce plots.
   - When disabled:
     - Only natural-language + structured answer, no plotting overhead.

3. **Backend & model selection (optional)**
   - Sidebar drop-down for backend:
     - e.g. `auto`, `llm_reasoning`, `semantic`, etc. (depending on your config).
   - Uses the same logic as CLI to resolve the planner:
     - If `LLM_PROVIDER` is set → LLM/Reasoning backend.
     - Else → semantic or mock fallback.

4. **Session-level chat history**
   - Recent user queries and assistant answers are stored in `st.session_state`.
   - Displayed as alternating “User” and “CricGPT” blocks.
   - **Clear history** button in sidebar to reset the current session’s conversation.

5. **Debug / trace visibility (optional)**
   - For development, we can show:
     - Chosen tool (e.g. `get_batter_stats`, `team_head_to_head`, `pressure_series`).
     - Parsed arguments (player, team, season, etc.).
     - Any fallback usage.
   - Usually hidden from normal users; can be toggled via a sidebar debug checkbox.

---

### ▶️ How to Run

From the project root (with your virtualenv activated):

```bash
# 1. Activate venv (example)
source .venv/bin/activate

# 2. Run Streamlit on 0.0.0.0:8501
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
````

Then open the URL shown in the terminal (e.g. `http://localhost:8501` or the forwarded port in Codespaces) in your browser.

> Make sure your `.env` / `LLM_PROVIDER` / model keys are configured as in previous phases.

---

### 💬 Usage Workflow in the UI

1. **Open app** → see a central chat area and a sidebar.

2. In the **sidebar**:

   * Choose **backend** (e.g. `llm_reasoning`).
   * Tick/untick **“Auto-generate plots (when available)”**.
   * (Optional) Clear chat history.

3. Type a query in the **chat box** and hit **Send**.

4. The app internally calls:

   ```python
   from cricket_tools.runner import run_query

   resp = run_query(
       user_message,
       backend=<selected_backend>,
       plot=<auto_plot_checkbox>,
       fallback=True,
       session_id=None,
   )
   ```

5. The UI renders:

   * `resp["reply"]` as markdown.
   * If `resp["plot_path"]` is present and auto-plot is enabled:

     * The plot is shown (or at least the file path / image).

---

### 🧪 Example Queries to Try in Streamlit

You can test all logic implemented so far directly in the UI:

#### Basic player stats

* `stats for Rohit`
* `batting stats for Virat Kohli in 2016`
* `bowling stats for Bumrah`
* `form of Gill in recent years`

#### Ambiguous name resolution

* `stats for pandya`
  → You should get a clarification asking **which Pandya**.
  → Reply with `Hardik Pandya` → tool runs with resolved player.

#### Comparisons

* `compare Rohit Sharma and Virat Kohli in 2019`
* `Kohli vs Rohit head to head batting in 2016`

#### Team queries

* `How did CSK perform in 2018?`
* `MI performance at Wankhede`
* `top 5 run scorers in 2013`
* `top 10 wicket takers in 2019`

#### Analytics tools (with Auto-plot ON / OFF)

* `CSK vs MI head to head`
* `momentum worm for CSK vs MI`
* `phase dominance for CSK vs MI in 2019`
* `threat map for Mumbai Indians`
* `partnership graph for CSK vs MI 2019`
* `pressure curve for CSK vs MI`
* `win probability series for CSK vs MI final 2019`

When **Auto-generate plots** is ON:

* You get both:

  * Natural-language summary (via `llm_formatter` Phase-3B).
  * Corresponding plot (momentum, phase, H2H, etc.), where implemented.

When OFF:

* You only see the **textual explanation**, which is still understandable.

---

### ℹ Notes & Limitations

* The app is intended as a **developer / analyst UI**, not yet a polished public product.
* Plot generation depends on:

  * Data availability for that match/season.
  * Corresponding functions implemented in `cricket_tools.analytics` and `cricket_tools.visuals`.
* For **non-IPL queries** (World Cup, bilateral series, etc.):

  * The reasoning planner routes to `llm_fallback`.
  * Streamlit UI still shows a **natural-language answer**, but not IPL stats.

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