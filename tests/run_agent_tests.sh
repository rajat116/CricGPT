#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_agent_tests.sh — Batch-test the Cricket Chat Agent
# Runs many sample questions through your agent and saves results to a log file.
# -----------------------------------------------------------------------------

BACKEND=${1:-auto}              # choose backend: auto | mock | semantic | openai
LOGFILE="tests/test_results_$(date +%Y%m%d_%H%M%S).log"

mkdir -p tests

echo "🧪 Starting batch test with backend='$BACKEND'..."
echo "Results will be saved to: $LOGFILE"
echo "----------------------------------------------------------" | tee "$LOGFILE"

# Helper to run each query and log output
run_test () {
  local query="$1"
  echo -e "\n❓ Query: $query" | tee -a "$LOGFILE"
  echo "----------------------------------------------------------------------" >> "$LOGFILE"
  python -m cricket_tools.agent "$query" --backend "$BACKEND" >> "$LOGFILE" 2>&1
  echo -e "\n" >> "$LOGFILE"
}

# ------------------------------------------------------------------
# 🎯 Test Suites
# ------------------------------------------------------------------

# -- Player performance
run_test "Show Rohit Sharma batting stats in 2023"
run_test "Bowling stats for Jasprit Bumrah last year"
run_test "Predict KL Rahul performance next match"

# -- Comparison
run_test "Compare Rohit and Virat in Chepuk 2023"
run_test "Compare Dhoni and Gill in Ahmedabad"
run_test "Compare Bumrah vs Shami this year"

# -- Team stats
run_test "Show MI performance at Wankhede"
run_test "How did Chennai Super Kings perform in 2020"
run_test "Team performance of RCB in Banglore 2019"

# -- Top players
run_test "Top 5 run scorers in Chennai 2021"
run_test "Top wicket takers at Eden Gardens"
run_test "Who scored the most runs for Mumbai Indians last season"

# -- Edge / typo tests
run_test "Top players in Chepuk stadium"
run_test "Compare Kohli vs Rohit at Chinnaswamy"
run_test "Predict Shubman Gill form at Narenda Modi"

# -- Free-text mixed cases
run_test "How did SRH do this year"
run_test "Show KKR performance in Kolkata"
run_test "Who are best batsmen at M. A. Chidambaram Stadium"
run_test "Compare Dhoni and Pant at Delhi in 2020"

# -- Extended edge & cross-filter tests
run_test "How did RCB perform at Chinnaswamy in 2018"
run_test "Show Virat Kohli batting for RCB in 2016"
run_test "Show RCB performance in Bengaluru"

# -- Fallback reasoning tests (non-IPL context)
echo "----------------------------------------------------------" | tee -a "$LOGFILE"
echo "🌍 Testing fallback reasoning (non-IPL queries)" | tee -a "$LOGFILE"

run_test "How did Rohit Sharma perform in 2016 ODI World Cup"
run_test "Who scored the most centuries in Asia Cup 2023"
run_test "How many runs did Kohli make in T20 World Cup 2022"
run_test "Top wicket takers in ICC World Cup 2019"
run_test "Best batsmen in bilateral series against England"

# ------------------------------------------------------------------
echo "✅ Batch testing complete."
echo "Full log saved to: $LOGFILE"