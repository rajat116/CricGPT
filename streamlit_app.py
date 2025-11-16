#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from cricket_tools.runner import run_query
import os

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="OverPrompt — IPL Analytics Chat",
    page_icon="🏏",
    layout="wide"
)

# -------------------------------------------------------------
# Session State Setup
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

backend = st.sidebar.selectbox(
    "Backend",
    ["llm_reasoning", "semantic", "mock"],
    index=0
)

use_fallback = st.sidebar.checkbox(
    "Enable LLM fallback",
    value=True
)

gen_plots = st.sidebar.checkbox(
    "Auto-generate plots",
    value=False
)

with st.sidebar.expander("💡 Example queries to try", expanded=True):
    st.markdown(
        """
**🧍 Player performance (text or simple bar plot)**  
- `Show Rohit Sharma batting stats in 2023`  
- `Bowling stats for Jasprit Bumrah last year`  
- `Show Virat Kohli batting for RCB in 2016`  

**⚔️ Player vs player comparison**  
- `Compare Rohit and Virat in Chepuk 2023`  
- `Compare Dhoni and Gill in Ahmedabad`  
- `Compare Bumrah vs Shami this year`  

**🧣 Team performance (by venue/season)**  
- `Show MI performance at Wankhede`  
- `How did Chennai Super Kings perform in 2020`  
- `Team performance of RCB in Banglore 2019`  
- `How did RCB perform at Chinnaswamy in 2018`  

**🏅 Top players / rankings**  
- `Top 5 run scorers in Chennai 2021`  
- `Top wicket takers at Eden Gardens`  
- `Who scored the most runs for Mumbai Indians last season`  

**📊 Advanced broadcast-style analytics**  
*(Turn **"Auto-generate plots"** ON for these to see charts)*  
- `Head to head matrix for MI vs CSK`  
- `Momentum worm for MI vs CSK in 2019`  
- `Phase dominance MI vs CSK in 2019`  
- `Threat map for Mumbai Indians bowlers in 2020`  
- `Partnership graph for match <MATCH_ID>`  
- `Pressure curve for match <MATCH_ID>`  
- `Win probability curve for match <MATCH_ID>`  

**🔍 Fuzzy names / typo handling**  
- `Top players in Chepuk stadium`  
- `Compare Kohli vs Rohit at Chinnaswamy`  
- `Show KKR performance in Kolkata`  

**🌍 Non-IPL / fallback reasoning (uses LLM only)**  
- `How did Rohit Sharma perform in 2016 ODI World Cup`  
- `Who scored the most centuries in Asia Cup 2023`  
- `How many runs did Kohli make in T20 World Cup 2022`  
- `Top wicket takers in ICC World Cup 2019`  
        """
    )

st.sidebar.info("Plots will auto-appear under the assistant's response.")

# -------------------------------------------------------------
# Title Section
# -------------------------------------------------------------
st.title("🏏 OverPrompt — IPL Chat, Analytics & Plots")

st.write(
    "OverPrompt lets you chat about IPL batsmen, bowlers, teams and seasons, "
    "run smart comparisons, and auto-generate visuals: head-to-head matrices, "
    "momentum worms, phase dominance, threat maps, partnerships, pressure curves, "
    "win-probability timelines, and more."
)

# -------------------------------------------------------------
# Chat History Renderer
# -------------------------------------------------------------
def render_message(role, content):
    """Chat bubble renderer."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)


# Render previous messages
for msg in st.session_state["messages"]:
    render_message(msg["role"], msg["content"])
    if "plot" in msg and msg["plot"] is not None:
        st.image(msg["plot"], use_column_width=True)


# -------------------------------------------------------------
# MAIN CHAT INPUT
# -------------------------------------------------------------
user_input = st.chat_input("Ask your IPL question…")

if user_input:
    # 1. Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_message("user", user_input)

    # 2. Call unified backend
    out = run_query(
        message=user_input,
        backend=backend,
        fallback=use_fallback,
        plot=gen_plots,
        session_id=st.session_state["session_id"]
    )

    reply_text = out["reply"]
    plot_path = out.get("plot_path", None)

    # 3. Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply_text)

        if gen_plots and plot_path:
            st.image(plot_path, use_column_width=True)

    # 4. Store in history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": reply_text,
        "plot": plot_path
    })