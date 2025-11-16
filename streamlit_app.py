#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from cricket_tools.runner import run_query
import os

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="CricGPT — IPL Analytics Chat",
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
    value=True
)

st.sidebar.info("Plots will auto-appear under the assistant's response.")

# -------------------------------------------------------------
# Title Section
# -------------------------------------------------------------
st.title("🏏 CricGPT — IPL Chat + Analytics + Plots")

st.write(
    "Ask anything about IPL batsmen, bowlers, teams, seasons, comparisons, win-probability, "
    "head-to-head, partnerships, pressure curves, momentum, and more."
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