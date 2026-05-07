import os
import uuid
import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ── 0. Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Wellbeing Lab Agent", page_icon="🎯")

# ── 1. Configuration & Environment ───────────────────────────────────────────
def _bootstrap_env() -> None:
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception: pass
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError: pass

_bootstrap_env()

# ── 2. Import Agent ──────────────────────────────────────────────────────────
# This agent loads its system prompt from "prompts/agent.prompt"
from PromptBasedAgent import graph 

# ── 3. Team C Testing Logic ──────────────────────────────────────────────────
SAMPLE_TRIGGER = {
    "trigger_id": "trigger_2026_05_03_low_daily_steps",
    "user_id": "demo_user_001",
    "trigger_type": "low_daily_steps",
    "risk_level": "Watch",
    "current_date": "2026-05-03",
    "today_steps": 1522,
    "baseline_daily_steps": 6215.7,
    "threshold": 4351.0,
    "difference_percent": -75.5,
    "reason_flags": ["low_activity"],
    "suggested_action": "send gentle activity check-in"
}

def build_trigger_prompt(trigger_event: dict) -> str:
    return f"""
You are receiving a processed trigger JSON from the wearable data pipeline.
Use only this trigger JSON. Do not ask for raw data.
Generate the required chatbot and health coach outputs according to your system prompt.

Trigger JSON:
{json.dumps(trigger_event, indent=2)}
"""

# ── 4. Main UI ───────────────────────────────────────────────────────────────
st.title("🎯 Wellbeing Technologies Lab")

tab1, tab2 = st.tabs(["💬 General Chat", "🧪 Trigger Testing (Team C)"])

# --- TAB 1: GENERAL CHAT ---
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message…", key="chat_input")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": "general_chat"}}
            input_msgs = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.chat_history]
            result = graph.invoke({"messages": input_msgs}, config=config)
            response = result["messages"][-1].content
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- TAB 2: TRIGGER TESTING ---
with tab2:
    st.header("Trigger-to-LLM Test Interface")
    st.write("Use this section to verify how the agent handles JSON triggers from the pipeline.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input: Trigger JSON")
        st.json(SAMPLE_TRIGGER)
    
    with col2:
        st.subheader("Action")
        if st.button("🚀 Send Trigger to Agent"):
            trigger_prompt = build_trigger_prompt(SAMPLE_TRIGGER)
            
            with st.spinner("Agent is processing..."):
                # CRITICAL: We use the graph here to test the 'agent.prompt' file
                config = {"configurable": {"thread_id": "testing_thread"}}
                result = graph.invoke({"messages": [HumanMessage(content=trigger_prompt)]}, config=config)
                st.session_state.latest_test_output = result["messages"][-1].content

    if "latest_test_output" in st.session_state:
        st.divider()
        st.subheader("Agent Output")
        st.info("Check if the JSON structure and tone below match the requirements.")
        st.markdown(st.session_state.latest_test_output)
