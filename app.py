import os
import uuid
import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ── 0. Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Wellbeing Lab Bot", page_icon="🎯")

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

# ── 2. Imports ───────────────────────────────────────────────────────────────
from PromptBasedAgent import graph 

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ── 3. Team C Logic (Health Trigger Analysis) ────────────────────────────────
LLM_SYSTEM_CONTEXT = """
You are a wearable-sensor-based health coaching chatbot.
Return only JSON in this exact format:
{
 "send_alert": true,
 "user_message": "",
 "health_coach_alert": "",
 "explanation": "",
 "suggested_action": "",
 "escalation_needed": false
}
"""

def mock_llm_response(trigger_event: dict) -> dict:
    return {
        "send_alert": True,
        "user_message": "Hi, your step count today is lower than your usual pattern.",
        "health_coach_alert": f"Low activity trigger for {trigger_event.get('user_id')}.",
        "explanation": "Steps lower than baseline.",
        "suggested_action": "Send activity check-in.",
        "escalation_needed": False
    }

def generate_llm_response(trigger_event: dict) -> dict:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"):
        return mock_llm_response(trigger_event)
    
    client = OpenAI()
    prompt_text = f"Trigger JSON:\n{json.dumps(trigger_event, indent=2)}\nGenerate JSON now."
    
    try:
        # Note: Using standard Chat Completion API
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": LLM_SYSTEM_CONTEXT},
                {"role": "user", "content": prompt_text}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return mock_llm_response(trigger_event)

# ── 4. Main UI ───────────────────────────────────────────────────────────────
st.title("🎯 Wellbeing Technologies Lab")

tab1, tab2 = st.tabs(["💬 Personal Agent", "📊 Health Alerts"])

# --- TAB 1: YOUR PROMPT-BASED AGENT ---
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_seed" not in st.session_state:
        st.session_state.session_seed = str(uuid.uuid4())

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message…")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        lc_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.chat_history
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"thread_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, st.session_state.session_seed))}}
                result = graph.invoke({"messages": lc_messages}, config=config)
                response = result["messages"][-1].content
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- TAB 2: COLLEAGUE'S TRIGGER SYSTEM ---
with tab2:
    st.header("Trigger-Based Coaching")
    # Mock trigger data simulating Team B's output
    sample_trigger = {"user_id": "user_456", "today_steps": 1200, "baseline_daily_steps": 5000}
    
    st.json(sample_trigger)
    if st.button("Generate Health Alert"):
        with st.spinner("Processing trigger..."):
            alert_data = generate_llm_response(sample_trigger)
            st.session_state.latest_alert = alert_data

    if "latest_alert" in st.session_state:
        alert = st.session_state.latest_alert
        st.info(f"**Coach Alert:** {alert.get('health_coach_alert')}")
        st.success(f"**Bot Message:** {alert.get('user_message')}")
        st.write(f"**Next Step:** {alert.get('suggested_action')}")
