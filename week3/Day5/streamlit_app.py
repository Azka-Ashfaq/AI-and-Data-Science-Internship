"""
streamlit_app.py — Minimal chat UI for demoing the AFL assistant (Day 5, Task 3).

Run:
    pip install streamlit
    streamlit run streamlit_app.py

Talks directly to the compiled LangGraph app (no need to run api.py first),
but keeps the same session-memory + entity-carryover behavior api.py uses,
so the demo experience matches what the FastAPI endpoint would give.
"""

import uuid

import streamlit as st

from graph_app import app as graph_app
from session_logic import run_with_carryover

st.set_page_config(page_title="AFL Assistant", page_icon="🏉")
st.title("🏉 AFL Assistant")
st.caption("Stats, head-to-heads, and match/player predictions — AFL only.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "off_topic_streak" not in st.session_state:
    st.session_state.off_topic_streak = 0
if "last_entities" not in st.session_state:
    st.session_state.last_entities = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            with st.expander("Prediction details"):
                st.json(msg["metadata"])

if prompt := st.chat_input("Ask about AFL stats, head-to-heads, or predictions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_for_graph = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]
    # Same shared carryover logic api.py uses, so the demo UI and the API
    # behave identically for multi-turn follow-ups.
    final = run_with_carryover(
        graph_app, prompt, history_for_graph, st.session_state.off_topic_streak,
        st.session_state.last_entities,
    )

    st.session_state.off_topic_streak = final.get("off_topic_streak", 0) or 0
    if final.get("resolved_entities"):
        st.session_state.last_entities = final["resolved_entities"]

    response_text = final.get("final_response", "I couldn't generate a response.")
    metadata = None
    tr = final.get("tool_results")
    if final.get("tool_called") in ("predict_match_winner", "predict_top_player") and isinstance(tr, dict):
        metadata = {
            k: v for k, v in tr.items()
            if k in ("win_probability", "confidence_level", "predicted_winner",
                      "ranked_players", "feature_drivers")
        }

    with st.chat_message("assistant"):
        st.markdown(response_text)
        if metadata:
            with st.expander("Prediction details"):
                st.json(metadata)

    st.session_state.messages.append({
        "role": "assistant", "content": response_text, "metadata": metadata,
    })

with st.sidebar:
    st.markdown("### Session")
    st.text(f"conversation_id: {st.session_state.conversation_id[:8]}...")
    st.text(f"turns: {len(st.session_state.messages) // 2}")
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.off_topic_streak = 0
        st.session_state.last_entities = {}
        st.rerun()