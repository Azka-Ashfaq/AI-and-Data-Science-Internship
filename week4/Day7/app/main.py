"""
Day 7 - Task 1: FastAPI Backend exposing the Day 5 LangGraph agent.
Endpoints:
  GET  /health        -> liveness check
  POST /chat          -> one conversational turn
  GET  /docs          -> Swagger UI (auto)
"""
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# point to the Day 5 agent
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "day5", "src"))

from graph import build_graph
from state import new_state

app = FastAPI(
    title="RealEstate Hub Voice Agent API",
    description="UrduLish AI voice agent for Pakistani real estate — Ayesha",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
SESSIONS: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    client_phone: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Optional[str]
    appointment_status: Optional[str]
    node_trace: list


@app.get("/health")
def health():
    return {"status": "ok", "service": "realestate-hub-voice-agent", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Load existing state or create fresh
    state = SESSIONS.get(session_id)
    if state is None:
        state = new_state(client_phone=req.client_phone or f"web-{session_id[:8]}")

    # Set current user message BEFORE invoking graph
    state["current_user_text"] = req.message

    # Ensure tool_outputs dict exists
    if "tool_outputs" not in state or state["tool_outputs"] is None:
        state["tool_outputs"] = {}

    # Debug logging (visible in server terminal)
    print(f"[DEBUG] session={session_id} user_text={req.message!r}")
    print(f"[DEBUG] BEFORE: intent={state.get('intent')} history_len={len(state.get('conversation_history', []))}")

    try:
        state = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Persist updated state
    SESSIONS[session_id] = state

    print(f"[DEBUG] AFTER:  intent={state.get('intent')} history_len={len(state.get('conversation_history', []))}\n")

    trace = [t.get("node") for t in state.get("execution_trace", [])][-10:]

    return ChatResponse(
        session_id=session_id,
        reply=state["conversation_history"][-1]["text"],
        intent=state.get("intent"),
        appointment_status=state.get("appointment_status"),
        node_trace=trace,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "intent": state.get("intent"),
        "appointment_status": state.get("appointment_status"),
        "conversation_length": len(state.get("conversation_history", [])),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)