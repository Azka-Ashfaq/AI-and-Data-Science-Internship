"""
state.py — State schema for the AFL LangGraph Orchestration System (Task 1).

Covers:
  - user_query: current user query
  - messages / conversation_history: multi-turn dialogue history
  - detected_intent: factual | retrieval | prediction | off_topic | ambiguous_fallback
  - extracted_entities & resolved_entities: parsed entities (teams, players, dates)
  - tool_results & tool_called: outputs from prediction or retrieval tools
  - validation_status & validation_error: flags from self-correction / fallback validation
  - feature_drivers: top 2-3 features grounding ML predictions
  - final_response: the framed final answer delivered to the user
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AFLGraphState(TypedDict, total=False):
    """LangGraph state schema for AFL Chat, Retrieval & Prediction orchestration."""
    
    # Message stream with reducer for multi-turn conversations
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Core query & routing metadata
    user_query: str
    conversation_history: List[Dict[str, str]]
    detected_intent: str  # 'factual' | 'retrieval' | 'prediction' | 'off_topic' | 'ambiguous_fallback'
    intent_confidence: float
    intent_rationale: str
    
    # Entity extraction and resolution
    extracted_entities: Dict[str, Any]  # e.g., {'home_team': 'Pies', 'away_team': 'Cats', 'date': 'this week'}
    resolved_entities: Dict[str, Any]   # e.g., {'home_team': 'Collingwood Magpies', ...}
    unresolved_entities: List[str]      # e.g., ['Smith'] (multiple matches) or ['Titans'] (unknown)
    
    # Tool execution results
    tool_called: Optional[str]
    tool_results: Optional[Any]
    feature_drivers: Optional[List[Dict[str, Any]]]  # e.g., [{'feature': 'home_win_rate_last5', 'importance': 0.35}]
    
    # Validation & self-correction status
    validation_status: str  # 'valid' | 'needs_clarification' | 'unsupported_fallback' | 'tool_error'
    validation_error: Optional[str]
    clarification_prompt: Optional[str]

    # Day 5 hardening: abuse/probing tracking. The caller (e.g. the FastAPI
    # session layer) passes in the streak count from the previous turn;
    # router_node increments it on another off_topic/jailbreak hit and
    # resets it to 0 on any on-scope turn. refusal_node reads it to decide
    # whether to escalate the refusal wording. Callers should persist the
    # returned value and pass it back in on the next turn of the same
    # conversation_id.
    off_topic_streak: int

    # Final framed response
    final_response: str
