"""
Day 3 - Task 1: Streaming Voice Pipeline (simulated) + latency budget.

Real audio I/O (microphone capture, telephony, TTS playback) can't run in
this text-only environment, so STT and TTS are represented as fixed,
clearly-labelled placeholder delays taken from typical provider figures
(Deepgram streaming STT ~150-300ms to first partial; Fish Audio ~200-400ms
to first audio chunk). Everything else in this pipeline is REAL and
measured on the actual clock: retrieval (SQL + vector), objection
detection, and the LLM call (Day 2's rag_answer, now Groq-based).

This lets Task 1's "keep total turn latency under 2 seconds" budget be
checked honestly against real numbers for the parts that can run here,
with the audio-hardware parts clearly marked as assumptions.
"""
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "day2", "src"))

from conversation_state import ConversationState
from speech_behaviors import add_natural_behaviors
from objection_handler import handle_objection

try:
    from rag_answer import build_context, call_llm, answer as rag_answer_dry
    from retriever import structured_search
    DAY2_AVAILABLE = True
except Exception as e:
    DAY2_AVAILABLE = False
    _IMPORT_ERROR = e

# Simulated audio-hardware latency (ms) -- NOT measured on this machine,
# taken from typical provider docs (see module docstring).
SIMULATED_STT_MS = 220
SIMULATED_TTS_FIRST_CHUNK_MS = 300


def process_turn(state: ConversationState, user_text: str, turn_index: int, dry_run=True):
    """Runs one full conversational turn and returns a dict with the
    spoken reply plus a latency breakdown for every stage."""
    t_start = time.time()
    timings = {}

    # 1. STT (simulated)
    timings["stt_ms"] = SIMULATED_STT_MS

    # 2. Update memory / slot-filling (real, measured)
    t0 = time.time()
    state.update_from_utterance(user_text)
    timings["memory_update_ms"] = round((time.time() - t0) * 1000, 1)

    # 3. Objection check (real, measured)
    t0 = time.time()
    property_ctx = None
    if state.last_result_ids and DAY2_AVAILABLE:
        try:
            df = structured_search(limit=1)  # cheap lookup for a sample agency name
            if not df.empty:
                property_ctx = {"agency": df.iloc[0]["agency"]}
        except Exception:
            pass
    objection_category, objection_text = handle_objection(user_text, property_ctx)
    timings["objection_check_ms"] = round((time.time() - t0) * 1000, 1)

    # 4. Retrieval + LLM answer (real, measured) -- reuses Day 2's pipeline
    t0 = time.time()
    if not DAY2_AVAILABLE:
        raw_answer = ("[Day 2 pipeline not found -- run this from within the project "
                       "layout so ../day2/src is reachable, or set DAY2_SRC_PATH]")
    elif objection_text:
        raw_answer = objection_text
    else:
        city = state.slots["city"]
        purpose = state.slots["purpose"]
        max_price = state.slots["budget"]
        min_bedrooms = state.slots["bedrooms"]
        location = state.slots["location_hint"]
        if dry_run or not os.environ.get("LLM_PROVIDER"):
            raw_answer = rag_answer_dry(user_text, city=city, purpose=purpose,
                                         max_price=max_price, min_bedrooms=min_bedrooms,
                                         location=location, dry_run=True)
        else:
            context = build_context(user_text, city=city, purpose=purpose,
                                     max_price=max_price, min_bedrooms=min_bedrooms,
                                     location=location)
            raw_answer = call_llm(user_text, context)
    timings["retrieval_plus_llm_ms"] = round((time.time() - t0) * 1000, 1)

    # 5. Natural speech behaviors wrap (real, measured)
    t0 = time.time()
    spoken_text, behavior_tags = add_natural_behaviors(user_text, raw_answer, turn_index)
    if objection_category:
        behavior_tags.append(f"objection:{objection_category}")
    timings["behavior_wrap_ms"] = round((time.time() - t0) * 1000, 1)

    # 6. TTS (simulated)
    timings["tts_first_chunk_ms"] = SIMULATED_TTS_FIRST_CHUNK_MS

    timings["total_ms"] = round(
        timings["stt_ms"] + timings["memory_update_ms"] + timings["objection_check_ms"] +
        timings["retrieval_plus_llm_ms"] + timings["behavior_wrap_ms"] + timings["tts_first_chunk_ms"], 1
    )
    timings["wall_clock_measured_ms"] = round((time.time() - t_start) * 1000, 1)

    state.log_turn("caller", user_text)
    state.log_turn("agent", spoken_text)

    return {
        "user_text": user_text, "spoken_text": spoken_text,
        "behavior_tags": behavior_tags, "objection_category": objection_category,
        "timings": timings, "slots": dict(state.slots),
    }


if __name__ == "__main__":
    state = ConversationState()
    turns = [
        "Budget 3 crore hai, DHA Lahore mein 3 bedroom chahiye.",
        "Yeh property thori expensive lag rahi hai.",
        "Theek hai, visit book kar dein.",
    ]
    for i, t in enumerate(turns):
        result = process_turn(state, t, i, dry_run=True)
        print(f"[{i}] Caller: {t}")
        print(f"    Agent:  {result['spoken_text'][:200]}")
        print(f"    Timings: {result['timings']}")
        print()
