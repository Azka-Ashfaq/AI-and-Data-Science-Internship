"""
nodes_formatter.py — Final response formatter (Task 3 framing guarantee).
"""

from state import AFLGraphState


def formatter_node(state: AFLGraphState) -> dict:
    if state.get("final_response"):
        return {}

    tr = state.get("tool_results") or {}
    tool = state.get("tool_called")

    if tool == "predict_match_winner" and tr.get("status") == "success":
        winner = tr["predicted_winner"]
        prob = tr["win_probability"]
        drivers = tr.get("feature_drivers", [])[:3]
        driver_lines = "\n".join(
            f"  • {d['description']} (impact {d['impact_score']})" for d in drivers
        )
        response = (
            f"Predicted winner: {winner} "
            f"(win probability {prob*100:.1f}%, confidence {tr['confidence_level']}).\n\n"
            f"Top drivers:\n{driver_lines}\n\n"
            f"{tr['disclaimer']}"
        )
        return {"final_response": response}

    if tool == "predict_top_player" and tr.get("status") == "success":
        top = tr["ranked_players"][0]
        drivers = tr.get("feature_drivers", [])[:3]
        driver_lines = "\n".join(f"  • {d['description']}" for d in drivers)
        response = (
            f"Predicted top player for {tr['team']}: {top['player_name']} "
            f"(projected fantasy score {top['predicted_score']}).\n\n"
            f"Grounding:\n{driver_lines}\n\n"
            f"{tr['disclaimer']}"
        )
        return {"final_response": response}

    if tr.get("status") == "success" and tr.get("summary"):
        return {"final_response": tr["summary"]}

    return {
        "final_response": "I processed your request but couldn't generate a response."
    }