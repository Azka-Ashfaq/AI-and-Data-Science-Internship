"""
Day 3 - Task 5: Human Evaluation.

Runs every scripted test conversation (data/test_conversations.json) through
the full voice pipeline, logs a complete transcript with per-turn latency,
and writes a blank scoring template for the human evaluation itself --
naturalness/persuasiveness/fluency/conversation-flow genuinely require a
person to read (or, with real TTS wired up later, listen to) the output and
judge it. That judgment can't be automated; latency, by contrast, IS
measured automatically here since it's an objective number.
"""
import json
import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conversation_state import ConversationState
from voice_pipeline import process_turn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def run_all(dry_run=True):
    with open(os.path.join(DATA_DIR, "test_conversations.json")) as f:
        conversations = json.load(f)

    all_timings = []
    transcript_lines = ["# Day 3 Conversation Transcripts\n"]
    eval_rows = []

    for convo in conversations:
        state = ConversationState()
        transcript_lines.append(f"## {convo['title']} (`{convo['id']}`)\n")
        for i, user_text in enumerate(convo["turns"]):
            result = process_turn(state, user_text, i, dry_run=dry_run)
            all_timings.append(result["timings"]["total_ms"])

            transcript_lines.append(f"**Caller:** {result['user_text']}")
            answer_display = result["spoken_text"]
            if "[DRY RUN" in answer_display:
                answer_display = "*(dry-run: no live LLM answer -- context assembly verified only)*"
            transcript_lines.append(f"**Ayesha:** {answer_display}")
            tags = ", ".join(result["behavior_tags"]) if result["behavior_tags"] else "none"
            transcript_lines.append(f"*[behaviors: {tags} | total latency: {result['timings']['total_ms']} ms]*\n")

        transcript_lines.append(f"*Final slots: {state.summary()}*\n\n---\n")
        eval_rows.append({
            "conversation_id": convo["id"], "title": convo["title"],
            "naturalness_1to5": "", "persuasiveness_1to5": "", "fluency_1to5": "",
            "conversation_flow_1to5": "", "notes": ""
        })

    # write transcript
    transcript_path = os.path.join(DATA_DIR, "transcripts.md")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))

    # write latency summary
    avg_latency = sum(all_timings) / len(all_timings)
    print(f"Ran {len(conversations)} conversations, {len(all_timings)} total turns.")
    print(f"Latency (simulated STT + measured retrieval/LLM/logic + simulated TTS):")
    print(f"  avg: {avg_latency:.1f} ms   min: {min(all_timings):.1f} ms   max: {max(all_timings):.1f} ms")
    print(f"  Budget: 2000 ms  -->  {'PASS' if max(all_timings) < 2000 else 'FAIL'} (max turn under budget)")
    print(f"\nTranscript written to {transcript_path}")

    # write human eval scoring template (blank -- fill in by hand after reading/listening)
    eval_path = os.path.join(DATA_DIR, "human_eval_template.csv")
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Human evaluation scoring template written to {eval_path}")
    print("(Score each conversation 1-5 on naturalness, persuasiveness, fluency, "
          "and conversation flow after reading transcripts.md -- or listening, once real TTS is wired up.)")


if __name__ == "__main__":
    dry = not os.environ.get("LLM_PROVIDER")
    run_all(dry_run=dry)
