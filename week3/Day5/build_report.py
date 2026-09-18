"""
build_report.py — generates the 2-page executive report PDF (Task 5).
Run: python build_report.py   ->  writes AFL_Assistant_Executive_Report.pdf
"""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=4,
                           textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Sub", fontSize=10.5, leading=13,
                           textColor=colors.HexColor("#555555"), spaceAfter=14))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6,
                           textColor=colors.HexColor("#16213e"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=9.7, leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle(name="BodySmall", fontSize=8.8, leading=12, spaceAfter=4,
                           textColor=colors.HexColor("#333333")))

story = []

story.append(Paragraph("AFL Assistant — Executive Report", styles["H1"]))
story.append(Paragraph(
    "Domain-locked AFL chat + prediction assistant &nbsp;|&nbsp; Capstone Week 3, Day 5 "
    "&nbsp;|&nbsp; System hardening, evaluation, API/UI deployment, and monitoring plan",
    styles["Sub"],
))

# --- Product goal ---
story.append(Paragraph("Product Goal", styles["H2"]))
story.append(Paragraph(
    "Ship a production-ready, AFL-only chat assistant that answers factual questions, "
    "retrieves historical player/team statistics, and generates probabilistic match-winner "
    "and top-player predictions — deployable behind an API and a demoable UI, as if going "
    "live on a client property such as Web3Geeks.",
    styles["Body"],
))

# --- Architecture ---
story.append(Paragraph("Architecture", styles["H2"]))
story.append(Paragraph(
    "A LangGraph state machine routes each query through a rule-based intent classifier "
    "(<b>router_node</b>) to one of four branches — <b>prediction</b> (wraps two trained "
    "scikit-learn models: match-winner classifier and top-player regressor), "
    "<b>retrieval</b> (structured pandas lookups over historical match/player data), "
    "<b>factual</b> (canned AFL-rules answers), or <b>refusal</b> (off-topic/off-scope). "
    "Prediction and retrieval outputs pass through a <b>validation</b> node that routes to "
    "a formatter, a clarification prompt, or a fallback message, then to the final response. "
    "Day 5 wraps this graph in a FastAPI <b>/chat</b> endpoint with per-conversation session "
    "memory, structured logging, and basic rate limiting, plus an optional Streamlit UI for demos.",
    styles["Body"],
))

# --- Evaluation results ---
story.append(Paragraph("Evaluation Results", styles["H2"]))
story.append(Paragraph(
    "A combined 31-case suite (<b>eval_suite.py</b>) spans four categories. 18 cases have no "
    "dependency on the local data/model files and were executed and verified in this review; "
    "13 retrieval/prediction/multi-turn cases require the project's afl_datasets/ and "
    "model_store/ directories and are wired to run automatically (<code>python eval_suite.py</code>) "
    "the first time they're run from the project folder — see the Known Limitations section.",
    styles["Body"],
))

cell = ParagraphStyle(name="Cell", fontSize=8.3, leading=10.5)
head = ParagraphStyle(name="Head", fontSize=8.3, leading=10.5, textColor=colors.white, fontName="Helvetica-Bold")


def P(text, style=cell):
    return Paragraph(text, style)


table_data = [
    [P("Category", head), P("Verified now", head), P("Pass rate", head), P("Pending local run", head)],
    [P("Factual Q&amp;A"), P("5 / 5"), P("100%"), P("—")],
    [P("Scope guardrails (incl. 4 prompt-injection attempts)"), P("10 / 10"), P("100%"), P("—")],
    [P("Retrieval"), P("0 / 6"), P("n/a"), P("6 (needs afl_datasets/)")],
    [P("Prediction sanity"), P("3 / 6"), P("100%"), P("3 (needs model_store/)")],
    [P("Multi-turn coherence"), P("0 / 4"), P("n/a"), P("4 (needs both)")],
]
t = Table(table_data, colWidths=[2.7 * inch, 1.0 * inch, 0.75 * inch, 1.45 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f7")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 8))

story.append(Paragraph(
    "<b>Prompt-injection resistance:</b> four adversarial prompts were tested against the router "
    "(instruction override, persona override, fake system message, prompt-exfiltration attempt) — "
    "all four were correctly refused, including two that were misclassified by the router's intent "
    "label before a Day 5 fix (see Known Limitations).",
    styles["BodySmall"],
))
story.append(Paragraph(
    "<b>Weakest category:</b> conversational coherence. The graph state accepts "
    "<code>conversation_history</code> but, as of Day 4, no node actually reads it — every turn is "
    "classified independently. <b>Concrete improvement (implemented for the common case):</b> the "
    "FastAPI/Streamlit session layer now caches the last resolved team/player and retries a "
    "clarification-triggering follow-up once with that entity appended as context (e.g. \"why do "
    "you think that\" after a prediction). This covers the most common follow-up pattern without "
    "touching the graph's routing logic; general coreference resolution inside the graph itself "
    "remains a next step.",
    styles["BodySmall"],
))
story.append(Paragraph(
    "<b>Naive-baseline comparison:</b> <code>baseline_comparison.py</code> compares the trained "
    "match-winner model's accuracy and Brier score against a naive \"higher pre-match ladder points "
    "wins\" baseline on the same holdout split. This must be run once against the project's actual "
    "Day 1/2 holdout file and model artifacts to produce real numbers — not fabricated here since "
    "those files weren't part of this review's inputs.",
    styles["BodySmall"],
))

story.append(Paragraph("Known Limitations", styles["H2"]))
limitations = [
    "Data recency — predictions and stats are only as current as the latest feature-snapshot refresh; "
    "see the weekly retrain loop in the monitoring checklist.",
    "Model accuracy ceiling — match outcomes have irreducible variance (injuries, selection, in-game "
    "swings); the model's real ceiling can only be quantified once baseline_comparison.py is run "
    "against the actual holdout set.",
    "Guardrail edge cases — the router is regex/keyword-based, not an LLM classifier, so novel phrasing "
    "outside the tested pattern lists can still be misrouted; router_test.py should be re-run after any "
    "pattern-list change.",
    "No general multi-turn memory — the shipped fix handles the single most common follow-up shape "
    "(missing team/player); pronoun resolution and multi-entity follow-ups are not yet covered.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(li, styles["BodySmall"]), leftIndent=6) for li in limitations],
    bulletType="bullet", start="•",
))

story.append(Paragraph("Recommended Next Steps", styles["H2"]))
next_steps = [
    "Run eval_suite.py and baseline_comparison.py once from the project's actual folder (with "
    "afl_datasets/ and model_store/ present) to replace the 13 pending cases with real pass/fail "
    "numbers and a real accuracy-lift figure over the naive baseline.",
    "Build general entity coreference into the graph itself (not just the session-layer patch) so "
    "pronoun and topic-shift follow-ups resolve correctly.",
    "Wire the structured logs from api.py into a real dashboard/alerting stack per the monitoring "
    "checklist before any public traffic.",
    "Schedule the first monthly retrain and the weekly snapshot refresh as calendar jobs, not manual steps.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(li, styles["BodySmall"]), leftIndent=6) for li in next_steps],
    bulletType="bullet", start="•",
))

doc = SimpleDocTemplate(
    "AFL_Assistant_Executive_Report.pdf", pagesize=LETTER,
    topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    title="AFL Assistant Executive Report",
)
doc.build(story)
print("Wrote AFL_Assistant_Executive_Report.pdf")
