```markdown
# Web3Geeks Support-Ticket Triage Agent — Capstone

## What this is
An end-to-end agent system that triages incoming client support tickets for
a freelance web3 dev studio: classifies the ticket, pulls the right answer
from a knowledge base, drafts a reply, routes consequential actions
(refunds, cancellations, billing disputes, contract-bug reports, P1s)
through a human-approval checkpoint, and dispatches the final response.

## Files
- `tools.py` — KB lookup tool (data/kb.json), priority scorer, mock send tool
- `agent_graph.py` — LangGraph state machine (the agent itself)
- `evaluate.py` — runs 10 test cases (7 core + 3 edge/adversarial) and scores them
- `api.py` — FastAPI wrapper with structured logging
- `data/kb.json` — local "database" the KB tool reads from
- `data/eval_results.csv` — output of the last `evaluate.py` run
- `logs/agent.log` — structured JSON logs from API requests
- `make_diagram.py` — generates the architecture diagram
- `architecture_diagram.png` — LangGraph state machine diagram (also in the report)

## Deliverables (submission bundle)
- `Web3Geeks_Agent_Executive_Report.pdf` — 2-page report: business goal →
  architecture → framework-choice rationale → evaluation results →
  known limitations → recommended next steps.
- `Web3Geeks_Agent_Monitoring_Checklist.docx` — production monitoring
  checklist: what to track, alert thresholds, re-evaluation cadence,
  escalation path.
- `Web3Geeks_Agent_Stakeholder_Slides.pptx` — 5–7 minute stakeholder
  presentation outline.

## Run it
```bash
pip install -r requirements.txt

# run the eval suite
python evaluate.py

# start the API
uvicorn api:app --reload --port 8000
```

Example request:
```bash
curl -X POST localhost:8000/triage -H "Content-Type: application/json" \
  -d '{"text": "I need a refund, this is urgent"}'
# -> status: "awaiting_human_approval", plus a ticket_id
```

Resume a paused (consequential) ticket:
```bash
curl -X POST localhost:8000/resume -H "Content-Type: application/json" \
  -d '{"ticket_id": "<ticket_id>", "approved": true}'
```

## Reused from Days 1-4 of this week (not reinvented)
- **Day 2 (LangChain):** the local-JSON-"database" tool pattern
  (`get_product_price` over `products.json`) -> here, `kb_lookup` over
  `data/kb.json`.
- **Day 3 (LangGraph):** the `generate -> critique -> route_critique ->
  revise` self-correction cycle, and the `interrupt_before` +
  `InMemorySaver` checkpointer mechanism for the human-approval gate
  (`graph.invoke()` to run to the interrupt, `graph.update_state(...,
  as_node=...)` to reject, `graph.invoke(None, config)` to resume) — used
  essentially as built in Day 3's `send_email` gate, just applied to a
  support ticket instead of an email.
- **Day 4 (CrewAI):** the role-scoped tool-access principle (each
  specialist only gets the tools its job needs) — carried over as a design
  rule for every node here, even though the framework itself is LangGraph
  rather than CrewAI (see the framework-choice rationale in the report).

## Notes for graders
- `classify_ticket`, `generate_draft`, and `critique_draft` use
  deterministic, structured reasoning functions instead of a live LLM call
  (same approach Day 3 used for `get_text()`), so the whole system runs
  with zero API keys. Each is written so its body can be swapped 1:1 for a
  real model call without touching the graph, routing, or API layer.
- The self-correction loop is not cosmetic: a no-KB-match ticket's first
  draft is a generic fallback, which the critique step scores below the
  80-point threshold; the `revise` node loops back to `generate_draft`,
  which then produces a draft that echoes the client's specific question,
  raising the critique score and ending the loop. This is exercised for
  real in `evaluate.py` (TC7).
- The human-approval checkpoint uses LangGraph's own `interrupt_before` +
  checkpointer, not a hand-rolled pause flag: `run_ticket()` returns
  `status: awaiting_human_approval` when the graph is paused at
  `human_checkpoint`, and `resume_ticket(ticket_id, approved)` resumes or
  rejects it via the same `thread_id`.
``
Author:

Azka-Ashfaq

AI-Data science Intern
