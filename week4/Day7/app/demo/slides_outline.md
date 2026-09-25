📊 SLIDE DECK (12 Slides)
Slide 1 — Title
text
RealEstate Hub AI Voice Agent — "Ayesha"

Production-grade UrduLish conversational AI
for Pakistani real estate

[Your Name] | Week 4 Capstone | [Date]
Visual: Company logo + voice waveform icon + a phone call illustration

Slide 2 — The Problem
text
Pakistani real estate firms lose millions of rupees yearly to:

❌ Dozens of unanswered inbound calls daily
❌ Expensive, inconsistent human agents (turnover, training, salaries)
❌ Missed leads after hours / weekends / holidays
❌ No standardized objection handling
❌ Zero follow-up on abandoned inquiries
❌ No 24/7 coverage without 3 shifts of staff

Bottom line: Every missed call = a lost customer.
Visual: A frustrated caller icon + a missed call count

Slide 3 — The Solution
text
Meet Ayesha — an AI voice agent that never sleeps.

✅ Speaks fluent UrduLish (natural Urdu + English mix)
✅ Answers property questions grounded in 700 verified listings
✅ Recommends properties by budget, city, bedrooms, purpose
✅ Handles objections with empathy-first rebuttals
✅ Books / reschedules / cancels property visits
✅ Syncs with Google Calendar + sends email to agents
✅ Logs every call to CRM with preferences + follow-up reminders

One agent. 24/7. Infinite scale.
Visual: Ayesha as a friendly cartoon character answering a phone

Slide 4 — Architecture
text
[Insert your Day 1 architecture diagram]

Flow:
Caller → STT (Deepgram) → LLM (Groq) → RAG (SQL+Vector)
       → LangGraph Agent (9 nodes) → Calendar / Email / CRM
       → TTS (Fish Audio) → Caller

Orchestration: LangGraph state machine with conditional routing
Visual: Day 1 architecture diagram from your .docx

Slide 5 — Tech Stack
text
Layer                    Choice                        Why
────────────────────────────────────────────────────────────────
LLM                      Groq (GPT-OSS-120B)          Free, fast, UrduLish-capable
STT                      Deepgram Nova-3              Streaming, Urdu-English
TTS                      Fish Audio                   Natural Urdu, code-switching
Agent Framework          LangGraph                    9-node state machine
Vector DB                ChromaDB                     1401 property chunks
Structured DB            SQLite                       700 properties, exact facts
Backend                  FastAPI                      Auto Swagger docs
Automation               n8n + Python workflow.py     Calendar/Email/CRM
Deployment               Docker + Railway/Render      One-command deploy
Visual: Clean tech stack diagram with logos

Slide 6 — Live Demo (Screenshots)
text
[3 screenshots side by side:]

1. BUYER QUERY              2. OBJECTION HANDLING       3. BOOKING CONFIRMED
"Lahore DHA mein            "Yeh mehnga hai"            "Book kar dein"
 3 bedroom chahiye"         → empathy-first rebuttal     → appointment booked
→ 3 real DHA properties     → alternative option        → ID: a6bb394e
Visual: Actual screenshots from your /docs runs

Slide 7 — RAG Pipeline
text
Dataset:
  • 700 properties across 5 Pakistani cities
  • Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad
  • For Sale + For Rent
  • 1401 vector chunks (400 char, 60 overlap)

Split Retrieval Strategy:
  SQL (exact facts)         →  price, availability, bedrooms, agent
  Vector (fuzzy)            →  amenities, schools, descriptions, FAQs

Evaluation:
  ✅ Retrieval accuracy: 90%
  ✅ Chunk precision@1: 80% (best across 150/400/800 sizes)
  ✅ Optimal chunk size: 400 characters
Visual: Chunk-size comparison table + SQL vs Vector diagram

Slide 8 — Evaluation Results
text
Test Suite (Day 6):
  ✅ 48 / 48 conversations passed (100%)
  ✅ 15 categories: buyer, renter, investor, objection, FAQ, etc.

Security (Day 6):
  ✅ 12 / 12 prompt injection attacks blocked (100%)
  ✅ No system prompt leak, no PII exposure, no unauthorized actions

Performance (Day 6):
  ✅ Latency p95: 412 ms  (target < 2000 ms)
  ✅ RAG accuracy: 100%

API (Day 7):
  ✅ 4 / 4 endpoint tests passed

Human Eval (Day 3):
  ✅ Naturalness: 4.0 / 5
  ✅ Persuasiveness: 4.0 / 5
  ✅ Fluency: 4.0 / 5
Visual: 5 metric cards, big numbers, green checkmarks

Slide 9 — Security
text
Guardrails Implemented:
  ✅ Prompt injection detection + rejection (12 attack patterns tested)
  ✅ PII redaction (client phones, names) before logs
  ✅ Role-override protection ("You are now...")
  ✅ API key protection (secrets never exposed to LLM)
  ✅ Fake booking prevention (SQL-level availability check)
  ✅ No internal pricing / margin disclosure

Rate limiting: planned for production
Audit logs: all calls + node traces persisted
Visual: Shield icon + list of blocked attack types

Slide 10 — Deployment
text
Everything is containerized:

  $ docker compose up --build
  ✓ FastAPI backend        port 8000
  ✓ Health check           /health
  ✓ Auto API docs          /docs
  ✓ Mock-safe integrations Calendar / Email default to mock mode

Deployable to:
  • Railway (free tier, 5 min)
  • Render (free tier, 5 min)
  • AWS / Azure (production scale)

One command. Any cloud.
Visual: Docker whale + deploy buttons for 3 clouds

Slide 11 — Roadmap
text
Next 3 Months:
  🔜 WhatsApp Business API integration
  🔜 SMS confirmations for appointments
  🔜 Salesforce / HubSpot CRM connectors
  🔜 Punjabi language support

Next 6 Months:
  🔜 Live MLS/property feed integration
  🔜 Voice cloning for brand representatives
  🔜 Analytics dashboard + lead scoring
  🔜 Automatic follow-up campaigns

Next 12 Months:
  🔜 Payment gateway integration
  🔜 Multi-tenant SaaS offering
  🔜 Predictive property valuation AI
Visual: Timeline graphic with milestones

Slide 12 — Thank You
text
Shukriya!

Questions?

📧 [your-email]
🔗 github.com/[your-repo]
📱 http://localhost:8000/docs  (demo)

"The agent that never sleeps — built in 7 days."