# Day 6 Task 4 — Monitoring & Alerting Plan

## Metrics Tracked

| Metric | Source | Threshold | Alert |
|---|---|---|---|
| **Average turn latency** | Prometheus + FastAPI middleware | > 2500 ms | Slack |
| **P95 turn latency** | Same | > 3500 ms | PagerDuty |
| **Voice quality (MOS)** | Fish Audio post-call rating | < 3.5 | Slack |
| **API failure rate (Groq)** | Response codes | > 2% | Slack |
| **Calendar API failure rate** | Google Calendar | > 5% | Slack |
| **Email send failure rate** | SMTP | > 5% | Slack |
| **Booking success rate** | crm.db daily rollup | < 85% | PagerDuty |
| **RAG miss rate** | "no_match" outcomes | > 30% | Slack |
| **Prompt injection attempts** | Security log | Any successful leak | PagerDuty |

## Dashboards

- **Latency dashboard**: rolling 1h avg + p95 of STT / LLM / TTS / total
- **Booking funnel**: calls → intents → recommendations → bookings
- **Security dashboard**: injection attempts, leaked phrases, blocked IPs
- **RAG health**: retrieval accuracy over last 1000 queries

## Logging

- **Structured JSON logs** for every node transition (`execution_traces.json` schema)
- **PII redaction** on client phone numbers before log persistence
- **Retention**: 90 days for call logs, 1 year for aggregated metrics

## Weekly Maintenance

- Retrain/reindex the vector DB with new property listings (Sunday 2 AM)
- Review prompt injection attempts from the past week
- Update system prompt if new jailbreak patterns emerge
- Run full `test_suite.py` — success rate must stay ≥ 95%
- Review RAG accuracy — must stay ≥ 85%

## Backup Strategy

- `crm.db` → daily snapshot to S3 (30-day retention)
- `chroma/` → weekly snapshot (4-week retention)
- `.env` secrets → 1Password vault (rotated quarterly)