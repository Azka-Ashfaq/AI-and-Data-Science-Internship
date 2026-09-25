# Maintenance Guide

## Weekly

- Refresh vector store with new property listings (Sunday 2 AM)
  ```bash
  cd day2/src && python build_vector_store.py
Review prompt injection attempts from past week (see day6/data/security_results.json)

Run full test suite: python day6/src/test_suite.py

Monthly
Backup crm.db to S3 (30-day retention)

Rotate API keys (Groq, Deepgram, Fish Audio)

Review latency dashboards — P95 must stay < 2000ms

Update system prompt if new jailbreak patterns emerge

Backups
Asset	Frequency	Retention
crm.db	Daily	30 days
chroma/	Weekly	4 weeks
.env secrets	Quarterly rotation	N/A
Versioning
Follow semver. Current: v1.0.0