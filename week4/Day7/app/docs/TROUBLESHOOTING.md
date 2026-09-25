# Troubleshooting

## "No module named 'langgraph'"
```bash
pip install langgraph langchain-core
"No Python at ... python.exe" (venv broken)
Recreate venv:

bash
rm -rf venv
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
Agent says "Sorry, thora clear kar dein"
Intent wasn't recognized. Check that:

City is one of: Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad

Budget is in "crore" or "lakh"

Query contains one of: house, flat, plot, shop, office

Low latency but wrong answers
RAG accuracy issue. Rebuild vector store:

bash
cd day2/src && python build_vector_store.py
Calendar events not created (mock mode)
Set GOOGLE_CALENDAR_CREDENTIALS_JSON in .env and restart.

Emails not sent (mock mode)
Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env.

"InconsistentVersionWarning" from sklearn
Rebuild embedder.pkl with current sklearn version:

bash
cd day2/src && python build_vector_store.py
Docker build fails
Ensure you run from week4/ (parent), not day7/:

bash
cd week4
docker compose -f day7/docker-compose.yml up --build