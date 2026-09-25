FROM python:3.13-slim

WORKDIR /app

# system deps for chromadb / sqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the three project layers
COPY app/ ./app/
COPY ../day2/ ./day2/
COPY ../day4/ ./day4/
COPY ../day5/ ./day5/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]