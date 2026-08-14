FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY graph ./graph
COPY ingestion ./ingestion
COPY log_rag ./log_rag
COPY server ./server

EXPOSE 8001
CMD ["uvicorn", "server.api_server:app", "--host", "0.0.0.0", "--port", "8001"]
