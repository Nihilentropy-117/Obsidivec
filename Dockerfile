FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Add sse-starlette explicitly for the manual implementation
RUN pip install --no-cache-dir -r requirements.txt sse-starlette

COPY server.py .
COPY start.sh .
COPY search/ ./search/
COPY base_engine.py .
COPY templates.md .

RUN chmod +x start.sh
RUN mkdir /vault
ENV VAULT_PATH="/vault"

EXPOSE 8000

ENTRYPOINT ["./start.sh"]