FROM python:3.12-slim

# Install ripgrep for text search
RUN apt-get update && \
    apt-get install -y --no-install-recommends ripgrep && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create data directory for ChromaDB
RUN mkdir -p /data/chromadb

# Config location
RUN mkdir -p /etc/vaultkeeper

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "vaultkeeper.server"]
