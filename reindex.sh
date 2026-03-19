#!/usr/bin/env bash
set -euo pipefail

# Load .env
set -a; source "$(dirname "$0")/.env"; set +a

docker run --rm \
  -v "${VAULT_PATH}:/vault" \
  -v "obsidivec_vaultkeeper-data:/data/chromadb" \
  -v "$(dirname "$0")/config.yml:/etc/vaultkeeper/config.yml:ro" \
  -v "$(dirname "$0")/reindex.py:/app/reindex.py:ro" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e VAULTKEEPER_CONFIG=/etc/vaultkeeper/config.yml \
  obsidivec-vaultkeeper:latest \
  python /app/reindex.py
