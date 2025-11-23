#!/bin/bash
set -e

# Directory for certificates
CERT_DIR="/app/certs"
mkdir -p "$CERT_DIR"

KEY_FILE="$CERT_DIR/key.pem"
CERT_FILE="$CERT_DIR/cert.pem"

# Generate self-signed certs if they don't exist
if [ ! -f "$KEY_FILE" ] || [ ! -f "$CERT_FILE" ]; then
    echo "Generating self-signed certificates for local debugging..."
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -days 365 -nodes \
        -subj '/CN=localhost'
fi

echo "Starting Secure MCP Server on port 8000..."
# We use 'exec' so uvicorn receives signals (like Ctrl+C) correctly
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE"