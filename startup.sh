#!/bin/bash
set -e

echo "========================================="
echo "Zeus Startup - Database Initialization"
echo "========================================="

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
max_retries=30
retry_count=0

while ! python -c "from src.database.database import engine; engine.connect()" 2>/dev/null; do
    retry_count=$((retry_count + 1))
    if [ $retry_count -ge $max_retries ]; then
        echo "ERROR: PostgreSQL did not become ready in time"
        exit 1
    fi
    echo "Waiting for PostgreSQL... (attempt $retry_count/$max_retries)"
    sleep 2
done

echo "✓ PostgreSQL is ready!"

# Initialize database tables
echo ""
echo "Initializing database tables..."
python scripts/init_db.py --create-admin

if [ $? -eq 0 ]; then
    echo "✓ Database initialization complete!"
else
    echo "⚠ Database initialization encountered errors (this is normal if tables already exist)"
fi

# Start the API server
echo ""
echo "========================================="
echo "Starting Zeus API Server"
echo "========================================="
exec python -m src.inference.server \
    --model-path "${MODEL_PATH:-models/splunk-query-llm-v2}" \
    --base-model "${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8080}" \
    --device "${DEVICE:-auto}"
