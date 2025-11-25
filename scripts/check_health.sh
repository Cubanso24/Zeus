#!/bin/bash

echo "========================================="
echo "Zeus Health Check"
echo "========================================="
echo ""

# Check if Docker Compose is running
if ! docker compose ps | grep -q "Up"; then
    echo "❌ Zeus is not running"
    echo "   Start it with: docker compose up -d"
    exit 1
fi

echo "✓ Docker containers are running"
echo ""

# Check PostgreSQL
echo "Checking PostgreSQL..."
if docker compose exec postgres pg_isready -U zeus_user -d zeus_db > /dev/null 2>&1; then
    echo "✓ PostgreSQL is healthy"
else
    echo "❌ PostgreSQL is not ready"
    docker compose logs postgres | tail -20
    exit 1
fi

echo ""
echo "Checking API servers..."

# Get list of API containers
API_CONTAINERS=$(docker compose ps zeus-api --format json 2>/dev/null | jq -r '.Name' 2>/dev/null || docker compose ps zeus-api | grep zeus-api | awk '{print $1}')

if [ -z "$API_CONTAINERS" ]; then
    echo "❌ No API containers found"
    exit 1
fi

# Check each API container
HEALTHY_COUNT=0
TOTAL_COUNT=0

for container in $API_CONTAINERS; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "  Checking $container..."

    # Check if model is loaded
    if docker exec $container python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>/dev/null | grep -q "CUDA"; then
        CUDA_STATUS=$(docker exec $container python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null)
        echo "    CUDA Available: $CUDA_STATUS"

        # Check health endpoint
        HEALTH=$(docker exec $container curl -s http://localhost:8080/health 2>/dev/null)
        if echo "$HEALTH" | grep -q "model_loaded.*true"; then
            echo "    ✓ Model loaded and healthy"
            HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
        else
            echo "    ⚠ Model not loaded"
            echo "    Response: $HEALTH"
        fi
    else
        echo "    ⚠ Container check failed"
        echo "    Last 10 log lines:"
        docker logs $container 2>&1 | tail -10 | sed 's/^/      /'
    fi
    echo ""
done

echo "========================================="
echo "Summary: $HEALTHY_COUNT/$TOTAL_COUNT API instances healthy"
echo "========================================="
echo ""

if [ $HEALTHY_COUNT -eq 0 ]; then
    echo "❌ No healthy API instances!"
    echo ""
    echo "Common issues:"
    echo "  1. Base model download failed (check logs)"
    echo "  2. Out of memory (reduce replicas)"
    echo "  3. GPU not detected (check nvidia-smi)"
    echo ""
    echo "Check logs with:"
    echo "  docker compose logs zeus-api"
    echo ""
    exit 1
elif [ $HEALTHY_COUNT -lt $TOTAL_COUNT ]; then
    echo "⚠ Some instances are unhealthy"
    echo "  This may indicate resource constraints"
    exit 1
else
    echo "✓ All systems operational!"
    echo ""
    echo "Access Zeus:"
    echo "  Web UI: http://localhost:8081"
    echo "  Admin: http://localhost:8081/admin.html"
    exit 0
fi
