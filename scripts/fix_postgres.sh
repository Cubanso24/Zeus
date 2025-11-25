#!/bin/bash

echo "========================================="
echo "PostgreSQL Diagnostic and Fix Script"
echo "========================================="
echo ""

# Check if postgres container exists
echo "1. Checking PostgreSQL container..."
if docker compose ps postgres 2>/dev/null | grep -q "postgres"; then
    echo "   ✓ PostgreSQL container exists"

    # Check if it's running
    if docker compose ps postgres | grep -q "Up"; then
        echo "   ✓ PostgreSQL container is running"
    else
        echo "   ❌ PostgreSQL container is not running"
        echo ""
        echo "   Checking logs..."
        docker compose logs postgres | tail -30
        echo ""
        echo "   Attempting to start PostgreSQL..."
        docker compose up -d postgres
        sleep 5
    fi
else
    echo "   ❌ PostgreSQL container not found"
    echo ""
    echo "   Creating PostgreSQL container..."
    docker compose up -d postgres
    sleep 5
fi

echo ""
echo "2. Checking PostgreSQL logs for errors..."
docker compose logs postgres | tail -20
echo ""

echo "3. Testing PostgreSQL connectivity..."

# Wait for postgres to be ready
echo "   Waiting for PostgreSQL to accept connections..."
for i in {1..30}; do
    if docker compose exec postgres pg_isready -U zeus_user -d zeus_db 2>/dev/null; then
        echo "   ✓ PostgreSQL is accepting connections!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# Test connection from host
echo ""
echo "4. Testing database connection..."
if docker compose exec postgres psql -U zeus_user -d zeus_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "   ✓ Database connection successful"
else
    echo "   ❌ Cannot connect to database"
    echo ""
    echo "   Possible issues:"
    echo "   - Database not initialized"
    echo "   - Wrong credentials"
    echo "   - Volume permission issues"
    echo ""
    echo "   Attempting to fix..."

    # Try to recreate the database
    docker compose down postgres
    docker volume rm zeus_postgres_data 2>/dev/null || true
    docker compose up -d postgres

    echo "   Waiting for PostgreSQL to initialize (30 seconds)..."
    sleep 30
fi

echo ""
echo "5. Checking network connectivity between containers..."
if docker compose run --rm zeus-api nc -zv postgres 5432 2>&1 | grep -q "succeeded"; then
    echo "   ✓ Network connectivity OK"
else
    echo "   ❌ Cannot reach PostgreSQL from zeus-api container"
    echo ""
    echo "   This might be a Docker network issue."
    echo "   Recreating network..."
    docker compose down
    docker network prune -f
    docker compose up -d postgres
    sleep 10
fi

echo ""
echo "========================================="
echo "Diagnostic Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Check if PostgreSQL is healthy:"
echo "     docker compose exec postgres pg_isready -U zeus_user -d zeus_db"
echo ""
echo "  2. Restart zeus-api containers:"
echo "     docker compose restart zeus-api"
echo ""
echo "  3. Monitor startup:"
echo "     docker compose logs -f zeus-api"
