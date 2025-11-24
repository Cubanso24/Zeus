#!/bin/bash
set -e

echo "========================================="
echo "Zeus Native Startup (Mac M1 Optimized)"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_DB="zeus_db"
POSTGRES_USER="zeus_user"
POSTGRES_PASSWORD="zeus_password_change_me"
API_PORT=8080
WEB_PORT=8081
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Step 1: Stop Docker containers if running
echo "1. Checking for running Docker containers..."
if docker compose ps 2>/dev/null | grep -q "zeus"; then
    print_info "Stopping Docker containers..."
    docker compose down
    print_success "Docker containers stopped"
else
    print_success "No Docker containers running"
fi
echo ""

# Step 2: Check if PostgreSQL is installed
echo "2. Checking PostgreSQL installation..."
if ! command -v postgres &> /dev/null; then
    print_error "PostgreSQL not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew not found. Please install from https://brew.sh"
        exit 1
    fi
    brew install postgresql@15
    brew services start postgresql@15
    sleep 3
    print_success "PostgreSQL installed and started"
else
    print_success "PostgreSQL found"
    # Make sure it's running
    if ! pg_isready -q 2>/dev/null; then
        print_info "Starting PostgreSQL..."
        brew services start postgresql@15 || brew services start postgresql
        sleep 3
    fi
    print_success "PostgreSQL is running"
fi
echo ""

# Step 3: Create database and user if they don't exist
echo "3. Setting up database..."
if psql -lqt | cut -d \| -f 1 | grep -qw "$POSTGRES_DB"; then
    print_success "Database '$POSTGRES_DB' already exists"
else
    print_info "Creating database and user..."
    createdb "$POSTGRES_DB" 2>/dev/null || true
    psql -d "$POSTGRES_DB" -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';" 2>/dev/null || true
    psql -d "$POSTGRES_DB" -c "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;" 2>/dev/null || true
    psql -d "$POSTGRES_DB" -c "GRANT ALL ON SCHEMA public TO $POSTGRES_USER;" 2>/dev/null || true
    print_success "Database and user created"
fi
echo ""

# Step 4: Activate conda environment
echo "4. Activating Zeus conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    print_error "Conda not found. Please install Miniconda or Anaconda"
    exit 1
fi

conda activate Zeus
print_success "Conda environment 'Zeus' activated"
echo ""

# Step 5: Set environment variables
echo "5. Setting environment variables..."
export DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:5432/$POSTGRES_DB"
export SECRET_KEY="your-secret-key-change-me-in-production"
export ALGORITHM="HS256"
export ACCESS_TOKEN_EXPIRE_MINUTES="10080"
print_success "Environment variables set"
echo ""

# Step 6: Initialize database tables and create admin user
echo "6. Initializing database tables..."
cd "$PROJECT_DIR"
python scripts/init_db.py --create-admin --admin-username admin --admin-password admin123 --admin-email admin@example.com
print_success "Database initialized"
echo ""

# Step 7: Kill any processes using the ports
echo "7. Checking for port conflicts..."
if lsof -ti:$API_PORT &> /dev/null; then
    print_info "Killing process on port $API_PORT..."
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
fi
if lsof -ti:$WEB_PORT &> /dev/null; then
    print_info "Killing process on port $WEB_PORT..."
    lsof -ti:$WEB_PORT | xargs kill -9 2>/dev/null || true
fi
print_success "Ports cleared"
echo ""

# Step 8: Start the API server in the background
echo "8. Starting Zeus API server..."
nohup python -m src.inference.server \
    --model-path models/splunk-query-llm-v2 \
    --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --host 0.0.0.0 \
    --port $API_PORT \
    > logs/api_server.log 2>&1 &
API_PID=$!
echo $API_PID > .api.pid
print_success "API server starting (PID: $API_PID)"
print_info "Log file: logs/api_server.log"
echo ""

# Step 9: Wait for API to be ready
echo "9. Waiting for API server to be ready..."
MAX_WAIT=60
COUNTER=0
while [ $COUNTER -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        print_success "API server is ready!"
        break
    fi
    sleep 1
    COUNTER=$((COUNTER + 1))
    if [ $((COUNTER % 10)) -eq 0 ]; then
        print_info "Still waiting... ($COUNTER seconds)"
    fi
done

if [ $COUNTER -eq $MAX_WAIT ]; then
    print_error "API server did not start within $MAX_WAIT seconds"
    print_info "Check logs/api_server.log for errors"
    exit 1
fi
echo ""

# Step 10: Start the web server in the background
echo "10. Starting web server..."
cd "$PROJECT_DIR/web"
nohup python3 -m http.server $WEB_PORT > ../logs/web_server.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > ../.web.pid
cd "$PROJECT_DIR"
print_success "Web server started (PID: $WEB_PID)"
print_info "Log file: logs/web_server.log"
echo ""

# Final summary
echo "========================================="
echo -e "${GREEN}Zeus is now running natively!${NC}"
echo "========================================="
echo ""
echo "Access Points:"
echo "  - Chat Interface: http://localhost:$WEB_PORT"
echo "  - Admin Dashboard: http://localhost:$WEB_PORT/admin.html"
echo "  - API Health: http://localhost:$API_PORT/health"
echo ""
echo "Login Credentials:"
echo "  - Username: admin"
echo "  - Password: admin123"
echo ""
echo "Performance:"
echo "  - Using M1 Max GPU (Metal)"
echo "  - Expected response time: 15-30 seconds"
echo ""
echo "Process IDs:"
echo "  - API Server: $API_PID"
echo "  - Web Server: $WEB_PID"
echo ""
echo "To stop Zeus:"
echo "  ./stop_native.sh"
echo ""
echo "To view logs:"
echo "  tail -f logs/api_server.log"
echo "  tail -f logs/web_server.log"
echo ""
