#!/bin/bash

echo "========================================="
echo "Stopping Zeus Native Deployment"
echo "========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Stop API server
if [ -f .api.pid ]; then
    API_PID=$(cat .api.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        print_info "Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || kill -9 $API_PID 2>/dev/null
        print_success "API server stopped"
    else
        print_info "API server not running"
    fi
    rm .api.pid
else
    print_info "No API server PID file found"
fi

# Stop web server
if [ -f .web.pid ]; then
    WEB_PID=$(cat .web.pid)
    if ps -p $WEB_PID > /dev/null 2>&1; then
        print_info "Stopping web server (PID: $WEB_PID)..."
        kill $WEB_PID 2>/dev/null || kill -9 $WEB_PID 2>/dev/null
        print_success "Web server stopped"
    else
        print_info "Web server not running"
    fi
    rm .web.pid
else
    print_info "No web server PID file found"
fi

# Also kill any processes on the ports as a backup
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true

echo ""
print_success "Zeus stopped successfully"
echo ""
