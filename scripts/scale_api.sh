#!/bin/bash

# Zeus API Scaling Script
# Easy management of API instances via Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to project root
cd "$PROJECT_ROOT"

# Show usage
usage() {
    echo "Zeus API Scaling Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  scale <count>     - Scale API instances to <count> replicas"
    echo "  up [count]        - Start all services with optional replica count (default: 3)"
    echo "  down              - Stop all services"
    echo "  restart [count]   - Restart services with optional replica count"
    echo "  status            - Show status of all containers"
    echo "  logs [service]    - Show logs (default: zeus-api)"
    echo ""
    echo "Examples:"
    echo "  $0 up 5           - Start Zeus with 5 API instances"
    echo "  $0 scale 8        - Scale running API to 8 instances"
    echo "  $0 status         - Show all running containers"
    echo "  $0 logs           - Show API logs"
    echo ""
}

# Scale API instances
scale_api() {
    local count=$1

    if [ -z "$count" ]; then
        echo -e "${RED}Error: Please specify number of instances${NC}"
        usage
        exit 1
    fi

    echo -e "${BLUE}Scaling Zeus API to $count instances...${NC}"
    docker compose up -d --scale zeus-api=$count --no-recreate

    echo ""
    echo -e "${GREEN}✓ API scaled to $count instances${NC}"
    echo ""
    show_status
}

# Start all services
start_services() {
    local count=${1:-3}

    echo -e "${BLUE}Starting Zeus services with $count API instances...${NC}"
    docker compose up -d --scale zeus-api=$count

    echo ""
    echo -e "${GREEN}✓ Zeus services started${NC}"
    echo ""
    show_status
}

# Stop all services
stop_services() {
    echo -e "${BLUE}Stopping Zeus services...${NC}"
    docker compose down
    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Restart services
restart_services() {
    local count=${1:-3}

    echo -e "${BLUE}Restarting Zeus services with $count API instances...${NC}"
    docker compose down
    sleep 2
    docker compose up -d --scale zeus-api=$count

    echo ""
    echo -e "${GREEN}✓ Services restarted${NC}"
    echo ""
    show_status
}

# Show status
show_status() {
    echo -e "${BLUE}Zeus Services Status:${NC}"
    echo "========================================="
    docker compose ps
    echo ""

    # Count running API instances
    local api_count=$(docker compose ps zeus-api --format json 2>/dev/null | jq -s 'length' 2>/dev/null || echo "0")
    echo -e "API Instances: ${GREEN}${api_count}${NC}"

    # Show nginx status
    local nginx_status=$(docker compose ps nginx --format '{{.State}}' 2>/dev/null || echo "stopped")
    echo -e "Load Balancer: ${GREEN}${nginx_status}${NC}"

    # Show database status
    local db_status=$(docker compose ps postgres --format '{{.State}}' 2>/dev/null || echo "stopped")
    echo -e "Database: ${GREEN}${db_status}${NC}"

    echo ""
    echo -e "${BLUE}Access Points:${NC}"
    echo "  Frontend:      http://localhost:8081"
    echo "  API (via LB):  http://localhost:8081/api/"
    echo "  Admin Panel:   http://localhost:8081/admin.html"
}

# Show logs
show_logs() {
    local service=${1:-zeus-api}
    echo -e "${BLUE}Showing logs for: $service${NC}"
    echo "Press Ctrl+C to exit"
    echo ""
    docker compose logs -f "$service"
}

# Main script logic
case "${1:-}" in
    scale)
        scale_api "$2"
        ;;
    up|start)
        start_services "$2"
        ;;
    down|stop)
        stop_services
        ;;
    restart)
        restart_services "$2"
        ;;
    status|ps)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    *)
        usage
        exit 1
        ;;
esac
