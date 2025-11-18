#!/bin/bash
# Zeus Deployment Script

set -e

echo "=================================="
echo "Zeus - Splunk Query LLM Deployment"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if model exists
if [ ! -d "models/splunk-query-llm" ]; then
    echo -e "${RED}Error: Model not found at models/splunk-query-llm${NC}"
    echo "Please train your model first or copy it to the models/ directory"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose found${NC}"
echo -e "${GREEN}✓ Model found${NC}"
echo ""

# Parse arguments
ACTION=${1:-"up"}

case $ACTION in
    "build")
        echo "Building Docker images..."
        docker-compose build
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;

    "up"|"start")
        echo "Starting Zeus services..."
        docker-compose up -d
        echo ""
        echo -e "${GREEN}✓ Zeus is starting!${NC}"
        echo ""
        echo "Waiting for services to be ready..."
        sleep 5

        # Wait for API to be healthy
        MAX_RETRIES=30
        RETRY=0
        while [ $RETRY -lt $MAX_RETRIES ]; do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
                echo -e "${GREEN}✓ API Server is ready!${NC}"
                break
            fi
            RETRY=$((RETRY+1))
            echo -n "."
            sleep 2
        done

        if [ $RETRY -eq $MAX_RETRIES ]; then
            echo -e "${RED}Warning: API server may not be ready yet${NC}"
            echo "Check logs with: docker-compose logs -f zeus-api"
        fi

        echo ""
        echo "=================================="
        echo -e "${GREEN}Zeus is running!${NC}"
        echo "=================================="
        echo ""
        echo "Access Zeus:"
        echo "  🌐 Web Interface: http://localhost:3000"
        echo "  📡 API Docs:      http://localhost:8000/docs"
        echo "  ❤️  Health Check:  http://localhost:8000/health"
        echo ""
        echo "From other devices on your network:"
        LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}')
        if [ -n "$LOCAL_IP" ]; then
            echo "  🌐 Web Interface: http://${LOCAL_IP}:3000"
            echo "  📡 API Docs:      http://${LOCAL_IP}:8000/docs"
        fi
        echo ""
        echo "To view logs:"
        echo "  docker-compose logs -f"
        echo ""
        echo "To stop Zeus:"
        echo "  ./deploy.sh stop"
        echo "=================================="
        ;;

    "down"|"stop")
        echo "Stopping Zeus services..."
        docker-compose down
        echo -e "${GREEN}✓ Zeus stopped${NC}"
        ;;

    "restart")
        echo "Restarting Zeus services..."
        docker-compose restart
        echo -e "${GREEN}✓ Zeus restarted${NC}"
        ;;

    "logs")
        docker-compose logs -f
        ;;

    "status")
        echo "Zeus Service Status:"
        echo ""
        docker-compose ps
        echo ""
        echo "Resource Usage:"
        docker stats --no-stream $(docker-compose ps -q)
        ;;

    "update")
        echo "Updating Zeus..."
        echo "1. Pulling latest changes..."
        git pull
        echo "2. Rebuilding containers..."
        docker-compose build
        echo "3. Restarting services..."
        docker-compose up -d
        echo -e "${GREEN}✓ Zeus updated${NC}"
        ;;

    *)
        echo "Zeus Deployment Script"
        echo ""
        echo "Usage: ./deploy.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build          - Build Docker images"
        echo "  up|start       - Start Zeus services (default)"
        echo "  down|stop      - Stop Zeus services"
        echo "  restart        - Restart Zeus services"
        echo "  logs           - View service logs"
        echo "  status         - Show service status and resource usage"
        echo "  update         - Update Zeus and restart services"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh build    # Build images"
        echo "  ./deploy.sh start    # Start services"
        echo "  ./deploy.sh logs     # View logs"
        echo "  ./deploy.sh stop     # Stop services"
        ;;
esac
