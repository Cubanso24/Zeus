#!/bin/bash
set -e

echo "========================================="
echo "Zeus Deployment Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Use docker compose or docker-compose based on what's available
if command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}Docker and Docker Compose found!${NC}"
echo ""

# Check for GPU support
echo "Checking for NVIDIA GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected!${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    GPU_AVAILABLE=true

    # Check if nvidia-container-toolkit is installed
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA Container Toolkit not properly configured.${NC}"
        echo "Zeus will work but won't use GPU acceleration."
        echo ""
        echo "To enable GPU support, install NVIDIA Container Toolkit:"
        echo "  Ubuntu/Debian:"
        echo "    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "    curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "    sudo systemctl restart docker"
        echo ""
        GPU_AVAILABLE=false
    fi
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Zeus will run on CPU (slower).${NC}"
    echo ""
    GPU_AVAILABLE=false
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}.env file not found. Creating from template...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env

        # Generate secure random secrets
        if command -v openssl &> /dev/null; then
            SECRET_KEY=$(openssl rand -base64 32)
            POSTGRES_PASSWORD=$(openssl rand -base64 24)

            # Update .env with generated secrets (cross-platform sed)
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s|your_super_secret_jwt_key_change_this_in_production|$SECRET_KEY|g" .env
                sed -i '' "s|your_secure_database_password_here|$POSTGRES_PASSWORD|g" .env
            else
                sed -i "s|your_super_secret_jwt_key_change_this_in_production|$SECRET_KEY|g" .env
                sed -i "s|your_secure_database_password_here|$POSTGRES_PASSWORD|g" .env
            fi

            echo -e "${GREEN}Generated secure random secrets in .env${NC}"
        else
            echo -e "${YELLOW}Please edit .env and set secure SECRET_KEY and POSTGRES_PASSWORD${NC}"
        fi
    else
        echo -e "${RED}Error: .env.example not found${NC}"
        exit 1
    fi
    echo ""
fi

# Ask for number of API instances
echo "How many API instances do you want to run?"
echo "  - More instances = higher throughput (recommended: 2-5)"
echo "  - Each instance uses ~4-8GB GPU memory or ~8-16GB RAM"
read -p "Number of instances [default: 3]: " NUM_INSTANCES
NUM_INSTANCES=${NUM_INSTANCES:-3}

if ! [[ "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [ "$NUM_INSTANCES" -lt 1 ]; then
    echo -e "${RED}Invalid number. Using default: 3${NC}"
    NUM_INSTANCES=3
fi

echo ""
echo "========================================="
echo "Deployment Configuration"
echo "========================================="
echo "GPU Support: $([ "$GPU_AVAILABLE" = true ] && echo "Yes" || echo "No")"
echo "API Instances: $NUM_INSTANCES"
echo "Web UI: http://localhost:8081"
echo "Admin Dashboard: http://localhost:8081/admin.html"
echo "========================================="
echo ""

read -p "Press Enter to start deployment, or Ctrl+C to cancel..."

echo ""
echo "Building and starting Zeus..."
echo ""

# Stop any running containers
$DOCKER_COMPOSE down 2>/dev/null || true

# Build and start services
if $DOCKER_COMPOSE up -d --build --scale zeus-api=$NUM_INSTANCES; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Zeus deployed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""

    # Wait for services to be healthy
    echo "Waiting for services to start (this may take 1-2 minutes)..."
    sleep 10

    # Check if containers are running
    if $DOCKER_COMPOSE ps | grep -q "Up"; then
        echo ""
        echo -e "${GREEN}Services are running!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Create an admin user:"
        echo "     $DOCKER_COMPOSE exec zeus-api python scripts/create_admin_user.py"
        echo ""
        echo "  2. Access the application:"
        echo "     - Web UI: http://localhost:8081"
        echo "     - Admin Dashboard: http://localhost:8081/admin.html"
        echo ""
        echo "  3. View logs:"
        echo "     $DOCKER_COMPOSE logs -f zeus-api"
        echo ""
        echo "  4. Monitor GPU usage (if available):"
        echo "     nvidia-smi -l 1"
        echo ""
        echo "To stop Zeus:"
        echo "  $DOCKER_COMPOSE down"
        echo ""
        echo "For more information, see README.md"
        echo ""
    else
        echo -e "${RED}Warning: Some services may not have started correctly.${NC}"
        echo "Check logs with: $DOCKER_COMPOSE logs"
    fi
else
    echo ""
    echo -e "${RED}Deployment failed!${NC}"
    echo "Check logs with: $DOCKER_COMPOSE logs"
    exit 1
fi
