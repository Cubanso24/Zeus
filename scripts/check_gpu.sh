#!/bin/bash

echo "========================================="
echo "GPU Diagnostic Script"
echo "========================================="
echo ""

# Check if containers are running
if ! docker compose ps | grep -q "zeus-api"; then
    echo "❌ Zeus API containers not running"
    exit 1
fi

# Pick first running container
CONTAINER=$(docker compose ps zeus-api --format json 2>/dev/null | head -1 | jq -r '.Name' 2>/dev/null || docker compose ps zeus-api | grep zeus-api | head -1 | awk '{print $1}')

echo "Checking container: $CONTAINER"
echo ""

echo "1. CUDA availability in Python:"
docker exec $CONTAINER python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('  ⚠ CUDA NOT AVAILABLE - will use CPU (very slow!)')
"

echo ""
echo "2. NVIDIA SMI from inside container:"
docker exec $CONTAINER nvidia-smi || echo "  ❌ nvidia-smi not accessible"

echo ""
echo "3. Check if model files exist:"
docker exec $CONTAINER ls -lh models/splunk-query-llm-v2/ | head -15

echo ""
echo "4. Check HuggingFace cache:"
docker exec $CONTAINER ls -lh /app/.cache/huggingface/hub/ 2>/dev/null | head -10 || echo "  Cache directory not found"

echo ""
echo "========================================="
