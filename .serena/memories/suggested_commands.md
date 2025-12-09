# Zeus Development Commands

## Docker Commands

### Start Services
```bash
# Start all services (GPU)
docker compose up -d --build

# Scale API instances
docker compose up -d --build --scale zeus-api=3

# Stop services
docker compose down
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f zeus-api
docker compose logs -f postgres

# Check GPU detection
docker compose logs zeus-api | grep -i cuda
```

### Container Execution
```bash
# Run Python commands in container
docker compose exec zeus-api python <script>

# Create admin user
docker compose exec zeus-api python scripts/create_admin_user.py

# Download base model
docker compose exec zeus-api python scripts/download_base_model.py
```

## Development Commands

### Testing
```bash
pytest                                  # Run all tests
pytest tests/ -v                       # Verbose
pytest --cov=src --cov-report=term-missing  # With coverage
```

### Linting & Formatting
```bash
black src/ --line-length 100           # Format code
flake8 src/                            # Lint code
mypy src/                              # Type checking
```

### Training
```bash
# Process training data
python -m src.data_preparation.prepare_data \
  --input-dir data/raw \
  --output-dir data/processed \
  --format-type alpaca

# Train model
python -m src.training.train --config configs/training_config.yaml

# Evaluate model
python -m src.evaluation.evaluate \
  --model-path models/splunk-query-llm \
  --test-file data/processed/test_alpaca.jsonl \
  --output-dir evaluation_results
```

### CLI Entry Points
```bash
zeus-train   # Training (src.training.train:main)
zeus-serve   # Server (src.inference.server:main)
zeus-eval    # Evaluation (src.evaluation.evaluate:main)
```

## Git Commands
```bash
git status
git add .
git commit -m "message"
git push
```

## Health Check
```bash
./scripts/check_health.sh
```

## System Utilities
Standard Linux utilities: `ls`, `cd`, `grep`, `find`, `cat`, `tail`, etc.
