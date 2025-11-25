# Zeus - Splunk Query LLM

Zeus is a fine-tuned Large Language Model that generates Splunk queries from natural language descriptions. Built for cybersecurity analysts, Zeus can convert questions like "Find failed SSH logins in the last 24 hours" into accurate SPL queries.

## Features

- Natural language to Splunk query generation
- Asks clarifying questions when requests are ambiguous
- Web UI with query history and feedback system
- Admin dashboard for monitoring and model retraining
- REST API for integrations
- GPU-accelerated inference (CUDA support)
- Horizontal scaling with load balancing

## Quick Start

### Prerequisites

For GPU servers (recommended):
- NVIDIA GPU with CUDA support
- Docker and Docker Compose with NVIDIA Container Toolkit
- 16GB+ GPU memory for optimal performance

For CPU-only deployment:
- 16GB+ RAM
- Docker and Docker Compose

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Zeus.git
cd Zeus
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env and set secure passwords and secret keys
```

3. Deploy with Docker Compose:
```bash
# For GPU servers (H100, A100, etc.)
docker compose up -d --build

# To scale to multiple API instances
docker compose up -d --build --scale zeus-api=3
```

4. Access the application:
- Web UI: http://localhost:8081
- API: http://localhost:8081/api (proxied through nginx)
- Admin Dashboard: http://localhost:8081/admin.html

5. Create admin user:
```bash
docker compose exec zeus-api python scripts/create_admin_user.py
```

## GPU Support

Zeus automatically detects and uses NVIDIA GPUs when available. The Docker setup is pre-configured for GPU acceleration using CUDA 11.8.

### Verify GPU Usage

Check if Zeus is using your GPU:
```bash
# Check container logs
docker compose logs zeus-api | grep -i cuda

# Monitor GPU usage
nvidia-smi -l 1
```

If GPU is not detected:
1. Ensure NVIDIA Container Toolkit is installed
2. Verify Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Check CUDA availability in the container: `docker compose exec zeus-api python -c "import torch; print(torch.cuda.is_available())"`

## Usage

### Web Interface

1. Open http://localhost:8081
2. Register a new account or login
3. Enter natural language queries like:
   - "Show me failed login attempts in the last hour"
   - "Find unusual network traffic patterns"
   - "List all events from index=security with error codes"
4. Provide feedback on generated queries to improve the model

### REST API

Generate queries programmatically:

```bash
# Get authentication token
TOKEN=$(curl -X POST http://localhost:8081/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}' \
  | jq -r '.access_token')

# Generate a query
curl -X POST http://localhost:8081/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Find failed SSH login attempts",
    "temperature": 0.1
  }'
```

### Python Client

```python
import requests

# Login
response = requests.post('http://localhost:8081/auth/login', json={
    'username': 'your-username',
    'password': 'your-password'
})
token = response.json()['access_token']

# Generate query
response = requests.post(
    'http://localhost:8081/generate',
    headers={'Authorization': f'Bearer {token}'},
    json={
        'instruction': 'Find all failed login attempts in the last 24 hours',
        'temperature': 0.1
    }
)

query = response.json()['query']
print(f"Generated query: {query}")
```

## Admin Dashboard

Access the admin dashboard at http://localhost:8081/admin.html to:
- View system metrics (queries/day, user activity, GPU usage)
- Monitor feedback and approval rates
- Review all user queries and feedback
- Export feedback data for retraining
- Trigger model retraining jobs
- Manage users and permissions

## Configuration

### Environment Variables

Edit `.env` to configure:

```bash
# Database
POSTGRES_PASSWORD=your_secure_database_password

# JWT Authentication
SECRET_KEY=your_super_secret_jwt_key_change_this
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Model Settings (optional)
MODEL_PATH=models/splunk-query-llm-v2
BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
DEVICE=auto  # auto, cuda, or cpu
```

### Scaling

Scale the number of API instances based on load:

```bash
# Scale to 5 instances
docker compose up -d --scale zeus-api=5

# Scale down to 2 instances
docker compose up -d --scale zeus-api=2
```

NGINX automatically load balances requests across all instances.

### Model Configuration

To use a different base model, edit `configs/training_config.yaml`:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"  # Change this
  model_max_length: 2048
  torch_dtype: "bfloat16"
```

Common models:
- `mistralai/Mistral-7B-Instruct-v0.2` - Fast, efficient (default)
- `meta-llama/Llama-2-7b-chat-hf` - Good general purpose
- `codellama/CodeLlama-7b-Instruct-hf` - Optimized for code/queries

## Training Your Own Model

### 1. Prepare Training Data

Create JSONL files in `data/raw/` with this format:

```json
{"instruction": "Find failed SSH login attempts", "input": "", "output": "index=linux sourcetype=linux_secure \"Failed password\" earliest=-24h"}
{"instruction": "Show me suspicious activity", "input": "", "output": "CLARIFICATION: What type of suspicious activity are you looking for? Failed logins, unusual network traffic, or malware activity?"}
```

### 2. Process Data

```bash
docker compose exec zeus-api python -m src.data_preparation.prepare_data \
  --input-dir data/raw \
  --output-dir data/processed \
  --format-type alpaca
```

### 3. Train Model

```bash
docker compose exec zeus-api python -m src.training.train \
  --config configs/training_config.yaml
```

Training will use all available GPUs and save checkpoints to `models/`.

### 4. Evaluate Model

```bash
docker compose exec zeus-api python -m src.evaluation.evaluate \
  --model-path models/splunk-query-llm \
  --test-file data/processed/test_alpaca.jsonl \
  --output-dir evaluation_results
```

## Health Check

After deployment, verify Zeus is working correctly:

```bash
./scripts/check_health.sh
```

This will check:
- Docker containers are running
- PostgreSQL is healthy
- API servers have loaded the model
- CUDA/GPU is detected

## Troubleshooting

### Login Works But Can't Generate Queries

**Problem**: You can access the login page, but query generation doesn't work or times out.

**Root Cause**: The base model (14GB) failed to download on first startup.

**Solution**:
1. Check if the model is being downloaded:
   ```bash
   docker compose logs zeus-api | grep -i "download\|model"
   ```

2. Manually download the base model:
   ```bash
   docker compose exec zeus-api python scripts/download_base_model.py
   ```

3. If download keeps failing, pre-download on your local machine and copy to server:
   ```bash
   # On a machine with good internet:
   python scripts/download_base_model.py

   # Copy the cache to your server:
   rsync -avz ~/.cache/huggingface/ your-server:~/.cache/huggingface/
   ```

4. Restart Zeus:
   ```bash
   docker compose restart zeus-api
   ```

### GPU Not Detected

**Problem**: Zeus runs on CPU despite having GPU

**Solution**:
1. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Verify Docker GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Rebuild and restart Zeus:
   ```bash
   docker compose down
   docker compose up -d --build
   ```

### Out of Memory Error

**Problem**: GPU runs out of memory during inference

**Solution**:
- Reduce batch size in generation requests
- Scale down the number of API replicas
- Use a smaller model
- Enable model quantization (edit model config to use 4-bit or 8-bit)

### Container Won't Start

**Problem**: `zeus-api` container crashes on startup

**Solution**:
1. Check logs: `docker compose logs zeus-api`
2. Verify database is ready: `docker compose logs postgres`
3. Check disk space: `df -h`
4. Verify model files exist: `ls -la models/`

### Slow Query Generation

**Problem**: Queries take too long to generate

**Solution**:
- Verify GPU is being used: `nvidia-smi`
- Reduce `max_new_tokens` in generation requests
- Scale up more API instances
- Use a smaller/faster model

## Project Structure

```
Zeus/
├── configs/              # Training and model configuration
├── data/
│   ├── raw/             # Training data (JSONL format)
│   ├── processed/       # Processed training data
│   └── feedback/        # User feedback for retraining
├── models/              # Trained models and checkpoints
├── src/
│   ├── data_preparation/  # Data processing
│   ├── training/          # Model training
│   ├── inference/         # API server and CLI
│   ├── evaluation/        # Metrics and evaluation
│   └── database/          # Database models and auth
├── scripts/             # Utility scripts
├── web/                 # Frontend HTML/JS/CSS
├── docker-compose.yml   # Docker orchestration
├── Dockerfile          # Container image definition
└── README.md           # This file
```

## API Reference

### Authentication

All API endpoints except `/auth/register` and `/auth/login` require authentication.

#### POST /auth/register
Register a new user account.

#### POST /auth/login
Login and receive JWT token.

#### GET /auth/me
Get current user information.

### Query Generation

#### POST /generate
Generate a Splunk query from natural language.

**Request:**
```json
{
  "instruction": "Find failed login attempts",
  "input": "",
  "temperature": 0.1,
  "max_new_tokens": 512
}
```

**Response:**
```json
{
  "query": "index=* (failed OR failure) (login OR authentication)",
  "is_clarification": false,
  "clarification_questions": [],
  "alternatives": [],
  "query_id": 123
}
```

#### POST /batch_generate
Generate multiple queries in one request.

#### POST /feedback
Submit feedback on generated query.

### Admin Endpoints (Requires Admin Role)

#### GET /api/admin/analytics
System analytics and metrics.

#### GET /api/admin/queries
View all user queries with filters.

#### GET /api/admin/feedback
View all user feedback with filters.

#### POST /api/admin/feedback/export
Export feedback as training data (JSONL).

#### POST /api/admin/training/start
Trigger model retraining with feedback data.

#### GET /api/admin/training/jobs
List training job history.

## Security

### Production Deployment

Before deploying to production:

1. Change default passwords in `.env`:
   ```bash
   # Generate secure random secrets
   openssl rand -base64 32  # For SECRET_KEY
   openssl rand -base64 24  # For POSTGRES_PASSWORD
   ```

2. Configure CORS in `src/inference/server.py`:
   ```python
   allow_origins=["https://yourdomain.com"]  # Change from ["*"]
   ```

3. Set up HTTPS with proper SSL certificates (use nginx or a reverse proxy)

4. Enable firewall rules to restrict access

5. Regularly update dependencies and base images

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Report issues: [GitHub Issues](https://github.com/yourusername/Zeus/issues)
- Documentation: This README
- Questions: Open a GitHub Discussion

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- Base models from [Mistral AI](https://mistral.ai/)
