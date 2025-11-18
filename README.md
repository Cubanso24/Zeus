# Zeus - Splunk Query LLM

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

Zeus is a fine-tuned Large Language Model (LLM) designed to generate accurate Splunk queries for cybersecurity analysts. The model intelligently generates SPL (Splunk Processing Language) queries from natural language descriptions or asks clarifying questions when the request is ambiguous.

## Features

- **Intelligent Query Generation**: Converts natural language requests into accurate Splunk queries
- **Clarification Requests**: Asks for more information when the request is ambiguous instead of making assumptions
- **Cybersecurity Focus**: Optimized for security use cases including threat detection, incident investigation, and security monitoring
- **Multiple Interfaces**: CLI, API server, web UI, and Python library
- **Admin Dashboard**: Monitor queries, manage user feedback, view system metrics, and trigger model retraining
- **Auto-Scaling**: Docker-based deployment with easy horizontal scaling for handling high loads
- **Fine-tuning Pipeline**: Complete pipeline for training on custom data
- **Evaluation Framework**: Comprehensive metrics for model evaluation

## Quick Start

### Docker Deployment (Recommended)

The easiest way to run Zeus is with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/Zeus.git
cd Zeus

# Start Zeus with 3 API instances
docker compose up -d --scale zeus-api=3

# Create admin user
docker compose exec zeus-api python scripts/create_admin_user.py

# Access the application
# Frontend: http://localhost:8081
# Admin Dashboard: http://localhost:8081/admin.html
```

See [SCALING.md](SCALING.md) for scaling instructions and [ADMIN.md](ADMIN.md) for admin dashboard documentation.

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Zeus.git
cd Zeus

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Using the Pre-trained Model

#### CLI Interface

```bash
# Interactive mode
python -m src.inference.cli \
  --model-path models/splunk-query-llm \
  --interactive

# Single query
python -m src.inference.cli \
  --model-path models/splunk-query-llm \
  --instruction "Find failed SSH login attempts in the last 24 hours"
```

#### Python API

```python
from src.inference.model import SplunkQueryGenerator

# Load model
generator = SplunkQueryGenerator(
    model_path="models/splunk-query-llm"
)

# Generate query
query = generator.generate_query(
    instruction="Show me all failed login attempts",
    input_text="Time range: last 24 hours"
)

print(query)
# Output: index=* (failed OR failure) (login OR authentication) earliest=-24h
```

#### REST API Server

```bash
# Start the server
python -m src.inference.server \
  --model-path models/splunk-query-llm \
  --host 0.0.0.0 \
  --port 8000

# Make a request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Find failed login attempts",
    "input": "Last 24 hours",
    "temperature": 0.1
  }'
```

## Training Your Own Model

### 1. Prepare Training Data

Create JSONL files in `data/raw/` with the following format:

```json
{"instruction": "Find failed SSH login attempts", "input": "", "output": "index=linux sourcetype=linux_secure \"Failed password\" earliest=-24h"}
{"instruction": "Show me suspicious activity", "input": "", "output": "CLARIFICATION: What type of suspicious activity are you looking for?..."}
```

### 2. Process Data

```bash
python -m src.data_preparation.prepare_data \
  --input-dir data/raw \
  --output-dir data/processed \
  --format-type alpaca \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

### 3. Configure Training

Edit `configs/training_config.yaml` to customize:
- Base model selection
- LoRA parameters
- Training hyperparameters
- Data paths

### 4. Fine-tune Model

```bash
python -m src.training.train \
  --config configs/training_config.yaml
```

### 5. Evaluate Model

```bash
python -m src.evaluation.evaluate \
  --model-path models/splunk-query-llm \
  --test-file data/processed/test_alpaca.jsonl \
  --output-dir evaluation_results \
  --save-predictions
```

## Project Structure

```
Zeus/
├── data/
│   ├── raw/              # Raw training data (JSONL)
│   ├── processed/        # Processed training data
│   ├── synthetic/        # Synthetically generated data
│   └── evaluation/       # Evaluation datasets
├── src/
│   ├── data_preparation/ # Data loading, validation, formatting
│   ├── training/         # Fine-tuning scripts
│   ├── inference/        # Model inference (CLI, API, library)
│   ├── evaluation/       # Evaluation metrics
│   └── utils/           # Utilities and configuration
├── models/              # Saved models and checkpoints
├── configs/            # Configuration files
├── scripts/            # Helper scripts
├── examples/           # Example usage
├── tests/              # Unit and integration tests
└── notebooks/          # Jupyter notebooks
```

## Training Data Format

Zeus supports multiple training data formats:

### Alpaca Format

```json
{
  "instruction": "Natural language request",
  "input": "Additional context (optional)",
  "output": "Splunk query or clarification request"
}
```

### Chat Format

```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User request"},
    {"role": "assistant", "content": "Splunk query"}
  ]
}
```

### Clarification Format

Prefix clarification requests with `CLARIFICATION:`:

```json
{
  "instruction": "Ambiguous request",
  "input": "",
  "output": "CLARIFICATION: Please provide more details about..."
}
```

## Model Behavior

Zeus is trained to:

1. **Generate Queries**: When given clear instructions, produce accurate Splunk queries
2. **Ask for Clarification**: When instructions are ambiguous, ask specific questions
3. **Handle Context**: Use additional context provided in the `input` field
4. **Focus on Security**: Prioritize cybersecurity use cases and best practices

### Example Interactions

**Clear Request:**
```
User: "Find failed login attempts in the last hour"
Zeus: index=* (failed OR failure) (login OR authentication) earliest=-1h
```

**Ambiguous Request:**
```
User: "Show me suspicious activity"
Zeus: CLARIFICATION: I can help you find suspicious activity, but I need more specifics. What type of suspicious activity are you looking for? For example:
- Failed login attempts or brute force attacks?
- Unusual network traffic or data exfiltration?
- Malware or process execution?
...
```

## Configuration

### Model Configuration

Configure base model, quantization, and model parameters in `configs/training_config.yaml`:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  model_max_length: 2048
  torch_dtype: "bfloat16"
  load_in_4bit: false
```

### LoRA Configuration

Adjust LoRA parameters for parameter-efficient fine-tuning:

```yaml
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
```

### Training Configuration

Customize training hyperparameters:

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4
```

## Evaluation Metrics

Zeus includes comprehensive evaluation metrics:

- **Exact Match**: Percentage of queries matching exactly
- **Normalized Match**: Match after normalizing whitespace and case
- **Token Overlap**: F1 score of token-level overlap
- **Command Accuracy**: Accuracy of SPL commands used
- **Syntax Validity**: Percentage of syntactically valid queries
- **Clarification Accuracy**: Correct identification of ambiguous requests

## API Reference

### REST API Endpoints

#### POST /generate

Generate a Splunk query from natural language.

**Request:**
```json
{
  "instruction": "string",
  "input": "string (optional)",
  "max_new_tokens": 512,
  "temperature": 0.1,
  "top_p": 0.95,
  "num_return_sequences": 1
}
```

**Response:**
```json
{
  "query": "string",
  "is_clarification": false,
  "clarification_questions": [],
  "alternatives": []
}
```

#### POST /batch_generate

Generate multiple queries in batch.

#### GET /health

Health check endpoint.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Zeus in your research or work, please cite:

```bibtex
@software{zeus_splunk_llm,
  title = {Zeus: Fine-tuned LLM for Splunk Query Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/Zeus}
}
```

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- Inspired by security analyst workflows

## Support

For issues, questions, or contributions:
- Open an issue on [GitHub](https://github.com/yourusername/Zeus/issues)
- Email: your.email@example.com

---

**Note**: This model is designed to assist cybersecurity analysts and should be used in conjunction with human expertise. Always validate generated queries before running them in production environments.