# Zeus - Quick Start Guide

This guide will help you get started with Zeus Splunk Query LLM in minutes.

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 20GB+ free disk space

## Installation

### Option 1: Automated Setup

```bash
# Run the quick start script
./scripts/quick_start.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Using Zeus (Without Training)

If you want to use a pre-trained model or skip training for now:

### 1. Interactive CLI

```bash
python -m src.inference.cli \
  --model-path models/splunk-query-llm \
  --interactive
```

Then enter your queries:
```
Query Request: Find failed login attempts in the last 24 hours
```

### 2. Python Script

```python
from src.inference.model import SplunkQueryGenerator

generator = SplunkQueryGenerator(model_path="models/splunk-query-llm")
query = generator.generate_query("Find failed SSH logins")
print(query)
```

### 3. REST API

Start the server:
```bash
python -m src.inference.server \
  --model-path models/splunk-query-llm \
  --port 8000
```

Make requests:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Find failed logins",
    "temperature": 0.1
  }'
```

## Training Your Own Model

### Step 1: Prepare Training Data

Your training data should be in JSONL format in `data/raw/`:

```json
{"instruction": "Find failed SSH logins", "input": "", "output": "index=linux sourcetype=linux_secure \"Failed password\""}
```

We've provided example data in:
- `data/raw/train_basic.jsonl` - Basic queries
- `data/raw/train_security.jsonl` - Security queries
- `data/raw/train_aggregation.jsonl` - Statistical queries
- `data/raw/train_advanced.jsonl` - Advanced queries
- `data/raw/train_clarification.jsonl` - Clarification examples

### Step 2: Process and Validate Data

```bash
python -m src.data_preparation.prepare_data \
  --input-dir data/raw \
  --output-dir data/processed \
  --format-type alpaca
```

This will:
- Validate all training examples
- Split data into train/validation/test sets
- Format data for training
- Generate statistics

### Step 3: Configure Training

Edit `configs/training_config.yaml`:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
```

### Step 4: Fine-tune Model

```bash
python -m src.training.train \
  --config configs/training_config.yaml
```

Training will:
- Download base model
- Apply LoRA for efficient fine-tuning
- Train on your data
- Save checkpoints to `models/splunk-query-llm/`

**Time estimate**: 1-4 hours depending on data size and hardware

### Step 5: Evaluate

```bash
python -m src.evaluation.evaluate \
  --model-path models/splunk-query-llm \
  --test-file data/processed/test_alpaca.jsonl \
  --output-dir evaluation_results \
  --save-predictions
```

## Common Tasks

### Add More Training Data

1. Create new JSONL file in `data/raw/`
2. Add your examples:
   ```json
   {"instruction": "Your request", "input": "Context", "output": "Splunk query"}
   ```
3. Re-run data preparation
4. Re-train model

### Test the Model

Run the example scripts:

```bash
# Basic usage
python examples/basic_usage.py

# Batch processing
python examples/batch_processing.py

# API client
python examples/api_client_example.py
```

### Change Base Model

Edit `configs/training_config.yaml`:

```yaml
model:
  base_model: "codellama/CodeLlama-7b-Instruct-hf"  # Use CodeLlama instead
```

Popular options:
- `mistralai/Mistral-7B-Instruct-v0.2` - Fast and efficient
- `meta-llama/Llama-2-7b-chat-hf` - Good general purpose
- `codellama/CodeLlama-7b-Instruct-hf` - Good for code/queries

### Reduce Memory Usage

Enable quantization in config:

```yaml
model:
  load_in_4bit: true  # Use 4-bit quantization
```

Or reduce batch size:

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

## Troubleshooting

### Out of Memory Error

- Enable 4-bit quantization
- Reduce batch size
- Reduce max sequence length
- Use gradient checkpointing (enabled by default)

### Model Not Generating Good Queries

- Add more training examples
- Increase training epochs
- Adjust learning rate
- Review data quality with validation

### Slow Training

- Use mixed precision (bf16 enabled by default)
- Increase batch size if you have memory
- Reduce max sequence length
- Use faster base model

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore example scripts in `examples/`
- Customize training configuration
- Add domain-specific training data
- Deploy as API service

## Support

- GitHub Issues: [Report issues](https://github.com/yourusername/Zeus/issues)
- Documentation: See [README.md](README.md)
- Examples: Check `examples/` directory

## Quick Command Reference

```bash
# Validate data
python -m src.data_preparation.prepare_data --validate-only

# Process data
python -m src.data_preparation.prepare_data

# Train model
python -m src.training.train --config configs/training_config.yaml

# Evaluate model
python -m src.evaluation.evaluate --model-path models/splunk-query-llm --test-file data/processed/test_alpaca.jsonl

# Interactive CLI
python -m src.inference.cli --model-path models/splunk-query-llm --interactive

# Start API server
python -m src.inference.server --model-path models/splunk-query-llm --port 8000

# Run examples
python examples/basic_usage.py
python examples/batch_processing.py
python examples/api_client_example.py
```
