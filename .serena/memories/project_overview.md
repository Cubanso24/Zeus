# Zeus Project Overview

## Purpose
Zeus is a fine-tuned Large Language Model (LLM) for generating Splunk queries from natural language descriptions. Built for cybersecurity analysts, it converts questions like "Find failed SSH logins in the last 24 hours" into accurate SPL (Splunk Processing Language) queries.

## Key Features
- Natural language to Splunk query generation
- Asks clarifying questions when requests are ambiguous
- Web UI with query history and feedback system
- Admin dashboard for monitoring and model retraining
- REST API for integrations
- GPU-accelerated inference (CUDA support)
- Horizontal scaling with load balancing
- **Wazuh EDR Integration**: Specialized RAG for Wazuh alert queries with automatic field name translation
- **Semantic Cache**: Learns from approved queries to serve similar requests instantly
- **Index Selector**: UI support for targeting specific indexes (wazuh-alerts, security, etc.)
- **Feedback Learning**: Bad queries with corrections are cached and used for future similar requests
- **Context Wizard**: Data source capability awareness

## Tech Stack
- **Language**: Python 3.9+
- **LLM Framework**: Hugging Face Transformers, PEFT (LoRA fine-tuning)
- **Base Model**: Mistral-7B-Instruct-v0.2
- **Web Framework**: FastAPI with Uvicorn
- **Database**: PostgreSQL
- **Caching**: Semantic similarity cache with sentence embeddings
- **Frontend**: HTML/JavaScript/CSS (vanilla)
- **Containerization**: Docker & Docker Compose
- **Load Balancing**: Nginx
- **GPU Support**: CUDA 11.8, PyTorch

## Project Structure
```
Zeus/
├── configs/              # Training and model configuration
├── data/
│   ├── raw/             # Training data (JSONL format)
│   ├── processed/       # Processed training data
│   ├── feedback/        # User feedback for retraining
│   └── wazuh_rag/       # Wazuh RAG knowledge base
├── models/              # Trained models and checkpoints
├── src/
│   ├── data_preparation/  # Data processing
│   ├── training/          # Model training
│   ├── inference/         # API server, semantic cache, Wazuh RAG
│   ├── evaluation/        # Metrics and evaluation
│   ├── database/          # Database models and auth
│   └── utils/             # Config utilities
├── scripts/             # Utility scripts
├── web/                 # Frontend HTML/JS/CSS
├── tests/               # Test suite
├── docker-compose.yml   # Docker orchestration
├── Dockerfile          # Container image definition
└── nginx.conf          # Nginx configuration
```
