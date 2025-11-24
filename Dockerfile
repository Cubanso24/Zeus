# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY startup.sh ./startup.sh

# Create necessary directories
RUN mkdir -p /app/data/feedback /app/logs

# Make startup script executable
RUN chmod +x /app/startup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Health check
# Long start-period to allow model download and loading (first startup can take 5-10 minutes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the startup script which initializes DB and starts the server
CMD ["/app/startup.sh"]
