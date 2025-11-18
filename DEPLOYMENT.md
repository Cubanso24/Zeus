# Zeus - Deployment Guide

This guide explains how to deploy Zeus using Docker so your analysts can access it from anywhere.

## Prerequisites

- Docker installed on the host machine
- Docker Compose installed
- Your fine-tuned model in `models/splunk-query-llm/`

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Start the Services

```bash
docker-compose up -d
```

This will start:
- **API Server** on port 8000
- **Web UI** on port 3000

### 3. Access Zeus

- **Web Chat Interface**: http://your-server-ip:3000
- **API Documentation**: http://your-server-ip:8000/docs
- **Health Check**: http://your-server-ip:8000/health

### 4. Stop the Services

```bash
docker-compose down
```

---

## Deployment Options

### Option 1: Local Network Deployment

Deploy on a server within your network:

```bash
# On your deployment server
git clone <your-repo>
cd Zeus

# Copy your trained model to the models directory
# (or mount from a shared drive)

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f
```

Analysts access via: `http://your-server-ip:3000`

### Option 2: Cloud Deployment (AWS/Azure/GCP)

#### AWS EC2 Example:

1. Launch an EC2 instance (recommend t3.xlarge or larger)
2. Install Docker and Docker Compose
3. Clone repository and copy model
4. Configure security group to allow ports 3000 and 8000
5. Run docker-compose

```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy Zeus
cd /home/ec2-user
git clone <your-repo>
cd Zeus
docker-compose up -d
```

Analysts access via: `http://your-ec2-public-ip:3000`

### Option 3: Behind Reverse Proxy (Production)

For production use with HTTPS:

```nginx
# /etc/nginx/sites-available/zeus
server {
    listen 80;
    server_name zeus.yourcompany.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name zeus.yourcompany.com;

    ssl_certificate /etc/ssl/certs/zeus.crt;
    ssl_certificate_key /etc/ssl/private/zeus.key;

    # Web UI
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

---

## Configuration

### Environment Variables

Edit `docker-compose.yml` to configure:

```yaml
environment:
  - MODEL_PATH=/app/models/splunk-query-llm
  - BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
  - LOG_LEVEL=info
```

### Resource Limits

Add resource limits to prevent OOM:

```yaml
services:
  zeus-api:
    # ... other config ...
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### Persistent Logs

Add volume for logs:

```yaml
volumes:
  - ./models:/app/models:ro
  - ./logs:/app/logs  # Add this
```

---

## Updating the Model

When you fine-tune a new version:

1. **Train locally** (as you do now)
2. **Copy model to server**:
   ```bash
   rsync -avz --progress models/ user@server:/path/to/Zeus/models/
   ```
3. **Restart container**:
   ```bash
   docker-compose restart zeus-api
   ```

---

## Monitoring

### Check Service Status

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f zeus-api
docker-compose logs -f zeus-web

# Check resource usage
docker stats
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Web UI
curl http://localhost:3000/
```

---

## Scaling

### Run Multiple API Instances

```yaml
services:
  zeus-api:
    # ... config ...
    deploy:
      replicas: 3  # Run 3 instances
```

### Add Load Balancer

Use nginx or HAProxy to load balance across instances.

---

## Security Best Practices

1. **Use HTTPS** in production (with reverse proxy)
2. **Restrict network access** (firewall rules, security groups)
3. **Add authentication** (API keys, OAuth)
4. **Keep base images updated**:
   ```bash
   docker-compose pull
   docker-compose up -d
   ```
5. **Use secrets** for sensitive config (Docker secrets or env files)

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs zeus-api

# Common issues:
# - Model path not found: Check volume mount
# - Out of memory: Increase Docker memory limit
# - Port already in use: Change ports in docker-compose.yml
```

### Slow query generation

- Increase container CPU/memory
- Consider GPU instance (AWS p3, Azure NC-series)
- Reduce `max_new_tokens` in generation settings

### Can't access from other devices

```bash
# Check firewall
sudo ufw status
sudo ufw allow 3000
sudo ufw allow 8000

# Check Docker network
docker network inspect zeus_zeus-network
```

---

## Cost Optimization

### For AWS:

- Use **Spot Instances** for non-critical workloads
- Use **EC2 Auto Scaling** based on usage
- Consider **AWS ECS Fargate** for serverless containers
- Use **CloudWatch** for monitoring

### For Azure:

- Use **Azure Container Instances** for simple deployment
- Use **Azure Kubernetes Service** for production scale
- Consider **Azure Spot VMs**

---

## Next Steps

1. ✅ Deploy to a test server
2. ✅ Test with your team
3. ✅ Set up HTTPS with proper domain
4. ✅ Add authentication if needed
5. ✅ Monitor usage and scale as needed

For questions or issues, check the logs first:
```bash
docker-compose logs -f
```
