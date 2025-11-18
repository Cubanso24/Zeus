# Zeus Splunk Query LLM - Docker Compose Deployment

This guide explains how to deploy Zeus using Docker Compose with multiple parallel instances and load balancing.

## Architecture

The deployment consists of:
- **3x Zeus API instances** - FastAPI servers with loaded models
- **1x Nginx** - Load balancer with rate limiting
- **1x PostgreSQL** - Centralized database
- **Persistent volumes** - For database and feedback data

## Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- At least 16GB RAM (model loading is memory intensive)
- 20GB free disk space

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your secure passwords
nano .env
```

**Important**: Change these values in `.env`:
- `POSTGRES_PASSWORD` - Strong database password
- `SECRET_KEY` - Random string for JWT tokens (generate with `openssl rand -hex 32`)

### 2. Build and Start

```bash
# Build images (first time only, ~10-15 mins)
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Initialize Database

The database will be automatically created. To initialize with schema:

```bash
docker-compose exec zeus-api-1 python scripts/init_db.py
```

### 4. Wait for Model Loading

**IMPORTANT**: First-time startup takes 5-10 minutes because the containers need to download the Mistral-7B base model (~14GB) from Hugging Face. Subsequent startups are faster if the model is cached.

Monitor the startup progress:
```bash
# Watch container status
docker-compose logs -f zeus-api-1

# Check if models are loaded (wait for "Uvicorn running" message)
docker-compose logs zeus-api-1 | grep -i "uvicorn running"
```

### 5. Access the Application

- **Frontend**: http://localhost:8081/
- **API Health**: http://localhost:8081/health
- **API Docs**: http://localhost:8081/api/docs

**Note**: Port changed to 8081 to avoid conflicts with system services.

## Management Commands

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f zeus-api-1
docker-compose logs -f nginx

# Restart all services
docker-compose restart

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Scale API instances (add more workers)
docker-compose up -d --scale zeus-api=5
```

## Scaling

To add more API instances, edit `docker-compose.yml` and add additional services:

```yaml
  zeus-api-4:
    build:
      context: .
      dockerfile: Dockerfile
    # ... copy config from zeus-api-1
```

Then update `nginx.conf` to include the new instance:

```nginx
upstream zeus_backend {
    server zeus-api-4:8080 max_fails=3 fail_timeout=30s;
}
```

## Monitoring

### Health Checks

```bash
# Check all container health
docker-compose ps

# Test load balancer
curl http://localhost/health

# Test API directly
curl http://localhost/api/health
```

### Resource Usage

```bash
# Monitor resource usage
docker stats

# Check disk usage
docker system df
```

## Troubleshooting

### Containers won't start

```bash
# Check logs
docker-compose logs

# Rebuild images
docker-compose build --no-cache
docker-compose up -d
```

### Database connection errors

```bash
# Check PostgreSQL is healthy
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Model loading issues

Models are mounted read-only from `./models`. Ensure:
- Model files exist in `models/splunk-query-llm-v2/`
- Sufficient RAM available (check with `docker stats`)

### Out of memory

```bash
# Reduce number of instances
docker-compose stop zeus-api-3

# Or allocate more memory to Docker
# Docker Desktop: Settings > Resources > Memory
```

## Production Recommendations

1. **SSL/TLS**: Add HTTPS support via nginx or reverse proxy
2. **Secrets Management**: Use Docker secrets or external vault
3. **Monitoring**: Add Prometheus + Grafana
4. **Logging**: Configure centralized logging (ELK, Loki)
5. **Backups**: Implement automated database backups
6. **Resource Limits**: Set memory/CPU limits in docker-compose.yml

Example resource limits:

```yaml
  zeus-api-1:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 8G
        reservations:
          cpus: '1.0'
          memory: 4G
```

## Upgrading

```bash
# Pull latest changes
git pull

# Rebuild images
docker-compose build

# Restart with zero-downtime (one at a time)
docker-compose up -d --no-deps --build zeus-api-1
docker-compose up -d --no-deps --build zeus-api-2
docker-compose up -d --no-deps --build zeus-api-3

# Or restart all at once
docker-compose up -d --build
```

## Backup & Restore

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U zeus_user zeus_db > backup.sql

# Or with Docker
docker-compose exec -T postgres pg_dump -U zeus_user zeus_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# Restore from backup
docker-compose exec -T postgres psql -U zeus_user zeus_db < backup.sql

# Or from gzip
gunzip < backup_20250117.sql.gz | docker-compose exec -T postgres psql -U zeus_user zeus_db
```

## Support

For issues, check:
1. Container logs: `docker-compose logs`
2. Resource usage: `docker stats`
3. Network connectivity: `docker-compose exec zeus-api-1 ping postgres`

