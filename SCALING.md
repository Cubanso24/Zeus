# Zeus API Scaling Guide

This guide explains how to scale the Zeus API to handle increased load from your analysts.

## Quick Start

When analysts complain about lag, use the scaling script:

```bash
# Scale to 5 instances (while system is running)
./scripts/scale_api.sh scale 5

# Or restart with more instances
./scripts/scale_api.sh restart 8
```

## How It Works

Zeus uses Docker Compose with nginx load balancing:

- **zeus-api**: Scalable FastAPI instances (default: 3 replicas)
- **nginx**: Load balancer distributing traffic across all API instances
- **postgres**: Shared database for all instances

Nginx automatically discovers new API instances using Docker's DNS resolver, so no configuration changes needed when scaling.

## Scaling Commands

### Start System

```bash
# Start with default 3 instances
./scripts/scale_api.sh up

# Start with specific number of instances
./scripts/scale_api.sh up 5
```

### Scale Existing System

```bash
# Scale to 8 instances (no downtime)
./scripts/scale_api.sh scale 8

# Scale down to 2 instances
./scripts/scale_api.sh scale 2
```

### Check Status

```bash
# View all running services
./scripts/scale_api.sh status

# Example output:
# Zeus Services Status:
# =========================================
# NAME                    IMAGE           STATUS
# zeus-nginx              nginx:alpine    Up 5 minutes
# zeus-api-1              zeus-api        Up 5 minutes
# zeus-api-2              zeus-api        Up 5 minutes
# zeus-api-3              zeus-api        Up 5 minutes
# zeus-postgres           postgres:15     Up 5 minutes
#
# API Instances: 3
# Load Balancer: running
# Database: running
```

### View Logs

```bash
# View API logs
./scripts/scale_api.sh logs

# View nginx logs
./scripts/scale_api.sh logs nginx

# View database logs
./scripts/scale_api.sh logs postgres
```

### Stop System

```bash
./scripts/scale_api.sh down
```

## Manual Docker Compose Commands

If you prefer using Docker Compose directly:

```bash
# Start with 3 instances
docker compose up -d --scale zeus-api=3

# Scale to 5 instances
docker compose up -d --scale zeus-api=5 --no-recreate

# View status
docker compose ps

# View logs
docker compose logs -f zeus-api

# Stop all
docker compose down
```

## Performance Considerations

### When to Scale

- **CPU Usage > 80%**: Scale up to distribute processing load
- **Response Time > 5s**: Add more instances to handle concurrent requests
- **Queue Buildup**: More instances = more parallel processing

### Resource Requirements (per instance)

- **RAM**: ~4GB (model loading + inference)
- **CPU**: 2-4 cores recommended
- **GPU**: Optional, but significantly faster inference

### Recommended Instance Counts

| Concurrent Users | Recommended Instances |
|------------------|----------------------|
| 1-5              | 2-3                  |
| 5-10             | 3-5                  |
| 10-20            | 5-8                  |
| 20+              | 8-12                 |

## Monitoring

### Health Checks

```bash
# Check nginx load balancer health
curl http://localhost:8081/health

# Check API health (through load balancer)
curl http://localhost:8081/api/health
```

### Resource Monitoring

Use the admin dashboard to monitor system resources:
- Navigate to: http://localhost:8081/admin.html
- Login with admin credentials
- Go to "System" tab to see:
  - CPU usage across all containers
  - RAM usage
  - Active connections
  - Request rate

### Container Stats

```bash
# Real-time resource usage for all containers
docker stats

# Resource usage for API instances only
docker stats $(docker ps --filter "name=zeus-api" -q)
```

## Troubleshooting

### Instance Won't Start

```bash
# Check logs
./scripts/scale_api.sh logs

# Common issues:
# - Out of memory: Reduce instance count or add more RAM
# - Model not found: Check that models/ directory is mounted correctly
# - Database connection: Ensure postgres is healthy
```

### Load Balancer Not Distributing

```bash
# Restart nginx to refresh DNS cache
docker compose restart nginx

# Check nginx configuration
docker compose exec nginx nginx -t
```

### Database Connection Issues

```bash
# Check database is healthy
docker compose ps postgres

# View database logs
./scripts/scale_api.sh logs postgres

# Restart database (warning: may cause brief downtime)
docker compose restart postgres
```

### High Memory Usage

```bash
# Check memory per container
docker stats --no-stream

# If needed, scale down
./scripts/scale_api.sh scale 2

# Or increase Docker memory limit in Docker Desktop settings
```

## Production Deployment

For production, consider:

1. **Environment Variables**: Set in `.env` file
   ```bash
   POSTGRES_PASSWORD=your-secure-password
   SECRET_KEY=your-secret-key
   ```

2. **Persistent Data**: Database is stored in Docker volume `postgres_data`
   ```bash
   # Backup database
   docker compose exec postgres pg_dump -U zeus_user zeus_db > backup.sql
   ```

3. **Resource Limits**: Add to docker-compose.yml
   ```yaml
   zeus-api:
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 4G
         reservations:
           cpus: '1'
           memory: 2G
   ```

4. **Monitoring**: Consider adding Prometheus + Grafana for production metrics

5. **Auto-scaling**: Use Docker Swarm or Kubernetes for automatic scaling based on load

## Access Points

Once running, Zeus is available at:

- **Frontend (Chat)**: http://localhost:8081
- **Admin Dashboard**: http://localhost:8081/admin.html
- **API (via Load Balancer)**: http://localhost:8081/api/
- **Health Check**: http://localhost:8081/health

All traffic goes through nginx load balancer on port 8081, which distributes to backend API instances automatically.
