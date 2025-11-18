# Zeus Admin Dashboard Documentation

This guide covers the Zeus Admin Dashboard and management commands for administrators.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Admin Dashboard Features](#admin-dashboard-features)
- [Docker Management](#docker-management)
- [Database Management](#database-management)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Quick Reference

### Essential Commands

```bash
# Start Zeus with 3 API instances
docker compose up -d --scale zeus-api=3

# Scale up for high load
docker compose up -d --scale zeus-api=8 --no-recreate

# Scale down
docker compose up -d --scale zeus-api=2 --no-recreate

# Check status
docker compose ps

# View logs
docker compose logs -f zeus-api
docker compose logs -f nginx
docker compose logs -f postgres

# Stop everything
docker compose down

# Restart services
docker compose restart
```

### Database Commands

```bash
# Create admin user (default credentials: admin/admin123)
docker compose exec zeus-api python scripts/create_admin_user.py

# Create custom admin user
docker compose exec zeus-api python scripts/create_admin_user.py \
  --username myusername \
  --password mypassword \
  --email admin@company.com

# Run database migrations
docker compose exec zeus-api python scripts/migrate_add_training_jobs.py

# Backup database
docker compose exec postgres pg_dump -U zeus_user zeus_db > backup.sql

# Restore database
cat backup.sql | docker compose exec -T postgres psql -U zeus_user zeus_db
```

### Monitoring Commands

```bash
# Real-time container resource usage
docker stats

# Container resource usage (API instances only)
docker stats $(docker ps --filter "name=zeus-api" -q)

# Check container health
docker compose ps

# View system logs
docker compose logs --tail=100

# Follow logs in real-time
docker compose logs -f
```

## Admin Dashboard Features

Access the admin dashboard at: **http://localhost:8081/admin.html**

Default credentials:
- Username: `admin`
- Password: `admin123`

### Dashboard Tabs

#### 1. Analytics Tab
- Total queries generated
- Total users
- Average feedback rating
- Query success rate
- Query volume over time (7-day chart)
- Rating distribution
- Top users by query count

#### 2. Queries Tab
Features:
- View all generated queries with timestamps
- Filter by user and date range
- Search functionality
- Pagination (20 queries per page)
- Displays:
  - Query instruction
  - Generated SPL query
  - User information
  - Timestamp

#### 3. Feedback Tab
Features:
- View all user feedback
- Edit feedback entries
- Delete poor quality feedback
- Filter by rating (good/bad/neutral)
- Export cleaned feedback for retraining
- Displays:
  - Original query
  - Feedback rating and comments
  - User information
  - Edit/Delete actions

**Export Feedback for Retraining:**
1. Filter feedback by rating (e.g., "good" only)
2. Click "Export Good Feedback" or "Export Bad Feedback"
3. Downloads JSONL file ready for model training

#### 4. System Tab
Real-time system monitoring:
- CPU usage percentage
- RAM usage (used/total)
- Disk usage (used/total)
- Auto-refreshes every 30 seconds
- Manual refresh button

#### 5. Training Tab
Model retraining features:
- View all training jobs
- Trigger new training jobs
- Monitor training status (pending/running/completed/failed)
- View training metrics
- Displays:
  - Job ID and status
  - Start and end times
  - Configuration details
  - Training metrics
  - Model output path

**Start New Training Job:**
1. Configure training parameters (epochs, batch size, learning rate)
2. Select input file path
3. Click "Start Training"
4. Monitor progress in the training jobs table

## Docker Management

### Starting Zeus

**Basic start (3 instances):**
```bash
docker compose up -d --scale zeus-api=3
```

**Start with specific instance count:**
```bash
docker compose up -d --scale zeus-api=5
```

**Start and rebuild:**
```bash
docker compose up -d --build --scale zeus-api=3
```

### Scaling Operations

**Scale up (no downtime):**
```bash
# Increase from 3 to 8 instances
docker compose up -d --scale zeus-api=8 --no-recreate
```

**Scale down:**
```bash
# Decrease from 8 to 3 instances
docker compose up -d --scale zeus-api=3 --no-recreate
```

**Restart with new scale:**
```bash
docker compose down
docker compose up -d --scale zeus-api=5
```

### Viewing Logs

**All services:**
```bash
docker compose logs -f
```

**Specific service:**
```bash
docker compose logs -f zeus-api
docker compose logs -f nginx
docker compose logs -f postgres
```

**Last N lines:**
```bash
docker compose logs --tail=50 zeus-api
```

**Specific container:**
```bash
docker logs zeus-zeus-api-1 -f
```

### Stopping Services

**Stop all services:**
```bash
docker compose down
```

**Stop and remove volumes (WARNING: deletes data):**
```bash
docker compose down -v
```

**Stop specific service:**
```bash
docker compose stop zeus-api
```

**Restart specific service:**
```bash
docker compose restart nginx
```

## Database Management

### Accessing the Database

**Interactive PostgreSQL shell:**
```bash
docker compose exec postgres psql -U zeus_user -d zeus_db
```

**Run SQL query:**
```bash
docker compose exec postgres psql -U zeus_user -d zeus_db -c "SELECT COUNT(*) FROM queries;"
```

### Backup and Restore

**Backup database:**
```bash
# Full backup
docker compose exec postgres pg_dump -U zeus_user zeus_db > zeus_backup_$(date +%Y%m%d).sql

# Compressed backup
docker compose exec postgres pg_dump -U zeus_user zeus_db | gzip > zeus_backup_$(date +%Y%m%d).sql.gz
```

**Restore database:**
```bash
# From uncompressed backup
cat zeus_backup_20250101.sql | docker compose exec -T postgres psql -U zeus_user zeus_db

# From compressed backup
gunzip -c zeus_backup_20250101.sql.gz | docker compose exec -T postgres psql -U zeus_user zeus_db
```

### Database Queries

**View table sizes:**
```bash
docker compose exec postgres psql -U zeus_user -d zeus_db -c "
SELECT
  schemaname as schema,
  tablename as table,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

**Query statistics:**
```bash
docker compose exec postgres psql -U zeus_user -d zeus_db -c "
SELECT
  COUNT(*) as total_queries,
  COUNT(DISTINCT user_id) as unique_users,
  AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE NULL END) as avg_rating
FROM queries;
"
```

## Monitoring

### Container Health

**Check all containers:**
```bash
docker compose ps
```

**Check specific container health:**
```bash
docker inspect --format='{{.State.Health.Status}}' zeus-zeus-api-1
```

### Resource Usage

**Real-time stats:**
```bash
docker stats
```

**One-time stats:**
```bash
docker stats --no-stream
```

**Specific containers:**
```bash
docker stats zeus-nginx zeus-postgres
```

### Health Endpoints

**Load balancer health:**
```bash
curl http://localhost:8081/health
```

**API health (through load balancer):**
```bash
curl http://localhost:8081/api/health
```

**Load balancer status:**
```bash
curl http://localhost:8081/lb-status
```

### System Monitoring via Admin Dashboard

1. Navigate to http://localhost:8081/admin.html
2. Login with admin credentials
3. Go to "System" tab
4. View real-time metrics:
   - CPU usage
   - Memory usage
   - Disk usage
5. Auto-refreshes every 30 seconds

## Troubleshooting

### Common Issues

#### Issue: Containers Won't Start

```bash
# Check logs for errors
docker compose logs

# Remove all containers and start fresh
docker compose down
docker compose up -d --scale zeus-api=3
```

#### Issue: Nginx Keeps Restarting

```bash
# Check nginx logs
docker compose logs nginx

# Test nginx configuration
docker compose exec nginx nginx -t

# Restart nginx
docker compose restart nginx
```

#### Issue: Database Connection Errors

```bash
# Check postgres health
docker compose ps postgres

# View database logs
docker compose logs postgres

# Restart postgres
docker compose restart postgres
```

#### Issue: High Memory Usage

```bash
# Check memory usage
docker stats --no-stream

# Scale down API instances
docker compose up -d --scale zeus-api=2 --no-recreate

# Increase Docker memory limit in Docker Desktop settings
```

#### Issue: Port Already in Use

```bash
# Find process using port 8081
lsof -i :8081

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Debugging Commands

**Enter container shell:**
```bash
# API container
docker compose exec zeus-api /bin/bash

# Nginx container
docker compose exec nginx /bin/sh

# Postgres container
docker compose exec postgres /bin/bash
```

**View container configuration:**
```bash
docker inspect zeus-zeus-api-1
```

**View environment variables:**
```bash
docker compose exec zeus-api env
```

### Performance Tuning

**Recommended instance counts based on load:**

| Concurrent Users | API Instances | RAM Required |
|------------------|---------------|--------------|
| 1-5              | 2-3           | 8GB          |
| 5-10             | 3-5           | 16GB         |
| 10-20            | 5-8           | 24GB         |
| 20-50            | 8-12          | 32GB+        |

**Scale based on CPU/Memory:**
- If CPU > 80%: Scale up API instances
- If Memory > 85%: Scale down or add more RAM
- If Response Time > 5s: Scale up

## Access Points

Once Zeus is running:

- **Frontend (Chat)**: http://localhost:8081
- **Admin Dashboard**: http://localhost:8081/admin.html
- **API (via Load Balancer)**: http://localhost:8081/api/
- **Health Check**: http://localhost:8081/health
- **Load Balancer Status**: http://localhost:8081/lb-status

## Security Best Practices

### Production Deployment

1. **Change default admin password:**
   ```bash
   docker compose exec zeus-api python scripts/create_admin_user.py \
     --username admin \
     --password <strong-password>
   ```

2. **Update environment variables in `.env`:**
   ```bash
   SECRET_KEY=<generate-strong-secret-key>
   POSTGRES_PASSWORD=<strong-database-password>
   ```

3. **Enable HTTPS** (use a reverse proxy like nginx or Traefik)

4. **Backup database regularly:**
   ```bash
   # Add to crontab for daily backups
   0 2 * * * cd /path/to/Zeus && docker compose exec postgres pg_dump -U zeus_user zeus_db | gzip > backups/zeus_$(date +\%Y\%m\%d).sql.gz
   ```

5. **Monitor logs for suspicious activity:**
   ```bash
   docker compose logs | grep -i "error\|fail\|unauthorized"
   ```

6. **Limit admin access** by IP or VPN

7. **Review and clean feedback** regularly to maintain data quality

## Advanced Configuration

### Resource Limits

Add to `docker-compose.yml` under `zeus-api` service:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### Custom Nginx Configuration

Edit `nginx.conf` and restart:

```bash
# Edit configuration
vim nginx.conf

# Test configuration
docker compose exec nginx nginx -t

# Apply changes
docker compose restart nginx
```

### Database Tuning

For high-load scenarios, tune PostgreSQL:

```bash
# Access postgres config
docker compose exec postgres vi /var/lib/postgresql/data/postgresql.conf

# Restart to apply
docker compose restart postgres
```

## Support

For issues or questions:
- Check logs: `docker compose logs`
- Review this documentation
- Check [SCALING.md](SCALING.md) for scaling-specific issues
- Open an issue on GitHub
