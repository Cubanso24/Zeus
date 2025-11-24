# Zeus Native Deployment (Mac M1 Optimized)

This guide covers running Zeus natively on your Mac to leverage M1 GPU acceleration for fast inference (15-30 seconds instead of 5-10 minutes on CPU).

## Quick Start

```bash
# Start Zeus (one command does everything!)
./start_native.sh

# Stop Zeus
./stop_native.sh
```

## What `start_native.sh` Does

The startup script automatically:

1. ✅ Stops Docker containers if running
2. ✅ Checks/installs PostgreSQL via Homebrew
3. ✅ Creates database and user
4. ✅ Activates Zeus conda environment
5. ✅ Initializes database tables
6. ✅ Creates admin user (username: `admin`, password: `admin123`)
7. ✅ Starts API server with M1 GPU acceleration
8. ✅ Starts web server
9. ✅ Verifies everything is running

## Prerequisites

- **Conda/Miniconda** installed
- **Zeus conda environment** created with dependencies
- **Homebrew** installed (script will use it to install PostgreSQL if needed)

## Access Points

Once started:
- **Chat Interface**: http://localhost:8081
- **Admin Dashboard**: http://localhost:8081/admin.html
- **API Health**: http://localhost:8080/health

**Login**: username `admin`, password `admin123`

## Performance

- **Device**: M1 Max GPU (Metal)
- **Response Time**: 15-30 seconds per query
- **vs Docker**: 20-40x faster than CPU-only Docker

## Logs

View real-time logs:
```bash
# API server logs
tail -f logs/api_server.log

# Web server logs
tail -f logs/web_server.log
```

## Troubleshooting

### PostgreSQL Issues

```bash
# Check if PostgreSQL is running
pg_isready

# Start PostgreSQL manually
brew services start postgresql@15

# View PostgreSQL logs
tail -f $(brew --prefix)/var/log/postgres.log
```

### Port Conflicts

```bash
# Check what's using ports 8080/8081
lsof -i :8080
lsof -i :8081

# The start script automatically kills conflicting processes
```

### API Not Starting

```bash
# Check API logs
tail -f logs/api_server.log

# Verify conda environment
conda activate Zeus
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

## Database Management

### Connect to Database

```bash
psql -d zeus_db
```

### Backup Database

```bash
pg_dump zeus_db > backup_$(date +%Y%m%d).sql
```

### Reset Database

```bash
dropdb zeus_db
createdb zeus_db
./start_native.sh  # Will reinitialize
```

## Process Management

The scripts create PID files:
- `.api.pid` - API server process ID
- `.web.pid` - Web server process ID

You can manually manage processes:
```bash
# Check if running
cat .api.pid && ps -p $(cat .api.pid)

# Kill manually
kill $(cat .api.pid)
kill $(cat .web.pid)
```

## Comparison: Native vs Docker

| Aspect | Native (Mac M1) | Docker (Mac) |
|--------|-----------------|--------------|
| GPU Access | ✅ M1 GPU (Metal) | ❌ CPU Only |
| Response Time | 15-30 seconds | 5-10 minutes |
| Setup Complexity | Low (one script) | Medium |
| Best For | Development, single user | Multi-user, production Linux |
| Persistence | PostgreSQL + files | Docker volumes |

## Switching Between Native and Docker

### From Docker to Native
```bash
# Export data from Docker (optional)
docker compose exec postgres pg_dump -U zeus_user zeus_db > docker_backup.sql

# Stop Docker
docker compose down

# Start native
./start_native.sh

# Import data (optional)
psql -d zeus_db < docker_backup.sql
```

### From Native to Docker
```bash
# Backup native database (optional)
pg_dump zeus_db > native_backup.sql

# Stop native
./stop_native.sh

# Start Docker
docker compose up -d --build --scale zeus-api=3

# Import data (optional)
cat native_backup.sql | docker compose exec -T postgres psql -U zeus_user zeus_db
```

## Production Deployment

For production with multiple users:
- **Mac Development**: Use native deployment (this guide)
- **Production Server**: Use Docker on Linux with NVIDIA GPU

The Docker setup is production-ready but slow on Mac due to GPU limitations.
