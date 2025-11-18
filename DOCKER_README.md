# Zeus - Docker Quick Start

Deploy Zeus in minutes with Docker!

## 🚀 Quick Start (3 Steps)

### 1. Make sure you have a trained model

```bash
# Your model should be at:
ls models/splunk-query-llm/
```

### 2. Build and start Zeus

```bash
./deploy.sh build
./deploy.sh start
```

### 3. Access Zeus

Open your browser to **http://localhost:3000**

That's it! 🎉

---

## 📋 What Gets Deployed

Zeus runs as 2 Docker containers:

1. **zeus-api** - The LLM backend (port 8000)
2. **zeus-web** - The chat interface (port 3000)

Both containers run automatically and restart if they crash.

---

## 🔧 Common Tasks

### View Logs
```bash
./deploy.sh logs
```

### Check Status
```bash
./deploy.sh status
```

### Stop Zeus
```bash
./deploy.sh stop
```

### Restart After Model Update
```bash
# After fine-tuning a new model locally
./deploy.sh restart
```

---

## 🌐 Access From Other Devices

### Find your server's IP address:
```bash
# On Mac
ipconfig getifaddr en0

# On Linux
hostname -I | awk '{print $1}'

# On Windows
ipconfig
```

### Then analysts can access from any device:
```
http://YOUR-SERVER-IP:3000
```

**Example:** If your server IP is `192.168.1.100`:
- Analysts visit: `http://192.168.1.100:3000`

---

## 📦 Workflow: Train Locally, Deploy Remotely

### On Your Local Machine (for training):

```bash
# 1. Add training data
vim data/raw/train_*.jsonl

# 2. Process data
python -m src.data_preparation.prepare_data

# 3. Train model
python -m src.training.train --config configs/training_config.yaml

# 4. Copy to deployment server
rsync -avz models/ user@server:/path/to/Zeus/models/
```

### On Your Deployment Server:

```bash
# 5. Restart Zeus to use new model
./deploy.sh restart
```

---

## 🔒 Security Recommendations

### For Internal Network:
- ✅ Use as-is on trusted internal networks
- ✅ Restrict firewall to your company IPs

### For Internet-Facing:
- ⚠️ Add HTTPS (see DEPLOYMENT.md for nginx reverse proxy)
- ⚠️ Add authentication
- ⚠️ Use strong firewall rules

---

## 🐛 Troubleshooting

### Zeus won't start?

```bash
# Check if model exists
ls -la models/splunk-query-llm/

# View error logs
docker-compose logs zeus-api
```

### Can't access from other devices?

```bash
# Check if ports are open
docker-compose ps

# Check firewall (Mac)
sudo pfctl -s rules | grep 3000

# Check firewall (Linux)
sudo ufw status
```

### Out of memory?

Edit `docker-compose.yml`:
```yaml
services:
  zeus-api:
    deploy:
      resources:
        limits:
          memory: 8G  # Increase this
```

### Queries are slow?

CPU-only inference is slow. For production:
- Use a GPU instance (AWS p3, Azure NC-series)
- Or use a larger CPU instance
- Or reduce `max_new_tokens` in the code

---

## 🎯 Production Deployment

For production deployment with HTTPS, authentication, and monitoring, see:
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Full deployment guide
- **[README.md](README.md)** - Full Zeus documentation

---

## 💡 Tips

1. **Start small**: Test on your local network first
2. **Monitor resources**: Use `./deploy.sh status` to check memory/CPU
3. **Keep model updated**: Re-train regularly with new examples
4. **Collect feedback**: Ask analysts what queries work/don't work
5. **Scale gradually**: Start with 1 instance, add more if needed

---

## 📞 Support

If something doesn't work:

1. Check logs: `./deploy.sh logs`
2. Check status: `./deploy.sh status`
3. Rebuild: `./deploy.sh build && ./deploy.sh start`

Happy querying! ⚡
