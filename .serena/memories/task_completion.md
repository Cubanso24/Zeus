# Task Completion Checklist

When completing a task in Zeus:

## Before Committing
1. **Test locally**: Ensure the feature works
2. **Check logs**: `docker compose logs -f zeus-api`
3. **Format code**: `black src/ --line-length 100`
4. **Lint**: `flake8 src/`

## If Modifying API
- Update relevant endpoint documentation
- Test with curl or web UI
- Check authentication requirements

## If Modifying Frontend
- Test in browser (http://localhost:8081)
- Check admin dashboard if applicable (http://localhost:8081/admin.html)
- Test with different browsers if UI-heavy

## If Modifying Model/Inference
- Test query generation
- Check semantic cache behavior
- Verify Wazuh RAG if applicable

## If Modifying Database
- Apply migrations if needed
- Test with fresh database
- Verify data persistence

## Common Verification
```bash
# Rebuild and restart
docker compose up -d --build

# Check health
./scripts/check_health.sh

# View logs
docker compose logs -f zeus-api
```
