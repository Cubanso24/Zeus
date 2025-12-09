# Zeus Code Style & Conventions

## Python Style
- **Line Length**: 100 characters (Black formatter)
- **Python Version**: 3.9+ target
- **Formatter**: Black
- **Linter**: flake8
- **Type Checker**: mypy (optional type hints)

## Code Conventions
- Use loguru for logging
- Pydantic models for request/response validation
- Async endpoints in FastAPI where beneficial
- Type hints encouraged but not strictly enforced

## File Organization
- Each module has `__init__.py` for exports
- Related functionality grouped in subpackages
- Tests mirror src structure in `tests/`

## Naming Conventions
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names preferred

## API Conventions
- RESTful endpoints
- JWT authentication via Bearer tokens
- JSON request/response bodies
- Pydantic models for validation

## Docker
- Services defined in docker-compose.yml
- Nginx as reverse proxy/load balancer
- PostgreSQL for persistence
- GPU support via NVIDIA Container Toolkit
