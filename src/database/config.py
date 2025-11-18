"""Database configuration."""

import os
from typing import Optional

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://zeus_user:zeus_password@localhost:5432/zeus_db"
)

# JWT configuration
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "your-secret-key-change-this-in-production"  # Change this in production!
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


def get_database_url() -> str:
    """Get the database URL from environment or default."""
    return DATABASE_URL


def get_secret_key() -> str:
    """Get the JWT secret key."""
    return SECRET_KEY
