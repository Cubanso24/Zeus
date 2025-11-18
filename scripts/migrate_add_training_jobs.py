"""
Database migration: Add training_jobs table.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from loguru import logger
from src.database.database import engine, Base
from src.database.models import TrainingJob  # Import to register the model

def migrate():
    """Add training_jobs table to database."""
    logger.info("Starting database migration: Add training_jobs table")

    try:
        # Check if table already exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='training_jobs'"
            ))
            if result.fetchone():
                logger.info("training_jobs table already exists, skipping migration")
                return

        # Create training_jobs table
        TrainingJob.__table__.create(engine)
        logger.info("✓ Created training_jobs table successfully")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    migrate()
    logger.info("Migration completed successfully")
