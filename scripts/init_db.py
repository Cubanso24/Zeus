"""Initialize the database with tables and optional admin user."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import click
from loguru import logger

from src.database.database import engine, Base, get_db_context
from src.database.models import User, Query, Feedback, TrainingJob
from src.database.auth import get_password_hash


@click.command()
@click.option('--create-admin', is_flag=True, help='Create an admin user')
@click.option('--admin-username', default='admin', help='Admin username')
@click.option('--admin-email', default='admin@example.com', help='Admin email')
@click.option('--admin-password', default='admin123', help='Admin password')
@click.option('--drop-existing', is_flag=True, help='Drop existing tables before creating')
def init_database(
    create_admin: bool,
    admin_username: str,
    admin_email: str,
    admin_password: str,
    drop_existing: bool,
):
    """
    Initialize the database with tables and optional admin user.

    Example:
        python scripts/init_db.py --create-admin --admin-password "secure_password"
    """
    logger.info("Initializing database...")

    try:
        # Drop existing tables if requested
        if drop_existing:
            logger.warning("Dropping existing tables...")
            Base.metadata.drop_all(bind=engine)
            logger.info("Tables dropped successfully")

        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Create admin user if requested
        if create_admin:
            logger.info(f"Creating admin user: {admin_username}")

            with get_db_context() as db:
                # Check if admin already exists
                existing_admin = db.query(User).filter(User.username == admin_username).first()

                if existing_admin:
                    logger.warning(f"Admin user '{admin_username}' already exists")
                else:
                    admin_user = User(
                        username=admin_username,
                        email=admin_email,
                        hashed_password=get_password_hash(admin_password),
                        full_name="Administrator",
                        is_admin=True,
                        is_active=True,
                    )
                    db.add(admin_user)
                    db.commit()
                    logger.info(f"Admin user '{admin_username}' created successfully")
                    logger.info(f"Email: {admin_email}")
                    logger.warning(f"Password: {admin_password} (Change this immediately!)")

        logger.info("Database initialization complete!")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    init_database()
