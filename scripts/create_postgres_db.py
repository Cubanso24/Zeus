"""Create PostgreSQL database and user for Zeus."""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from loguru import logger

def create_database():
    """Create the zeus_db database and zeus_user role."""

    # Database configuration
    DB_NAME = "zeus_db"
    DB_USER = "zeus_user"
    DB_PASSWORD = "zeus_password"

    try:
        # Connect to PostgreSQL as default user
        # Try common default configurations
        connection_attempts = [
            {"user": "postgres", "password": "", "host": "localhost"},
            {"user": "postgres", "password": "postgres", "host": "localhost"},
            {"user": os.getenv("USER"), "password": "", "host": "localhost"},  # Current user
        ]

        conn = None
        for attempt in connection_attempts:
            try:
                logger.info(f"Attempting to connect as user: {attempt['user']}")
                conn = psycopg2.connect(
                    database="postgres",
                    **attempt
                )
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                logger.info(f"Successfully connected as {attempt['user']}")
                break
            except psycopg2.OperationalError as e:
                logger.debug(f"Failed to connect as {attempt['user']}: {e}")
                continue

        if not conn:
            logger.error("Could not connect to PostgreSQL with any default credentials")
            logger.info("Please ensure PostgreSQL is running and you have access")
            return False

        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (DB_USER,))
        user_exists = cursor.fetchone() is not None

        if not user_exists:
            logger.info(f"Creating user: {DB_USER}")
            cursor.execute(f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}'")
            logger.info(f"User {DB_USER} created successfully")
        else:
            logger.info(f"User {DB_USER} already exists")

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
        db_exists = cursor.fetchone() is not None

        if not db_exists:
            logger.info(f"Creating database: {DB_NAME}")
            cursor.execute(f"CREATE DATABASE {DB_NAME} OWNER {DB_USER}")
            logger.info(f"Database {DB_NAME} created successfully")
        else:
            logger.info(f"Database {DB_NAME} already exists")

        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER}")
        logger.info(f"Granted privileges to {DB_USER}")

        cursor.close()
        conn.close()

        logger.info("Database setup completed successfully!")
        logger.info(f"Connection string: postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}")

        return True

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


if __name__ == "__main__":
    import os
    success = create_database()
    exit(0 if success else 1)
