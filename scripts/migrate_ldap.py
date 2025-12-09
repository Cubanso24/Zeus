"""Database migration script to add LDAP support columns to users table.

This script adds the following columns to the users table:
- is_ldap_user: Boolean flag to track LDAP vs local users
- ldap_dn: String to store the user's LDAP Distinguished Name

Also modifies hashed_password to be nullable for LDAP-only users.

Creates the system_settings table for storing LDAP configuration in DB.

Run this script after updating to the LDAP-enabled version:
    python scripts/migrate_ldap.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import click
from loguru import logger
from sqlalchemy import text

from src.database.database import engine


def check_column_exists(conn, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    result = conn.execute(text(f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
        )
    """), {"table": table, "column": column})
    return result.scalar()


@click.command()
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
def migrate_ldap(dry_run: bool):
    """
    Add LDAP support columns to the users table.

    This migration is idempotent - safe to run multiple times.
    """
    logger.info("Starting LDAP migration...")

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    with engine.connect() as conn:
        # Check if users table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = 'users'
            )
        """))
        if not result.scalar():
            logger.error("Users table does not exist. Run init_db.py first.")
            return

        changes_made = []

        # Add is_ldap_user column
        if not check_column_exists(conn, 'users', 'is_ldap_user'):
            logger.info("Adding is_ldap_user column...")
            if not dry_run:
                conn.execute(text("""
                    ALTER TABLE users
                    ADD COLUMN is_ldap_user BOOLEAN DEFAULT FALSE NOT NULL
                """))
            changes_made.append("Added is_ldap_user column")
        else:
            logger.info("is_ldap_user column already exists - skipping")

        # Add ldap_dn column
        if not check_column_exists(conn, 'users', 'ldap_dn'):
            logger.info("Adding ldap_dn column...")
            if not dry_run:
                conn.execute(text("""
                    ALTER TABLE users
                    ADD COLUMN ldap_dn VARCHAR(500) NULL
                """))
            changes_made.append("Added ldap_dn column")
        else:
            logger.info("ldap_dn column already exists - skipping")

        # Modify hashed_password to be nullable (for LDAP-only users)
        # Check current nullability
        result = conn.execute(text("""
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'hashed_password'
        """))
        row = result.fetchone()
        if row and row[0] == 'NO':
            logger.info("Making hashed_password nullable for LDAP users...")
            if not dry_run:
                conn.execute(text("""
                    ALTER TABLE users
                    ALTER COLUMN hashed_password DROP NOT NULL
                """))
            changes_made.append("Made hashed_password nullable")
        else:
            logger.info("hashed_password is already nullable - skipping")

        # Create system_settings table if it doesn't exist
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = 'system_settings'
            )
        """))
        if not result.scalar():
            logger.info("Creating system_settings table...")
            if not dry_run:
                conn.execute(text("""
                    CREATE TABLE system_settings (
                        id SERIAL PRIMARY KEY,
                        key VARCHAR(100) UNIQUE NOT NULL,
                        value JSONB,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by INTEGER REFERENCES users(id)
                    )
                """))
                conn.execute(text("""
                    CREATE INDEX ix_system_settings_key ON system_settings(key)
                """))
            changes_made.append("Created system_settings table")
        else:
            logger.info("system_settings table already exists - skipping")

        if not dry_run:
            conn.commit()

        if changes_made:
            logger.info("Migration completed successfully!")
            logger.info("Changes made:")
            for change in changes_made:
                logger.info(f"  - {change}")
        else:
            logger.info("No changes needed - database is already up to date")


if __name__ == "__main__":
    migrate_ldap()
