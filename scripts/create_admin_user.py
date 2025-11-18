"""
Create an admin user for Zeus.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from loguru import logger
from src.database.database import SessionLocal
from src.database.models import User
from src.database.auth import get_password_hash

def create_admin_user(
    username: str = "admin",
    email: str = "admin@zeus.local",
    password: str = "admin123",
    full_name: str = "Zeus Administrator"
):
    """Create an admin user."""
    db: Session = SessionLocal()

    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            # Update to admin if not already
            if not existing_user.is_admin:
                existing_user.is_admin = True
                db.commit()
                logger.info(f"✓ User '{username}' updated to admin")
            else:
                logger.info(f"Admin user '{username}' already exists")
            return existing_user

        # Create new admin user
        hashed_password = get_password_hash(password)
        admin_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_admin=True
        )

        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        logger.info(f"✓ Created admin user: {username}")
        logger.info(f"  Email: {email}")
        logger.info(f"  Password: {password}")
        logger.info(f"  ⚠️  IMPORTANT: Change this password after first login!")

        return admin_user

    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create admin user for Zeus")
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument("--email", default="admin@zeus.local", help="Admin email")
    parser.add_argument("--password", default="admin123", help="Admin password")
    parser.add_argument("--full-name", default="Zeus Administrator", help="Full name")

    args = parser.parse_args()

    create_admin_user(
        username=args.username,
        email=args.email,
        password=args.password,
        full_name=args.full_name
    )
