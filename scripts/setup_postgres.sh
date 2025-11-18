#!/bin/bash
# Setup PostgreSQL database for Zeus

echo "Setting up PostgreSQL database for Zeus..."

# Create database and user
psql postgres <<EOF
-- Create database
CREATE DATABASE zeus_db;

-- Create user with password
CREATE USER zeus_user WITH PASSWORD 'zeus_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE zeus_db TO zeus_user;

-- Connect to zeus_db and grant schema privileges
\c zeus_db
GRANT ALL ON SCHEMA public TO zeus_user;

EOF

echo "PostgreSQL database setup complete!"
echo ""
echo "Database: zeus_db"
echo "User: zeus_user"
echo "Password: zeus_password"
echo ""
echo "Connection string: postgresql://zeus_user:zeus_password@localhost:5432/zeus_db"
