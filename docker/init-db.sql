-- ==============================================================================
-- Financial GraphRAG: PostgreSQL Initialization Script
-- ==============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- Guarantee financial_user exists with proper permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'financial_user') THEN
        CREATE ROLE financial_user WITH LOGIN SUPERUSER PASSWORD 'password123';
    ELSE
        ALTER ROLE financial_user WITH SUPERUSER PASSWORD 'password123';
    END IF;
END
$$;

-- Grant all permissions on financial_db
GRANT ALL PRIVILEGES ON DATABASE financial_db TO financial_user;
