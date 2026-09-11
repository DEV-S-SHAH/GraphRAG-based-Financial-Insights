#!/usr/bin/env bash
# ==============================================================================
# Financial GraphRAG - Single-Command Startup Script
# ==============================================================================
# This script:
# 1. Verifies prerequisites (Docker, Python 3)
# 2. Ensures .env configuration is present
# 3. Starts PostgreSQL + pgvector and Neo4j via docker compose
# 4. Waits for databases to be healthy and accepting connections
# 5. Activates or sets up Python virtual environment
# 6. Initializes database schemas and loads ground truth if needed
# 7. Launches the Streamlit research dashboard
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "   Financial GraphRAG: Multi-Year Corporate Intelligence"
echo "================================================================="

# 1. Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH."
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon is not running."
    echo "Please start Docker Desktop and run this script again."
    exit 1
fi

# 2. Check .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📋 .env not found; creating .env from .env.example..."
        cp .env.example .env
    else
        echo "❌ Error: .env file not found."
        exit 1
    fi
fi

# Source .env for variables
set -a
source .env
set +a

POSTGRES_DB=${POSTGRES_DB:-financial_db}
POSTGRES_USER=${POSTGRES_USER:-financial_user}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-password123}

# 3. Start Database Containers
echo ""
echo "🚀 Starting Database Services (PostgreSQL + pgvector, Neo4j 5)..."
docker compose up -d financial-postgres financial-neo4j

echo "⏳ Waiting for PostgreSQL (port 5432) to accept connections..."
until docker exec financial-postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1; do
    sleep 1
done
echo "✅ PostgreSQL is ready ($POSTGRES_DB)."

echo "⏳ Waiting for Neo4j Bolt (port 7687) to be available..."
for i in {1..30}; do
    if docker exec financial-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1;" > /dev/null 2>&1; then
        echo "✅ Neo4j is ready."
        break
    fi
    sleep 1
done

# 4. Setup Python Environment
if [ -d ".venv" ]; then
    echo "🐍 Activating existing virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "🐍 Using active virtual environment ($VIRTUAL_ENV)..."
else
    echo "🐍 Creating virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "📦 Installing requirements..."
    pip install -q -r requirements.txt
fi

# 5. Initialize Schema & Knowledge Graph if needed
echo ""
echo "⚙️  Verifying database schemas and Knowledge Graph..."
python3 scripts/init_db.py

# 6. Launch Streamlit UI
echo ""
echo "================================================================="
echo "🎉 System ready! Launching Streamlit Research Dashboard..."
echo "👉 Web UI: http://localhost:8501"
echo "👉 Neo4j Browser: http://localhost:7474 (user: neo4j / pwd: $NEO4J_PASSWORD)"
echo "👉 PostgreSQL CLI: docker exec -it financial-postgres psql -U $POSTGRES_USER -d $POSTGRES_DB"
echo "================================================================="
echo ""

streamlit run ui/app.py
