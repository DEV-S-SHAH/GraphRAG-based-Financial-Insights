@echo off
REM ==============================================================================
REM Financial GraphRAG - Single-Command Startup Script for Windows
REM ==============================================================================

echo =================================================================
echo    Financial GraphRAG: Multi-Year Corporate Intelligence (Windows)
echo =================================================================

REM 1. Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

REM 2. Check .env
if not exist .env (
    echo [ERROR] .env file not found.
    pause
    exit /b 1
)

REM 3. Start Database Containers
echo.
echo [1/4] Starting Database Services (PostgreSQL + pgvector, Neo4j 5)...
docker compose up -d financial-postgres financial-neo4j

echo [2/4] Waiting for databases to initialize...
timeout /t 5 /nobreak >nul

REM 4. Setup Python Environment
echo.
echo [3/4] Setting up Python virtual environment...
if exist .venv\Scripts\activate.bat (
    echo Activating existing virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -q -r requirements.txt
)

REM 5. Initialize Schema & Knowledge Graph
echo.
echo [4/4] Verifying database schemas and Knowledge Graph...
python scripts\init_db.py

REM 6. Launch Streamlit UI
echo.
echo =================================================================
echo System ready! Launching Streamlit Research Dashboard...
echo Web UI: http://localhost:8501
echo Neo4j Browser: http://localhost:7474 (user: neo4j / pwd: password123)
echo PostgreSQL CLI: docker exec -it financial-postgres psql -U financial_user -d financial_db
echo =================================================================
echo.

streamlit run ui\app.py
