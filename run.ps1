# ==============================================================================
# Financial GraphRAG - PowerShell Startup Script (Windows / macOS / Linux)
# ==============================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Financial GraphRAG: Multi-Year Corporate Intelligence" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
}

# 2. Check .env
if (-not (Test-Path ".env")) {
    Write-Error ".env file not found."
    exit 1
}

# 3. Start Database Containers
Write-Host "`n🚀 [1/4] Starting Database Services (PostgreSQL + pgvector, Neo4j 5)..." -ForegroundColor Green
docker compose up -d financial-postgres financial-neo4j

Write-Host "⏳ [2/4] Waiting for databases to accept connections..." -ForegroundColor Green
Start-Sleep -Seconds 4

# 4. Setup Python Environment
Write-Host "`n🐍 [3/4] Setting up Python virtual environment..." -ForegroundColor Green
if (Test-Path ".venv/Scripts/Activate.ps1") {
    & .venv/Scripts/Activate.ps1
} elseif (Test-Path ".venv/bin/Activate.ps1") {
    & .venv/bin/Activate.ps1
} else {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
    if (Test-Path ".venv/Scripts/Activate.ps1") {
        & .venv/Scripts/Activate.ps1
    }
    pip install -q -r requirements.txt
}

# 5. Initialize Schema
Write-Host "`n⚙️  [4/4] Verifying database schemas and Knowledge Graph..." -ForegroundColor Green
python scripts/init_db.py

# 6. Launch Streamlit UI
Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "🎉 System ready! Launching Streamlit Research Dashboard..." -ForegroundColor Cyan
Write-Host "👉 Web UI: http://localhost:8501" -ForegroundColor Yellow
Write-Host "👉 Neo4j Browser: http://localhost:7474 (user: neo4j / pwd: password123)" -ForegroundColor Yellow
Write-Host "👉 PostgreSQL CLI: docker exec -it financial-postgres psql -U financial_user -d financial_db" -ForegroundColor Yellow
Write-Host "=================================================================`n" -ForegroundColor Cyan

streamlit run ui/app.py
