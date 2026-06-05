# start_local.ps1 — start PostgreSQL (if not running) then the FastAPI backend
# Run from the repo root each time you want to develop.

$PG_HOME = "$env:USERPROFILE\.causalstress\pg16"
$PG_DATA = "$env:USERPROFILE\.causalstress\pgdata"
$PG_LOG  = "$env:USERPROFILE\.causalstress\pg.log"
$PG_PORT = "5433"

if (-not (Test-Path "$PG_HOME\bin\pg_ctl.exe")) {
    Write-Error "PostgreSQL not found. Run .\setup_local.ps1 first."
    exit 1
}

# Start PostgreSQL if not already running
$pgRunning = $false
try {
    $result = & "$PG_HOME\bin\pg_ctl.exe" status -D $PG_DATA 2>&1
    $pgRunning = $result -match "server is running"
} catch {}

if (-not $pgRunning) {
    Write-Host "Starting PostgreSQL on port $PG_PORT ..."
    & "$PG_HOME\bin\pg_ctl.exe" start -D $PG_DATA -l $PG_LOG -w
    Start-Sleep -Seconds 2
} else {
    Write-Host "PostgreSQL already running."
}

# Install Python deps if needed
$fastApiCheck = python -c "import fastapi" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Python dependencies ..."
    python -m pip install -r backend\requirements.txt
}

# Start backend
# PYTHONPATH=backend  → makes `app` importable
# Repo root stays as CWD → ml_pipeline and psycopg2 shim are both findable
Write-Host "Starting CausalStress API at http://localhost:8000 ..."
Write-Host "API docs:  http://localhost:8000/docs"
Write-Host ""
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
