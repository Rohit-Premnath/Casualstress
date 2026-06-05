# setup_local.ps1 — one-time setup: portable PostgreSQL + CausalStress database
# No admin rights required. Run from the repo root.

$ErrorActionPreference = "Stop"

$PG_HOME  = "$env:USERPROFILE\.causalstress\pg16"
$PG_DATA  = "$env:USERPROFILE\.causalstress\pgdata"
$PG_LOG   = "$env:USERPROFILE\.causalstress\pg.log"
$PG_PORT  = "5433"
$DB_USER  = "causalstress"
$DB_PASS  = "causalstress_dev_2026"
$DB_NAME  = "causalstress"

function Pg { & "$PG_HOME\bin\$args" }

# ── Step 1: Get PostgreSQL binaries ──────────────────────────────────────────
if (-not (Test-Path "$PG_HOME\bin\postgres.exe")) {
    Write-Host ""
    Write-Host "PostgreSQL binaries not found at $PG_HOME"
    Write-Host ""
    Write-Host "Download the Windows x64 binaries ZIP (not the installer) from:"
    Write-Host "  https://www.enterprisedb.com/download-postgresql-binaries"
    Write-Host ""
    Write-Host "Choose PostgreSQL 16.x, Windows x86-64, then click 'Download'."
    Write-Host "Save the zip anywhere, then paste the full path below."
    Write-Host ""
    $zipPath = Read-Host "Path to downloaded zip (e.g. C:\Users\you\Downloads\postgresql-16.6-1-windows-x64-binaries.zip)"

    if (-not (Test-Path $zipPath)) {
        Write-Error "File not found: $zipPath"
        exit 1
    }

    Write-Host "Extracting to $PG_HOME ..."
    New-Item -ItemType Directory -Force "$env:USERPROFILE\.causalstress" | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath "$env:USERPROFILE\.causalstress\pg16-extract" -Force
    # The zip contains a 'pgsql' folder
    $extracted = Get-ChildItem "$env:USERPROFILE\.causalstress\pg16-extract" | Select-Object -First 1
    Move-Item $extracted.FullName $PG_HOME -Force
    Remove-Item "$env:USERPROFILE\.causalstress\pg16-extract" -Recurse -Force
    Write-Host "Extracted OK."
}

# ── Step 2: Initialise cluster (only once) ────────────────────────────────────
if (-not (Test-Path "$PG_DATA\PG_VERSION")) {
    Write-Host "Initialising database cluster at $PG_DATA ..."
    New-Item -ItemType Directory -Force $PG_DATA | Out-Null
    & "$PG_HOME\bin\initdb.exe" -D $PG_DATA -U $DB_USER --auth=trust --encoding=UTF8
    # Set custom port so it doesn't clash with any system PostgreSQL
    (Get-Content "$PG_DATA\postgresql.conf") -replace "#port = 5432", "port = $PG_PORT" | Set-Content "$PG_DATA\postgresql.conf"
    Write-Host "Cluster initialised."
} else {
    Write-Host "Cluster already exists at $PG_DATA"
}

# ── Step 3: Start PostgreSQL ──────────────────────────────────────────────────
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

# ── Step 4: Create database and user ─────────────────────────────────────────
$env:PGPORT = $PG_PORT
$env:PGHOST = "localhost"

$userExists = & "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" 2>$null
if ($userExists -ne "1") {
    Write-Host "Creating database user '$DB_USER' ..."
    & "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
}

$dbExists = & "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>$null
if ($dbExists -ne "1") {
    Write-Host "Creating database '$DB_NAME' ..."
    & "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
}

# ── Step 5: Run schema migrations ────────────────────────────────────────────
Write-Host "Running schema init ..."
& "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d $DB_NAME -f "backend\db_init\01_init.sql"
& "$PG_HOME\bin\psql.exe" -U $DB_USER -p $PG_PORT -d $DB_NAME -f "backend\db_init\02_runtime_schema_compat.sql"

# ── Step 6: Write .env ────────────────────────────────────────────────────────
$envContent = @"
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASS
POSTGRES_DB=$DB_NAME
POSTGRES_HOST=localhost
POSTGRES_PORT=$PG_PORT

REDIS_HOST=localhost
REDIS_PORT=6379

API_HOST=0.0.0.0
API_PORT=8000

FRED_API_KEY=
ANTHROPIC_API_KEY=

ENV=development
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173
"@

Set-Content -Path ".env" -Value $envContent -Encoding utf8
Write-Host ""
Write-Host "Setup complete!"
Write-Host "  .env written with POSTGRES_PORT=$PG_PORT"
Write-Host "  Add your FRED_API_KEY and ANTHROPIC_API_KEY to .env if needed."
Write-Host ""
Write-Host "To start the backend, run:  .\start_local.ps1"
