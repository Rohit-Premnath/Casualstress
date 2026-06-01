# CausalStress Setup Guide

This file shows how to get this project running from zero.

It is written in a very simple way on purpose.

If you follow the steps in order, you should be able to get the app running.

## What this project is

This repo has 4 main parts:

1. A database (`Postgres`)
2. A cache/helper service (`Redis`)
3. A backend API (`FastAPI`)
4. A frontend website (`React + Vite`)

There is also a big `ml_pipeline` folder.

That pipeline fills the database with real data.

Important:
If you only start the database and backend, the app may open, but a lot of pages will be empty.
To make the app actually useful, you also need to run the pipeline steps.

## The short version

If you want the fastest summary, this is the order:

1. Install tools
2. Clone the repo
3. Create `.env`
4. Start Docker services
5. Create Python virtual environment
6. Install Python packages
7. Run the ML pipeline scripts
8. Start the backend
9. Start the frontend
10. Open the app in your browser

## What you need first

Please install these tools before doing anything else:

- `Git`
- `Docker Desktop`
- `Python 3.11`
- `Node.js 18+` or `Node.js 20+`

Helpful but optional:

- A code editor like `VS Code`

## Where to run commands

Open `PowerShell`.

Most commands below should be run from the project folder:

```powershell
cd "C:\path\to\causalstress"
```

In this repo, that folder is:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress"
```

## Step 1: Clone the repo

If you already have the folder, you can skip this.

```powershell
git clone <your-repo-url> causalstress
cd causalstress
```

## Step 2: Create the environment file

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Now open `.env` and make it look like this:

```env
POSTGRES_USER=causalstress
POSTGRES_PASSWORD=causalstress_dev_2026
POSTGRES_DB=causalstress
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

REDIS_HOST=localhost
REDIS_PORT=6379

API_HOST=0.0.0.0
API_PORT=8000

FRED_API_KEY=your_fred_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

ENV=development
CORS_ORIGINS=http://localhost:8080,http://localhost:5173,http://127.0.0.1:5173
```

### What these mean

- `FRED_API_KEY` is needed for pulling economic data
- `ANTHROPIC_API_KEY` is optional unless you want the AI advisor chat to work for real
- `POSTGRES_HOST=localhost` and `POSTGRES_PORT=5433` are important when you run Python on your own computer instead of inside Docker

## Step 3: Start the database and Redis

Start Docker Desktop first.

Then run:

```powershell
docker compose up -d db redis
```

This starts:

- Postgres on port `5433`
- Redis on port `6379`

If you also want Docker to run the backend, you can do this:

```powershell
docker compose up -d --build backend
```

But for beginners, running the backend locally is easier to debug.

## Step 4: Create a Python virtual environment

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

When the virtual environment is active, you usually see `(.venv)` at the start of the line.

## Step 5: Install Python packages

First upgrade pip:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Then install the backend packages:

```powershell
pip install -r backend\requirements.txt
```

Then install the extra ML packages used by the pipeline:

```powershell
pip install scikit-learn hmmlearn causal-learn tigramite apscheduler stable-baselines3
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Why extra installs are needed

`backend/requirements.txt` is not the whole story.

The repo also uses extra packages inside `ml_pipeline`, like:

- `scikit-learn`
- `hmmlearn`
- `causal-learn`
- `tigramite`
- `apscheduler`
- `stable-baselines3`
- `torch`

Without those, parts of the pipeline will fail.

## Step 6: Fill the database with real data

This is the part many people miss.

The app expects real data inside the database.

Run these commands one by one from the repo root while your virtual environment is active:

```powershell
python ml_pipeline\data_ingestion\fred_fetcher.py
python ml_pipeline\data_ingestion\yahoo_fetcher.py
python ml_pipeline\data_ingestion\data_processor.py
python ml_pipeline\regime_detection\hmm_model.py
python ml_pipeline\causal_discovery\dynotears_engine.py
python ml_pipeline\regime_detection\regime_causal_graphs.py
python ml_pipeline\generative_engine\scenario_generator.py
```

### What each one does

- `fred_fetcher.py` downloads macroeconomic data from FRED
- `yahoo_fetcher.py` downloads market data from Yahoo Finance
- `data_processor.py` cleans and transforms the data
- `hmm_model.py` finds market regimes like calm or crisis
- `dynotears_engine.py` creates a causal graph
- `regime_causal_graphs.py` creates different causal graphs for different regimes
- `scenario_generator.py` generates the starter stress scenarios used by the app

Important:
Some of these steps can take a while.
That is normal.

## Step 7: Start the backend

Open a new PowerShell terminal.

Go to the project folder:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress"
.\.venv\Scripts\Activate.ps1
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If it starts correctly, the backend should be available at:

`http://localhost:8000`

You can also open:

`http://localhost:8000/docs`

That page shows the API documentation.

## Step 8: Start the frontend

Open another new PowerShell terminal.

Run:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress\frontend"
npm install
npm run dev
```

The frontend should run at:

`http://localhost:8080`

Important:
This repo uses port `8080` for the frontend, not `3000`.

## Step 9: Open the app

Open these in your browser:

- Frontend: `http://localhost:8080`
- Backend health check: `http://localhost:8000/health`
- Backend docs: `http://localhost:8000/docs`

## Step 10: Check that it really works

Here are good signs:

- `http://localhost:8000/health` shows a healthy response
- `http://localhost:8000/docs` opens
- `http://localhost:8080` opens the website
- Dashboard pages show data instead of being blank

You can also test this API route:

`http://localhost:8000/api/v1/dashboard/summary`

If that returns real JSON with regime and system info, things are in pretty good shape.

## Super simple beginner flow

If you want the easiest possible path, use this exact checklist:

1. Install `Git`, `Docker Desktop`, `Python 3.11`, `Node.js`
2. Copy `.env.example` to `.env`
3. Put your `FRED_API_KEY` into `.env`
4. Run `docker compose up -d db redis`
5. Create and activate `.venv`
6. Install Python packages
7. Run all 7 Python pipeline scripts
8. Start backend with `uvicorn`
9. Start frontend with `npm run dev`
10. Open `http://localhost:8080`

## Common mistakes

### Mistake 1: Forgetting the pipeline

If you skip the ML pipeline scripts, the app may start but have missing or empty data.

### Mistake 2: Wrong database host/port

If Python is running on your computer and Postgres is in Docker, use:

- `POSTGRES_HOST=localhost`
- `POSTGRES_PORT=5433`

### Mistake 3: Using the wrong frontend port

This frontend runs on:

`http://localhost:8080`

Not `3000`.

### Mistake 4: Forgetting to activate the virtual environment

If `python` cannot find installed packages, make sure `(.venv)` is active.

### Mistake 5: No FRED API key

Without a real `FRED_API_KEY`, the FRED fetcher will fail.

## If you want automatic daily refreshes later

The project has a scheduler:

```powershell
python ml_pipeline\scheduler.py --mode run-now
```

That runs the fetch/process/regime steps in one go.

Important:
It does not fully replace every heavy pipeline step for a full fresh rebuild, but it is useful later.

## If you want to run tests

Backend tests:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress\backend"
..\ .venv\Scripts\python.exe -m pytest tests\test_regressions.py
```

If the path above gives you trouble in PowerShell, use this cleaner version from the repo root:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress"
.\.venv\Scripts\Activate.ps1
cd backend
pytest tests\test_regressions.py
```

Frontend tests:

```powershell
cd "C:\Users\megha\OneDrive\Desktop\Casual Stress\Casualstress\frontend"
npm test
```

## If something breaks

Check these things first:

1. Is Docker Desktop running?
2. Is `.env` filled in correctly?
3. Is the Python virtual environment active?
4. Did you install both backend and ML dependencies?
5. Did you run the pipeline scripts in order?
6. Is the backend running on `8000`?
7. Is the frontend running on `8080`?

## Useful file locations

- Main backend app: `backend/app/main.py`
- Backend config: `backend/app/config.py`
- Frontend API client: `frontend/src/services/api.ts`
- Docker services: `docker-compose.yml`
- Database setup SQL: `backend/db_init/01_init.sql`
- Scheduler: `ml_pipeline/scheduler.py`

## Final sanity check

When everything is working, you should have:

- Docker running `db` and `redis`
- Python backend running on `http://localhost:8000`
- React frontend running on `http://localhost:8080`
- Real data inside the database
- Scenarios already generated

Then the app should feel alive instead of empty.

## One-command memory helper

These are the main commands again:

```powershell
docker compose up -d db redis

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r backend\requirements.txt
pip install scikit-learn hmmlearn causal-learn tigramite apscheduler stable-baselines3
pip install torch --index-url https://download.pytorch.org/whl/cpu

python ml_pipeline\data_ingestion\fred_fetcher.py
python ml_pipeline\data_ingestion\yahoo_fetcher.py
python ml_pipeline\data_ingestion\data_processor.py
python ml_pipeline\regime_detection\hmm_model.py
python ml_pipeline\causal_discovery\dynotears_engine.py
python ml_pipeline\regime_detection\regime_causal_graphs.py
python ml_pipeline\generative_engine\scenario_generator.py

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

cd frontend
npm install
npm run dev
```

If you get stuck, go back to the top and do the steps slowly in order.
