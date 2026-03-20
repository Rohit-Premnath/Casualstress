# CausalStress — AI-Powered Financial Crisis Simulator

A unified stress testing platform that combines **causal discovery**, **regime detection**, and **generative AI** to test portfolios against financial crises that have never occurred historically.

## The Problem

Banks stress test portfolios against past crises (2008, COVID). But the next crisis will look different. Current methods rely on correlations which break down during crises — the exact moment they're needed most.

## Our Solution

CausalStress fills a gap explicitly identified by the academic community: no existing system combines causal discovery + regime-aware modeling + generative scenario creation in one framework.

| Layer | What It Does | Technology |
|-------|-------------|------------|
| Causal Discovery | Learns cause-and-effect chains across 50+ economic variables | DYNOTEARS, PCMCI |
| Regime Detection | Classifies market state (calm/stressed/crisis) in real time | Hidden Markov Models |
| Crisis Generation | Creates novel, economically plausible crisis scenarios | Conditional Diffusion Models |
| Portfolio Impact | Computes VaR, CVaR, drawdown for any portfolio | NumPy, SciPy |
| Dashboard | Interactive visualization and stress test interface | React, D3.js |

## Architecture
```
FRED + Yahoo Finance + EDGAR
        │
        ▼
  Data Ingestion & Processing (PostgreSQL + TimescaleDB)
        │
        ▼
  Causal Discovery Engine (DYNOTEARS + PCMCI)
        │
        ▼
  Regime Detection (HMM + Bayesian Changepoint)
        │
        ▼
  Generative Crisis Engine (Conditional Diffusion Model)
        │
        ▼
  Portfolio Impact Engine (VaR, CVaR, Drawdown)
        │
        ▼
  React Dashboard + PDF Reports
```

## Tech Stack

- **Backend:** Python, FastAPI
- **Database:** PostgreSQL + TimescaleDB
- **ML/Causal:** causal-learn, tigramite, hmmlearn, PyTorch
- **Frontend:** React, D3.js, Recharts
- **Data:** FRED API, Yahoo Finance, EDGAR

## Project Structure
```
causalstress/
├── backend/            # FastAPI application
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   ├── models/
│   │   ├── services/
│   │   └── db/
│   ├── requirements.txt
│   └── Dockerfile
├── ml_pipeline/        # ML and data processing
│   ├── data_ingestion/
│   ├── causal_discovery/
│   ├── regime_detection/
│   ├── generative_engine/
│   └── risk_engine/
├── frontend/           # React dashboard
│   └── src/
├── research_paper/     # LaTeX paper
├── docker-compose.yml
└── .env.example
```

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)

### Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/causalstress.git
cd causalstress

# Copy environment file
cp .env.example .env
# Edit .env and add your FRED API key

# Start all services
docker-compose up -d

# The API will be available at http://localhost:8000
# The frontend will be available at http://localhost:3000
```

## Research Paper

**Title:** "Causal Stress Testing: Regime-Aware Scenario Generation Using Causal Discovery and Score-Based Diffusion Models for Financial Risk Assessment"

*Paper in progress — targeting NeurIPS 2026 AI for Finance Workshop*

## License

MIT License — see [LICENSE](LICENSE) for details.
```

---

**Step 4: Update the .gitignore**

The Python .gitignore from GitHub is good but we need to add a few project-specific items. Open `.gitignore` and add these lines at the bottom:
```
# Project specific
.env
*.env.local
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/

# Data files (too large for git)
data/raw/
data/processed/
*.csv
*.parquet
*.h5

# ML model files
models/checkpoints/
*.pt
*.pth
*.pkl

# Node
node_modules/
frontend/dist/
frontend/build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker
docker-compose.override.yml

# OS
.DS_Store
Thumbs.db