# System Overview

## What CausalStress Is

CausalStress is a financial stress-testing system that generates novel crisis scenarios by learning causal relationships between financial variables, conditioning those relationships on market regimes, and propagating shocks through the resulting causal graph. The core claim is that standard stress testing — which either replays historical events or samples from an unconditional distribution — misses crisis dynamics that only emerge when markets are already stressed. CausalStress addresses this by discovering regime-conditional causal structure and building scenarios from that structure.

The system is end-to-end: it ingests raw market data, discovers causal graphs, classifies regimes, generates and filters scenario trajectories, computes portfolio risk metrics, compares projections against regulatory benchmarks, and (in Phase 2B) trains adversarial RL agents to find portfolio-specific worst cases. A FastAPI backend and React frontend expose all of this to end users.

---

## The Four-Layer Pipeline

### Layer 1 — Data Ingestion and Preprocessing

**What it does:** Fetches 56 financial and macroeconomic variables from FRED and Yahoo Finance, aligns them to daily frequency, applies stationarity transforms, and writes to PostgreSQL.

**Implementation:** [ml_pipeline/data_ingestion/](ml_pipeline/data_ingestion/)
- [fred_fetcher.py](ml_pipeline/data_ingestion/fred_fetcher.py) — 35 FRED variables (macro rates, credit spreads, lending standards, inflation)
- [yahoo_fetcher.py](ml_pipeline/data_ingestion/yahoo_fetcher.py) — 21 Yahoo variables (equity indices, sectors, volatility, commodities)
- [data_processor.py](ml_pipeline/data_ingestion/data_processor.py) — stationarity transforms, missing-data handling, database write

**Output:** `processed.time_series_data` TimescaleDB hypertable. 5,548 trading days from 2005-01-04 to 2026-04-14.

---

### Layer 2 — Causal Discovery and Regime Detection

**What it does:** Learns directed acyclic graphs (DAGs) of causal relationships between variables. Runs separately for each market regime so that causal structure reflects how relationships change during crises.

**Two parallel causal discovery methods:**
- **DYNOTEARS** ([ml_pipeline/causal_discovery/dynotears_engine.py](ml_pipeline/causal_discovery/dynotears_engine.py)) — LASSO-regularized VAR with BIC scoring. Produces a sparse adjacency matrix by optimizing over the space of DAGs.
- **PCMCI** ([ml_pipeline/causal_discovery/pcmci_engine.py](ml_pipeline/causal_discovery/pcmci_engine.py)) — conditional independence testing via the tigramite library. Conservative: edges only added when the null hypothesis of no causal link is rejected at p < 0.05.
- Both run at max lag = 5 trading days (one calendar week).
- Ensemble: 255 consensus edges (found by both) + 994 PCMCI-only edges retained = 1,249 total.

**Regime detection:**
- **5-state Gaussian HMM** ([ml_pipeline/regime_detection/hmm_model.py](ml_pipeline/regime_detection/hmm_model.py)) classifies every trading day into calm, normal, elevated, stressed, or crisis.
- **Regime-conditional graph discovery** ([ml_pipeline/regime_detection/regime_causal_graphs.py](ml_pipeline/regime_detection/regime_causal_graphs.py)) re-runs causal discovery separately on data labeled within each regime. The stressed-regime canonical graph has 330 edges. 211 of those edges appear only during stress (contagion edges). 97 edges that are present in calm regimes disappear under stress (decoupling).

**Output:** `models.causal_graphs` and `models.regimes` in PostgreSQL. `regime_causal_graphs.json` on disk.

---

### Layer 3 — Scenario Generation

**What it does:** Given a shock event family (e.g., market crash, credit crisis), generates 200 Monte Carlo trajectories over a 60-day horizon using a regime-conditioned VAR model, causal propagation, and data-fit Student-t innovations. Applies soft plausibility filtering to weight trajectories by statistical coherence.

**Implementation:** [ml_pipeline/generative_engine/scenario_generator.py](ml_pipeline/generative_engine/scenario_generator.py)

**The canonical model** (`causal_regime_multi_root_soft_filtered_ttails_datafit`, locked 2026-04-19):
1. Fit a VAR(2) model on data labeled as elevated, stressed, or crisis regime days.
2. Load the stressed-regime causal graph (330 edges).
3. Initialize shock from a family template (e.g., market_crash: ^GSPC −3.0σ, ^VIX +3.5σ, XLF −3.5σ, HY +3.0σ, DGS10 −1.5σ).
4. Propagate shock 3 hops through the causal graph with 0.4× decay per hop.
5. Add Student-t innovations sampled with data-fit degrees of freedom (df_normal = 5.97, df_crisis = 3.84).
6. Maintain 5-day shock persistence with exponential decay λ = 0.72.
7. Generate 400 candidate trajectories, score each for plausibility, weight by score^6.0, and sample 200.

**Output:** Per-event fan charts (5th, 50th, 95th percentile trajectories across 25 variables × 60 days). Written to `models.scenarios`.

**Also in this layer:**
- [ml_pipeline/generative_engine/vecm_engine.py](ml_pipeline/generative_engine/vecm_engine.py) — VECM for cointegrated variable groups (research branch, not in canonical path)
- [ml_pipeline/generative_engine/copula_engine.py](ml_pipeline/generative_engine/copula_engine.py) — Student-t copula for tail dependence modeling (research branch)

---

### Layer 4 — Risk Metrics, Regulatory Comparison, and Adversarial Testing

**What it does:** Takes scenario trajectories and computes portfolio risk metrics, compares against regulatory stress tests, and (Phase 2B) uses RL agents to find portfolio-specific worst-case shocks.

**Risk engine** ([ml_pipeline/risk_engine/portfolio.py](ml_pipeline/risk_engine/portfolio.py)):
- VaR at 95% and 99%, CVaR, max drawdown, sector decomposition, marginal risk contribution.

**Regulatory engine** ([ml_pipeline/regulatory/regulatory_engine.py](ml_pipeline/regulatory/regulatory_engine.py)):
- DFAST 2026 Severely Adverse scenario: ingested from the official Federal Reserve CSV. Variable-by-variable comparison against causal model projections.
- EBA 2025 Adverse: illustrative only, approximated from published themes. Not a verified benchmark.
- Causal Difference Reports: shows where and by how much the causal model diverges from the regulatory assumption, with causal path explanations.

**Adversarial RL** ([ml_pipeline/generative_engine_rl/](ml_pipeline/generative_engine_rl/)):
- Phase 2B. A 1-step MDP where an agent learns to select the shock (variable, magnitude, event family) that maximizes portfolio loss while respecting causal plausibility constraints.
- Supports 4 portfolio profiles (balanced, tech_heavy, bond_heavy, credit_heavy).
- Pre-trained PPO and bandit models served via REST API.

**Narrative engine** ([ml_pipeline/narrative/narrative_engine.py](ml_pipeline/narrative/narrative_engine.py)):
- Uses Claude (Anthropic API) to generate plain-English explanations of stress test results, causal divergences, and portfolio vulnerabilities.

---

## Infrastructure and Data Flow

```
                         ┌──────────────────────────────┐
FRED API ────────────────►                              │
                         │   data_ingestion/            │
Yahoo Finance ───────────►   fred_fetcher.py            │
                         │   yahoo_fetcher.py           │
                         │   data_processor.py          │
                         └─────────────┬────────────────┘
                                       │ raw_fred / raw_yahoo
                                       ▼
                         ┌──────────────────────────────┐
                         │   PostgreSQL + TimescaleDB   │
                         │   processed.time_series_data │
                         └─────────────┬────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        │
  causal_discovery/           regime_detection/                │
  dynotears_engine.py         hmm_model.py                    │
  pcmci_engine.py             regime_causal_graphs.py         │
       │                            │                          │
       └──────────┬─────────────────┘                         │
                  │                                            │
                  ▼                                            │
     regime_causal_graphs.json ◄─────────────────────────────┘
     models.causal_graphs
     models.regimes
                  │
                  ▼
       generative_engine/
       scenario_generator.py
       (canonical VAR model)
                  │
                  ├──► models.scenarios (PostgreSQL)
                  │
                  ├──► risk_engine/portfolio.py
                  │         (VaR, CVaR, drawdown)
                  │
                  ├──► regulatory/regulatory_engine.py
                  │         (DFAST comparison)
                  │
                  └──► generative_engine_rl/
                            (adversarial worst-case)
                  │
                  ▼
          backend/app/main.py (FastAPI)
          7 routers, 20+ endpoints
                  │
                  ▼
          frontend/ (React + TypeScript)
          6 pages: Dashboard, Causal Graph,
          Regimes, Scenario Lab, Stress Test,
          AI Advisor
```

---

## Component-to-File Map

| Component | Primary Files |
|-----------|--------------|
| Data ingestion | [ml_pipeline/data_ingestion/fred_fetcher.py](ml_pipeline/data_ingestion/fred_fetcher.py), [yahoo_fetcher.py](ml_pipeline/data_ingestion/yahoo_fetcher.py), [data_processor.py](ml_pipeline/data_ingestion/data_processor.py) |
| DYNOTEARS causal discovery | [ml_pipeline/causal_discovery/dynotears_engine.py](ml_pipeline/causal_discovery/dynotears_engine.py) |
| PCMCI causal discovery | [ml_pipeline/causal_discovery/pcmci_engine.py](ml_pipeline/causal_discovery/pcmci_engine.py) |
| Regime detection | [ml_pipeline/regime_detection/hmm_model.py](ml_pipeline/regime_detection/hmm_model.py) |
| Regime-conditional graphs | [ml_pipeline/regime_detection/regime_causal_graphs.py](ml_pipeline/regime_detection/regime_causal_graphs.py) |
| Canonical scenario generator | [ml_pipeline/generative_engine/scenario_generator.py](ml_pipeline/generative_engine/scenario_generator.py) |
| VECM (research branch) | [ml_pipeline/generative_engine/vecm_engine.py](ml_pipeline/generative_engine/vecm_engine.py) |
| Copula (research branch) | [ml_pipeline/generative_engine/copula_engine.py](ml_pipeline/generative_engine/copula_engine.py) |
| Risk metrics | [ml_pipeline/risk_engine/portfolio.py](ml_pipeline/risk_engine/portfolio.py) |
| Regulatory comparison | [ml_pipeline/regulatory/regulatory_engine.py](ml_pipeline/regulatory/regulatory_engine.py) |
| Adversarial RL environment | [ml_pipeline/generative_engine_rl/causal_stress_env.py](ml_pipeline/generative_engine_rl/causal_stress_env.py) |
| RL reward function | [ml_pipeline/generative_engine_rl/rewards.py](ml_pipeline/generative_engine_rl/rewards.py) |
| RL inference server | [ml_pipeline/generative_engine_rl/adversarial_serve.py](ml_pipeline/generative_engine_rl/adversarial_serve.py) |
| Narrative (LLM) | [ml_pipeline/narrative/narrative_engine.py](ml_pipeline/narrative/narrative_engine.py) |
| Locked paper numbers | [ml_pipeline/canonical_paper_numbers.py](ml_pipeline/canonical_paper_numbers.py) |
| Locked model config | [ml_pipeline/canonical_best_model.py](ml_pipeline/canonical_best_model.py) |
| Action space spec | [ml_pipeline/action_space/action_space.yaml](ml_pipeline/action_space/action_space.yaml) |
| FastAPI backend | [backend/app/main.py](backend/app/main.py) |
| Frontend entry | [frontend/src/](frontend/src/) |

---

## Database Schema

Four schemas in PostgreSQL + TimescaleDB:

| Schema | Tables | Purpose |
|--------|--------|---------|
| `raw_fred` | `observations` | FRED API raw staging |
| `raw_yahoo` | `daily_prices` | Yahoo Finance raw staging |
| `processed` | `time_series_data` | Aligned, transformed daily data (TimescaleDB hypertable) |
| `models` | `causal_graphs`, `regimes`, `scenarios` | ML pipeline outputs |
| `regulatory` | `scenarios`, `causal_difference_reports` | Regulatory comparison results |
| `app` | `stress_test_results` | User-facing stress test outputs |

---

## Backend API Surface

The FastAPI backend ([backend/app/main.py](backend/app/main.py)) exposes 7 routers:

| Router | Key Endpoints | Returns |
|--------|---------------|---------|
| `/api/v1/dashboard` | `GET /summary` | Current regime, S&P value, data freshness |
| `/api/v1/causal` | `GET /graph`, `/regime-graph` | Causal graph adjacency + metadata |
| `/api/v1/regimes` | `GET /current`, `/history`, `/statistics` | Regime classifications and transitions |
| `/api/v1/scenarios` | `GET /latest`, `POST /generate` | Scenario fan charts + plausibility |
| `/api/v1/stress-test` | `POST /run` | VaR, CVaR, drawdown, sector decomp |
| `/api/v1/advisor` | `POST /chat` | Claude-powered narrative explanation |
| `/api/v1/adversarial` | `POST /worst-case`, `GET /status` | RL worst-case shock sequence |

---

## Key Design Decisions

**Why VAR not a diffusion model?** VAR is interpretable — coefficients directly represent linear causal influence. This matters because the causal graph must be readable by practitioners and must respect economic priors (e.g., Fed funds rate leads Treasury yields). Diffusion models generate more realistic samples but the latent space cannot be constrained by causal structure.

**Why regime conditioning?** The causal graph changes during stress. Banks' lending-standards contagion, which barely moves in calm markets, amplifies by 6.9× in crisis. A model trained on all regimes together averages away this effect. Regime-conditional training captures it.

**Why Student-t innovations?** Financial returns have fat tails. Pre-2020 VAR residuals reject normality — the Kolmogorov-Smirnov test prefers Student-t for 20/20 core variables in both regime sets. Using MLE-fit degrees of freedom (df_normal = 5.97, df_crisis = 3.84) is a data-driven choice rather than an assumption.

**Why soft plausibility filtering over hard top-k?** Hard top-k discards scenarios below a fixed rank, which creates discontinuities and reduces diversity at the scenario margin. Soft filtering (score^6.0 weighting) retains all scenarios but strongly suppresses implausible ones, preserving tail diversity while pushing probability mass toward coherent trajectories.
