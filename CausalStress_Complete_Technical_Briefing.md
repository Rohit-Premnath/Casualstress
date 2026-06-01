# CausalStress Complete Technical Briefing

Prepared on May 27, 2026.

This briefing is intended to be a current, end-to-end technical explanation of the CausalStress system as it exists in this repository today.

It is based on:

- direct code inspection across `backend/`, `frontend/`, and `ml_pipeline/`
- verification of recent schema and API fixes
- backend regression tests
- frontend build and test checks
- Docker Compose config validation and backend container build validation
- inspection of saved ML and RL artifacts in `ml_pipeline/runs/`

This document is meant to help someone reach a complete practical understanding of:

- what the product does
- how the runtime system is structured
- how the modeling pipeline works
- what is fully implemented versus still prototype-grade
- what has recently been verified versus what remains a caveat

It does **not** pretend that every generated artifact, every legacy experiment script, or every saved run directory is equally important to understanding the app. The goal is to explain the real system clearly.

---

## 1. Executive Summary

### 1.1 What CausalStress is

**CausalStress** is a full-stack financial stress-testing platform that combines:

- macro and market data ingestion
- causal discovery
- market regime detection
- regime-aware scenario generation
- portfolio stress testing
- regulatory scenario comparison
- AI-assisted narrative explanation
- adversarial portfolio-specific worst-case scenario search

In practical terms, the system is trying to answer:

- what is driving financial markets right now
- how those relationships change during stressed regimes
- what plausible crisis paths could emerge from today’s state
- how a given portfolio would perform under those paths
- what the most damaging plausible shock sequences are for that portfolio

### 1.2 What makes it different

The differentiator is **not** the dashboard alone. It is the modeling stack:

- causal graphs instead of only correlation structures
- regime-conditional graphs instead of one fixed network
- generated stressed scenarios instead of only historical replay
- portfolio-specific adversarial search instead of only static scenario sets

### 1.3 Best one-sentence description

> CausalStress is a regime-aware causal stress-testing platform that turns historical macro-financial data into explainable crisis scenarios, portfolio loss analysis, regulatory comparison reports, and ranked portfolio-specific adversarial stress paths.

---

## 2. What the System Does End to End

The cleanest way to understand the system is as a ten-stage flow.

### 2.1 Stage 1: ingest raw data

The pipeline collects:

- macro, rates, credit, liquidity, and financial-condition data from FRED
- market and asset proxy data from Yahoo Finance

These are written into raw database tables.

### 2.2 Stage 2: clean and align the data

The data processor:

- aligns everything to trading days
- forward-fills lower-frequency macro series
- interpolates small gaps
- transforms prices into return-like forms
- differences rates/spreads where appropriate
- creates a processed time-series matrix for downstream modeling

The cleaned output lands in `processed.time_series_data`.

### 2.3 Stage 3: learn causal structure

The system uses two causal-discovery approaches:

- a DYNOTEARS/Lasso-style sparse temporal graph search
- a PCMCI conditional-independence-based causal graph

Those are combined into an ensemble graph, with stronger trust placed on edges supported by both methods.

### 2.4 Stage 4: detect market regimes

A Gaussian Hidden Markov Model labels the market state over time using a compact set of market stress indicators.

The regime system is meant to distinguish conditions such as:

- calm / normal
- elevated
- stressed
- crisis-like

Exact label strings vary across pipeline artifacts, but the key idea is that the system maintains discrete latent market-state categories and confidence scores for each date.

### 2.5 Stage 5: rebuild graphs by regime

This is one of the central ideas in the repo.

Instead of assuming one static causal graph for all historical periods, the system rebuilds graphs within each regime so that:

- some links only appear during stress
- contagion pathways can change by market state
- crisis transmission is not forced to look like normal times

### 2.6 Stage 6: generate stress scenarios

The canonical scenario generator:

- fits a stressed-regime VAR-like model
- uses event-family shock templates
- propagates shocks through the stressed-regime causal graph
- applies heavy-tailed innovations
- scores scenario plausibility
- stores generated scenario sets

This is the engine behind the core scenario generation surface.

### 2.7 Stage 7: stress test portfolios

The system maps holdings to market variables and computes:

- VaR
- CVaR
- path-wise max drawdown
- loss probabilities
- sector decomposition
- holding-level contribution summaries

This exists both:

- in pipeline/offline form
- and as a live backend API for the frontend

### 2.8 Stage 8: compare to regulatory scenarios

The system stores regulatory scenarios and can generate difference reports between:

- official supervisory paths
- internally generated causal stress paths

This lets the system show where the causal model agrees or diverges from regulator assumptions.

### 2.9 Stage 9: expose all of this through APIs and UI

The FastAPI backend serves:

- dashboard summaries
- causal graph visualizations
- regime summaries
- generated scenario data
- portfolio stress-test results
- AI advisor interactions
- adversarial ranked scenario search

The React frontend is the interactive interface over those APIs.

### 2.10 Stage 10: adversarial search for worst plausible stress paths

The adversarial layer is the most advanced part of the product prototype.

It uses learned reward-model-based search to find:

- the most damaging plausible stress sequences for a vulnerability profile
- or, via holdings fingerprinting, the closest matching profile for a user portfolio

The output is not just one scalar loss. It can return a ranked list of distinct adversarial pathways.

---

## 3. Current System Boundaries

### 3.1 What the app is

This is a **real, working, local full-stack prototype** with:

- database-backed state
- backend APIs
- a production-buildable frontend
- an ML pipeline with saved artifacts
- an adversarial search subsystem with saved models

### 3.2 What the app is not

This is **not yet** a hardened enterprise product. It still lacks:

- authentication and authorization
- tenant isolation
- proper background job orchestration
- a robust migration chain like a mature Alembic workflow
- deep frontend test coverage
- broad backend API test coverage
- production-grade service decomposition

### 3.3 Best maturity description

The most accurate description is:

> a serious research/product prototype with real end-to-end functionality, real modeling depth, and visible operational hardening debt

---

## 4. Runtime Architecture

### 4.1 Main runtime components

| Layer | Technology | Role |
|---|---|---|
| Frontend | React + Vite + TypeScript + Tailwind | UI and user workflows |
| Backend API | FastAPI | Serves live DB-backed and model-backed responses |
| Database | PostgreSQL + TimescaleDB | Stores raw, processed, model, app, and regulatory outputs |
| Cache / queue placeholder | Redis | Configured in Compose, minimally used in current code |
| ML pipeline | Python scripts | Ingestion, modeling, generation, regulatory comparison, RL/adversarial work |

### 4.2 Docker/runtime setup

`docker-compose.yml` defines:

- `db`: TimescaleDB/Postgres
- `redis`: Redis
- `backend`: FastAPI container

Important runtime facts:

- the frontend is **not** containerized in the current Compose file
- the frontend is intended to run separately under Vite
- the backend image now builds from repo root context and copies both `backend` and `ml_pipeline`

### 4.3 Backend startup behavior

The backend is not a thin stateless CRUD server.

At startup, [main.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/main.py):

- creates the FastAPI app
- configures CORS from environment-backed settings
- registers all routers
- adds request-duration logging middleware
- preloads adversarial models through `ml_pipeline.generative_engine_rl.adversarial_serve.load_all_engines`

That adversarial preload means backend startup is heavier than a typical API.

### 4.4 Configuration model

[config.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/config.py) manages:

- database settings
- Redis settings
- API host/port
- `FRED_API_KEY`
- `ENV`
- `CORS_ORIGINS`

CORS is no longer hardcoded only to local defaults in the router layer; it is now centrally configurable.

---

## 5. Database and Schema Model

### 5.1 Main schemas

The repository now explicitly supports these schemas in visible initialization SQL:

- `raw_fred`
- `raw_yahoo`
- `processed`
- `models`
- `app`
- `regulatory`

### 5.2 Core tables

Important tables include:

- `raw_fred.observations`
- `raw_yahoo.daily_prices`
- `processed.time_series_data`
- `models.causal_graphs`
- `models.regimes`
- `models.scenarios`
- `app.stress_test_results`
- `regulatory.scenarios`
- `regulatory.causal_difference_reports`

### 5.3 Recent schema normalization status

This part changed materially in the latest verified fix pass.

The current checked-in SQL now explicitly supports:

- `models.scenarios.event_type`
- `models.scenarios.anchor_variable`
- `models.scenarios.regime_condition` as `VARCHAR(50)`
- `app.stress_test_results.scenario_id`
- `regulatory` schema bootstrap

Relevant files:

- [01_init.sql](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/db_init/01_init.sql)
- [02_runtime_schema_compat.sql](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/db_init/02_runtime_schema_compat.sql)

### 5.4 Practical schema maturity note

The schema is in a better state than before, but it is still not a fully mature migration system.

Most accurate wording:

- the visible bootstrap SQL now matches runtime assumptions much better
- compatibility SQL exists for in-place schema repair
- but the project still does not present a full migration discipline comparable to a mature production stack

---

## 6. Backend Deep Dive

### 6.1 Router inventory

The backend routers are:

- [dashboard.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/dashboard.py)
- [causal.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/causal.py)
- [regimes.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/regimes.py)
- [scenarios.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py)
- [stress_test.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py)
- [advisor.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/advisor.py)
- [adversarial.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/adversarial.py)

### 6.2 Root and health endpoints

The backend exposes:

- `/`
- `/live`
- `/health`

Behavior:

- `/live` is a lightweight liveness probe that does not require DB access
- `/health` performs a real DB connectivity check

### 6.3 Dashboard router

Role:

- executive summary of the system

Returns:

- current regime
- regime streak
- regime probabilities
- system counts
- latest S&P value
- recent S&P history
- simplified regime chart
- top causal links

Important nuance:

- the dashboard now correctly returns `confidence: null` for top causal links when confidence is unavailable
- it no longer fabricates `100%`

### 6.4 Causal router

Role:

- serves the causal graph exploration page

It provides:

- nodes
- edges
- graph stats
- global and regime-aware graph views

If a regime-specific graph is unavailable, it can fall back to a preferred global graph representation.

### 6.5 Regimes router

Role:

- serves current regime and historical regime views

It provides:

- current regime summary
- timeline compression
- transition matrix
- regime characteristics

### 6.6 Scenarios router

Role:

- scenario metadata
- latest scenario retrieval
- recent scenario list
- live scenario generation

This is one of the central product routers.

Current important behavior:

- scenario families are explicit
- family-specific latest retrieval now uses normalized schema fields
- invalid scenario-family generation now raises real HTTP 400 errors
- response metadata reflects normalized `event_type` and `anchor_variable`

### 6.7 Stress-test router

Role:

- runs a portfolio against stored scenario paths

Current behavior includes:

- scenario lookup
- holdings mapping
- VaR/CVaR calculations
- path-wise max drawdown
- contribution summaries
- result persistence to `app.stress_test_results`

Important recent behavior changes:

- unsupported holdings now return HTTP 400 instead of being silently ignored
- `scenario_id` is now persisted with results
- `maxDrawdown` is derived from path-wise portfolio drawdown rather than worst terminal P&L only
- missing/invalid requests use real HTTP errors instead of fake success-style JSON

### 6.8 Advisor router

Role:

- frontend chat bridge into the narrative/advisor engine

Important operational characteristics:

- it depends on `ANTHROPIC_API_KEY`
- it uses a process-local cached `AdvisorChat` object
- conversation state is memory-local, not durable multi-tenant chat state
- it is tool-using and database-backed rather than a static FAQ bot

### 6.9 Adversarial router

Role:

- expose the adversarial search layer

Endpoints:

- `POST /api/v1/adversarial/worst-case`
- `POST /api/v1/adversarial/ranked-scenarios`
- `GET /api/v1/adversarial/status`

Capabilities:

- fixed vulnerability-profile inference
- holdings-based profile fingerprinting
- ranked scenario search
- causal pathway labels
- model availability reporting

### 6.10 Backend strengths

- clear router separation
- database-backed responses for core product surfaces
- live health and liveness endpoints
- real adversarial model preload
- recent correctness fixes now covered by regression tests

### 6.11 Backend weaknesses

- synchronous `psycopg2` usage inside async handlers
- repeated local `get_conn()` patterns instead of a single DB abstraction
- no auth, tenancy, or role model
- placeholder backend packages (`db`, `models`, `services`) are largely unused
- lightweight migration story compared with mature production apps

---

## 7. Frontend Deep Dive

### 7.1 Frontend shape

The frontend is a Vite React TypeScript app using:

- `react-router-dom`
- `@tanstack/react-query`
- Tailwind
- shadcn/Radix UI primitives
- Recharts
- D3
- Framer Motion

### 7.2 App shell

[App.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/App.tsx) wires:

- React Query provider
- theme provider
- tooltip provider
- toaster providers
- layout wrapper
- route map

Routes:

- `/`
- `/causal-graph`
- `/regimes`
- `/scenarios`
- `/stress-test`
- `/ai-advisor`
- `/settings`
- fallback `*`

### 7.3 Important frontend pages

#### Dashboard: [Index.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/Index.tsx)

Shows:

- current regime
- system counts
- S&P history
- regime chart
- top causal links
- quick navigation

Important nuance:

- the “LIVE / PARTIAL / OFFLINE” indicator is still a surface-level connectivity summary, not a fully hardened data-freshness guarantee
- this remains a known caveat

#### Causal graph: [CausalGraph.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/CausalGraph.tsx)

Purpose:

- deep network exploration

Features include:

- regime filtering
- compare mode
- search
- hover and detail panes
- strong-link views
- transmission-path exploration

This is the most interaction-heavy page in the app.

#### Regimes: [Regimes.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/Regimes.tsx)

Shows:

- current regime
- historical regime strip/timeline
- transition information
- regime characteristics

#### Scenario lab: [ScenarioLab.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/ScenarioLab.tsx)

Purpose:

- create and inspect generated stress scenarios

User controls include:

- family
- severity
- horizon
- optional anchor override
- optional seed

Displays:

- scenario metadata
- shock template
- scenario distribution/fan-chart-like outputs
- key variable stress range

#### Stress test: [StressTest.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/StressTest.tsx)

Purpose:

- portfolio loss analysis
- plus adversarial ranked scenario search

Notable behavior:

- runs canonical stress testing and adversarial ranked search concurrently
- supports preset and custom portfolios
- shows profile fingerprint and ranked adversarial scenarios

This is the clearest business-value screen in the app.

#### AI advisor: [AIAdvisor.tsx](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/AIAdvisor.tsx)

Purpose:

- chat interface over the system state and tools

#### Settings: placeholder route

`/settings` currently renders a “Coming Soon” placeholder rather than a functional settings module.

### 7.4 API client

[api.ts](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/services/api.ts) is the typed frontend bridge to the backend.

Important notes:

- `VITE_API_URL` is optional
- Vite proxying can handle `/api`, `/health`, and `/live` in local development
- the API layer now supports `confidence: number | null` for top causal links

### 7.5 Frontend strengths

- clean route structure
- typed API layer
- strong visual polish on primary pages
- real integration with backend for core product surfaces

### 7.6 Frontend weaknesses

- bundle is large
- automated frontend testing is still extremely shallow
- some mock/demo scaffolding remains in the repo
- some placeholder or decorative UI elements are not fully productized
- lint debt still exists outside the specific bugfix pass

---

## 8. ML Pipeline Deep Dive

The ML pipeline is the real heart of the project.

### 8.1 Package structure

The important ML pipeline subdirectories are:

- `data_ingestion/`
- `causal_discovery/`
- `regime_detection/`
- `generative_engine/`
- `risk_engine/`
- `regulatory/`
- `narrative/`
- `action_space/`
- `generative_engine_rl/`
- `runs/`

### 8.2 Data ingestion

#### [yahoo_fetcher.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/data_ingestion/yahoo_fetcher.py)

Downloads market data and stores it in `raw_yahoo.daily_prices`.

#### [fred_fetcher.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/data_ingestion/fred_fetcher.py)

Downloads macro, credit, labor, rates, and financial-condition data into `raw_fred.observations`.

#### [data_processor.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/data_ingestion/data_processor.py)

Builds the aligned processed matrix used downstream.

### 8.3 Causal discovery

#### [dynotears_engine.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/causal_discovery/dynotears_engine.py)

Sparse temporal causal graph estimation.

#### [pcmci_engine.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/causal_discovery/pcmci_engine.py)

Conditional-independence-based causal discovery used for cross-checking and ensembling.

### 8.4 Regime detection

#### [hmm_model.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regime_detection/hmm_model.py)

Learns hidden market regimes and their probabilities over time.

#### [regime_causal_graphs.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regime_detection/regime_causal_graphs.py)

Rebuilds graphs by regime and is central to the “structure changes across market states” thesis.

### 8.5 Canonical scenario generation

#### [canonical_best_model.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/canonical_best_model.py)

The locked “best model” configuration for canonical scenario generation.

#### [scenario_generator.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine/scenario_generator.py)

The main generation engine.

Its current role includes:

- regime-conditioned VAR-style fitting
- event-family shock templates
- multi-hop causal propagation
- shock persistence
- heavy-tailed innovations
- plausibility scoring
- DB scenario storage

Important recent storage normalization:

- `event_type`
- `anchor_variable`
- non-overloaded `shock_variable`

### 8.6 Risk engine

#### [portfolio.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/risk_engine/portfolio.py)

Offline/batch stress-testing logic used for pipeline evaluation and reporting.

### 8.7 Regulatory engine

#### [regulatory_engine.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regulatory/regulatory_engine.py)

Compares regulatory scenarios to causal-model projections.

The repo includes official supervisory CSVs and regulatory difference-report logic.

### 8.8 Narrative engine

#### [advisor_engine.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/narrative/advisor_engine.py)

Implements the tool-using advisor logic.

Important current behavior:

- it now prefers the normalized scenario schema for lookups
- it can still fall back to legacy-style lookup behavior where needed

---

## 9. Adversarial / RL Layer

### 9.1 What this subsystem is for

The adversarial layer is the system’s portfolio-specific worst-case search module.

Instead of asking only:

- “how bad is this portfolio under the latest scenario set?”

it also asks:

- “what are the most damaging plausible shock sequences for this type of portfolio?”

### 9.2 Current architecture

The main runtime-facing adversarial files are:

- [adversarial_serve.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine_rl/adversarial_serve.py)
- [causal_stress_env.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine_rl/causal_stress_env.py)
- [neural_bandit.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine_rl/neural_bandit.py)
- [portfolio_model.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine_rl/portfolio_model.py)
- [sequence_compare.py](C:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine_rl/sequence_compare.py)

### 9.3 Current conceptual framing

The search problem is a short-horizon sequential decision problem.

The implementation behaves as:

- a finite-horizon `T=2` sequential stress-search problem
- solved with a learned reward model and greedy per-step argmax rollout

This is more precise than calling it a pure classical contextual bandit.

### 9.4 Runtime-facing product use

The user-facing product flow is:

1. user submits holdings
2. portfolio exposures are fingerprinted
3. the nearest vulnerability profile is inferred
4. the corresponding model is used to search across starting states
5. the system returns ranked distinct adversarial scenarios

### 9.5 Current vulnerability profiles

The live product uses four profiles:

- `balanced`
- `tech_heavy`
- `bond_heavy`
- `credit_heavy`

### 9.6 Current product maturity of the adversarial layer

The adversarial layer is real and integrated, but its benchmark quality is uneven across profiles.

Current benchmark headline numbers in the repo’s saved results are:

- `balanced`: 62.6% of beam reference
- `tech_heavy`: 77.1%
- `bond_heavy`: 86.4%
- `credit_heavy`: 69.6%

Important interpretation:

- all four models load and serve
- `bond_heavy` is the strongest profile and clears the project’s internal 85% gate
- the other profiles are useful but remain below that benchmark

### 9.7 Product framing that best fits the current system

The best product use of RL/adversarial search is:

> a ranked portfolio-specific adversarial scenario search layer

not:

- replacing the canonical scenario engine
- generating all stress scenarios by itself
- claiming uniform quality across all portfolio types

---

## 10. Verification Status as of This Briefing

This section is important because the repo has changed recently.

### 10.1 Recently verified backend/data-model fixes

The following were verified in code and by regression checks:

- normalized scenario schema support is present in checked-in SQL
- family-specific latest scenario retrieval is fixed
- scenario storage no longer overloads `shock_variable` with family names
- stress-test rows persist `scenario_id`
- unsupported holdings are rejected with proper errors
- `maxDrawdown` is path-wise, not just worst terminal P&L
- invalid scenario generation and invalid stress-test requests use proper HTTP errors
- dashboard top causal-link confidence no longer fabricates `100`
- advisor scenario lookup prefers normalized schema

### 10.2 Verification checks run

The following checks were run during the recent verification pass:

- backend regression tests: passed
- frontend production build: passed
- frontend test run: passed
- backend container build: passed
- Docker Compose config parse: passed

### 10.3 Current automated test reality

Despite the successful checks above, test coverage is still shallow:

- backend has one focused regression test file
- frontend has one trivial example Vitest test
- there is no broad CI-grade end-to-end suite in the repo

So the system is now **more trustworthy than before**, but not fully covered by comprehensive automated testing.

---

## 11. Product Surfaces and Business Meaning

### 11.1 Dashboard

Best for:

- executive summary
- system status
- regime awareness
- high-level causal and market context

### 11.2 Causal graph explorer

Best for:

- researcher/analyst exploration
- understanding modeled transmission links
- comparing graph structure across regimes

### 11.3 Scenario lab

Best for:

- inspecting generated stress scenarios
- comparing families/severities/horizons
- seeing how shock templates map into scenario distributions

### 11.4 Stress test page

Best for:

- direct portfolio impact analysis
- showcasing the clearest business value

### 11.5 Adversarial ranked scenarios

Best for:

- answering “where is this portfolio most vulnerable?”
- giving ranked worst plausible pathways instead of a single generic scenario

### 11.6 AI advisor

Best for:

- exploratory narrative assistance
- interpreting current system outputs

Worst thing to assume:

- that it is a hardened enterprise chat service; it is not

---

## 12. Technical Strengths

The strongest things about this system are:

1. It is not just a UI wrapper; there is a serious modeling pipeline underneath.
2. It treats causal structure as regime-dependent rather than fixed.
3. It supports both generated scenarios and portfolio stress outputs.
4. It includes a real regulatory-comparison layer.
5. It includes a real adversarial search layer with saved models and integrated serving.
6. The latest core schema/API correctness bugs were addressed and verified.

---

## 13. Current Weaknesses and Technical Debt

### 13.1 Operational maturity gaps

- no authentication or authorization
- no multi-tenant isolation
- no visible rate limiting
- Redis is configured but not meaningfully leveraged as a job system
- backend startup is heavier than ideal because adversarial models preload eagerly

### 13.2 Engineering structure gaps

- research scripts and production-serving code live in the same repo tree
- sync DB access inside async FastAPI routes
- repeated raw DB connection helpers
- lightweight migration approach
- large number of artifacts and experiment outputs mixed into the repo

### 13.3 Testing gaps

- backend testing is targeted, not broad
- frontend tests are minimal
- there is no strong end-to-end browser/API regression suite

### 13.4 Performance and maintainability risks

- frontend bundle is large
- causal graph page is D3-heavy
- advisor state is process-local
- backend startup loads substantial modeling state eagerly

### 13.5 Product-trust caveats

- the dashboard “live” indicator is still not a full stale-data detector
- adversarial quality is uneven across profiles
- some frontend routes/features remain placeholders or partially productized

---

## 14. Runtime vs Research Boundaries

This repo is easier to understand if you separate runtime-critical files from research-support files.

### 14.1 Runtime-critical areas

Most important runtime code:

- `backend/app/main.py`
- `backend/app/config.py`
- `backend/app/routers/*.py`
- `backend/db_init/*.sql`
- `frontend/src/App.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/pages/*.tsx`
- `ml_pipeline/generative_engine/scenario_generator.py`
- `ml_pipeline/narrative/advisor_engine.py`
- `ml_pipeline/generative_engine_rl/adversarial_serve.py`
- `ml_pipeline/generative_engine_rl/causal_stress_env.py`
- `ml_pipeline/generative_engine_rl/neural_bandit.py`
- `ml_pipeline/generative_engine_rl/portfolio_model.py`

### 14.2 Mostly offline / research / analysis support

Important but not runtime-critical:

- many `figure_*.py` files
- paper experiment drivers
- extraction scripts
- coverage / precision analysis scripts
- saved run directories
- generated PDFs/PNGs/JSON artifacts

These matter for evaluation and paper work, but not for understanding the app’s main runtime behavior.

---

## 15. File Role Map

This section focuses on the files that matter most to understanding the app.

### 15.1 Root

- `docker-compose.yml` - local multi-service runtime definition
- `CausalStress_Complete_Technical_Briefing.md` - this briefing
- `VERIFIED_BUG_AUDIT.md` - historical audit of verified issues
- `README.md` - repo-level overview, partially stale compared with current system

### 15.2 Backend

- `backend/Dockerfile` - backend image build
- `backend/requirements.txt` - backend Python dependencies
- `backend/db_init/01_init.sql` - bootstrap schema
- `backend/db_init/02_runtime_schema_compat.sql` - compatibility schema repair script
- `backend/app/main.py` - FastAPI entrypoint and model preload
- `backend/app/config.py` - environment-backed settings
- `backend/app/routers/dashboard.py` - dashboard APIs
- `backend/app/routers/causal.py` - causal graph APIs
- `backend/app/routers/regimes.py` - regime APIs
- `backend/app/routers/scenarios.py` - scenario APIs
- `backend/app/routers/stress_test.py` - portfolio stress-test API
- `backend/app/routers/advisor.py` - advisor API
- `backend/app/routers/adversarial.py` - adversarial scenario APIs
- `backend/tests/test_regressions.py` - focused backend regression suite

### 15.3 Frontend

- `frontend/src/App.tsx` - route and provider root
- `frontend/src/services/api.ts` - typed API client
- `frontend/src/pages/Index.tsx` - dashboard page
- `frontend/src/pages/CausalGraph.tsx` - graph explorer
- `frontend/src/pages/Regimes.tsx` - regimes page
- `frontend/src/pages/ScenarioLab.tsx` - scenario lab
- `frontend/src/pages/StressTest.tsx` - stress test and ranked adversarial output
- `frontend/src/pages/AIAdvisor.tsx` - advisor page
- `frontend/src/components/layout/Layout.tsx` - main layout shell
- `frontend/src/components/layout/Sidebar.tsx` - navigation
- `frontend/src/test/example.test.ts` - current minimal frontend test

### 15.4 ML pipeline

- `ml_pipeline/data_ingestion/*` - fetch and process inputs
- `ml_pipeline/causal_discovery/*` - causal graph engines
- `ml_pipeline/regime_detection/*` - HMM and regime graphs
- `ml_pipeline/generative_engine/scenario_generator.py` - canonical scenario generator
- `ml_pipeline/risk_engine/portfolio.py` - offline portfolio stress engine
- `ml_pipeline/regulatory/regulatory_engine.py` - regulatory comparison
- `ml_pipeline/narrative/advisor_engine.py` - advisor engine
- `ml_pipeline/generative_engine_rl/*` - adversarial search stack
- `ml_pipeline/RL_RESULTS_REPORT.md` - RL/adversarial results narrative
- `ml_pipeline/runs/*` - saved artifacts and evaluation outputs

---

## 16. If You Need to Explain This to a VP

### 16.1 Two-minute version

CausalStress is a causal AI stress-testing platform for portfolios.
It ingests macro and market data, learns how variables influence one another, identifies the market regime, generates regime-aware crisis scenarios, translates them into portfolio losses, compares them with regulatory scenarios, and adds a worst-case adversarial search layer for portfolio-specific vulnerabilities.

### 16.2 Five strengths to emphasize

1. It is a full modeling pipeline, not just a dashboard.
2. It models market state as dynamic and regime-dependent.
3. It can produce explainable stress scenarios through causal pathways.
4. It supports both portfolio analysis and regulatory comparison.
5. It has a live adversarial search capability for ranked worst plausible scenarios.

### 16.3 Three honest caveats

1. It is more mature as a research/product prototype than as a hardened enterprise platform.
2. Testing and operational hardening still lag the modeling ambition.
3. The adversarial layer is promising but not uniformly benchmark-strong across all profiles.

### 16.4 Best answer to “Is it production-ready?”

> The core capabilities are real and demonstrable today, especially the causal/regime/scenario stack, portfolio stress outputs, and adversarial search integration. The remaining work to reach enterprise-grade production readiness is mostly in hardening: auth, migration discipline, jobs, service boundaries, broader automated tests, and operational guardrails.

---

## 17. Bottom Line

The most accurate way to understand CausalStress is:

- a real full-stack financial risk product prototype
- built around a serious modeling and scenario-generation pipeline
- with live DB-backed outputs and integrated portfolio workflows
- and with a genuine adversarial search subsystem layered on top

Its strongest story is:

> turning changing macro-financial causal structure into explainable, regime-aware, portfolio-specific stress testing

Its weakest area is:

> operational hardening, test depth, and clean separation between research code and production engineering

That is the complete and current picture of the system.
