# CausalStress System Reference

Status: cleaned and aligned to the current repo code paths as inspected on 2026-06-03.

This version is intentionally conservative. It only states behavior that is supported by the current codebase, schema, or checked project artifacts. Where the repo contains research branches or UI placeholders, that is called out explicitly.

---

## 1. What the system is

CausalStress is a financial stress-testing platform built around three linked ideas:

1. ingest macro and market time series into a common daily panel
2. detect market regimes and learn directed cross-variable relationships
3. generate stressed scenario paths and score portfolio losses across those paths

At a product level, the repo currently contains:

- a FastAPI backend
- a React + Vite frontend
- a PostgreSQL / TimescaleDB data store
- a Redis service in Docker Compose
- an `ml_pipeline` folder that performs data ingestion, regime detection, causal discovery, and scenario generation

Redis is provisioned in the stack, but the core inspected pipeline and API paths are primarily Postgres-backed.

---

## 2. High-level architecture

The active end-to-end flow is:

`FRED + Yahoo -> processed.time_series_data -> models.regimes + models.causal_graphs -> models.scenarios -> stress testing / advisor / frontend`

The backend currently registers 7 routers in `backend/app/main.py`.

Those routers are:

- `dashboard`
- `causal`
- `regimes`
- `scenarios`
- `stress_test`
- `advisor`
- `adversarial`

The frontend currently exposes 6 main product pages plus a settings placeholder:

- Dashboard
- Causal Graph
- Regimes
- Scenario Lab
- Stress Test
- AI Advisor

The dev frontend port is `8080` in `frontend/vite.config.ts`, and the backend health/docs run on port `8000`.

---

## 3. Data layer

### 3.1 Sources

The live ingestion code uses two sources:

- FRED for macro, rates, spreads, and survey series
- Yahoo Finance for market tickers

### 3.2 Variable counts

The current processor definitions in `ml_pipeline/data_ingestion/data_processor.py` contain:

- 35 FRED series in `FRED_TRANSFORMS`
- 21 Yahoo tickers in `YAHOO_TRANSFORMS`

That gives:

- 56 base variables total

Important detail:

- `DX-Y.NYB` is currently part of `YAHOO_TRANSFORMS`
- `XLRE` is also currently part of `YAHOO_TRANSFORMS`

### 3.3 Engineered features

`compute_features()` adds up to 13 engineered features:

- rolling 21d and 63d volatility for `^GSPC`, `^NDX`, `^RUT`, and `^VIX` = 8 features
- `HY_SPREAD_ZSCORE`
- `VIX_ZSCORE`
- `HY_IG_SPREAD_GAP`
- `CCC_BB_SPREAD_GAP`
- `SOFR_CP_SPREAD`

So the processed panel can reach:

- 56 base variables
- up to 13 engineered variables
- up to 69 distinct `variable_code` values

The dashboard widget for variables tracked uses `COUNT(DISTINCT variable_code)` over `processed.time_series_data`.

### 3.4 Storage

The core schemas created by `backend/db_init/01_init.sql` are:

- `raw_fred.observations`
- `raw_yahoo.daily_prices`
- `processed.time_series_data`
- `models.causal_graphs`
- `models.regimes`
- `models.scenarios`
- `app.stress_test_results`
- `app.scheduler_log`

`processed.time_series_data` stores both:

- `raw_value`
- `transformed_value`

This matters because regime and display logic can use raw levels, while modeling uses transformed values.

---

## 4. Data processing pipeline

The active processor flow in `ml_pipeline/data_ingestion/data_processor.py` is:

1. load raw FRED and Yahoo data
2. align everything to Yahoo's market-day calendar
3. forward-fill lower-frequency FRED series onto that calendar
4. handle missing data with:
   - linear interpolation up to 5 days
   - forward-fill for remaining gaps
   - backward-fill at the start if needed
5. apply stationarity transforms
6. compute engineered features
7. write results to `processed.time_series_data`

Transform classes currently used:

- `log_return` for price-like series
- `first_diff` for many rate/spread series
- `none` for selected already-stationary or as-is series

The code stores source labels as:

- `fred`
- `yahoo`
- `engineered`

---

## 5. Regime detection

Regime detection is implemented in `ml_pipeline/regime_detection/hmm_model.py`.

What is verified from code:

- it uses `GaussianHMM` from `hmmlearn`
- it stores one row per trading day in `models.regimes`
- each row includes:
  - `date`
  - `regime_label`
  - `regime_name`
  - `probability`
  - `transition_probs`

Important nuance:

- the code supports multiple regime counts
- regime names depend on the chosen state count
- for 5 states the names are `calm`, `normal`, `elevated`, `stressed`, `crisis`
- for 6 states the names are `calm`, `normal`, `elevated`, `stressed`, `high_stress`, `crisis`

So any statement that the system always uses exactly one fixed label set should be avoided unless tied to a specific saved run.

---

## 6. Causal discovery

There are two main causal-discovery paths in the repo:

### 6.1 Global discovery

- `ml_pipeline/causal_discovery/dynotears_engine.py`
- `ml_pipeline/causal_discovery/pcmci_engine.py`

Important accuracy note:

- `dynotears_engine.py` is not a mathematically exact DYNOTEARS implementation
- the file itself describes it as a practical approximation using penalized VAR / Lasso
- stored method names include `dynotears_lasso`

PCMCI is implemented with `tigramite` and `ParCorr`, and the ensemble result is stored with method name:

- `ensemble_dynotears_pcmci`

### 6.2 Regime-specific discovery

`ml_pipeline/regime_detection/regime_causal_graphs.py` builds regime-conditional graphs using `LassoCV` on regime-filtered data.

The scenario generator prefers:

1. the canonical local graph from `regime_causal_graphs.json` for the canonical stressed graph
2. otherwise a regime-specific DB graph if available
3. otherwise a fallback ensemble graph

### 6.3 What the product surfaces

The causal API exposes:

- `GET /api/v1/causal/graph`
- `GET /api/v1/causal/regime-comparison`

The frontend Causal Graph page supports:

- graph viewing
- node search/filtering
- regime comparison
- a transmission-paths view based on strongest directed 2-step and 3-step chains

The frontend does not expose a general arbitrary source-to-target path query form. That capability exists in the advisor tool layer instead.

---

## 7. Scenario generation

Scenario generation is implemented in `ml_pipeline/generative_engine/scenario_generator.py`, with canonical settings locked in `ml_pipeline/canonical_best_model.py`.

### 7.1 Canonical settings

Verified canonical constants include:

- horizon: `60` days
- `MAX_VAR_VARIABLES = 25`
- training regimes: `["elevated", "stressed", "high_stress", "crisis"]`
- target displayed scenarios: `200`
- candidate multiplier: `2`
- canonical candidate count: `400`
- filter mode: `soft_plausibility`
- innovation mode: `student_t_data_fit`
- fitted Student-t degrees of freedom:
  - normal: `5.97`
  - crisis: `3.84`
  - mid: `4.79`

### 7.2 The 25 core variables

The live `CORE_VARIABLES` list is:

- `^GSPC`
- `^VIX`
- `^NDX`
- `^RUT`
- `DGS10`
- `DGS2`
- `T10Y2Y`
- `FEDFUNDS`
- `CL=F`
- `GC=F`
- `BAMLH0A0HYM2`
- `BAMLH0A3HYC`
- `BAMLC0A0CM`
- `XLF`
- `XLK`
- `XLE`
- `XLV`
- `XLY`
- `XLU`
- `TLT`
- `LQD`
- `HYG`
- `EEM`
- `CPIAUCSL`
- `UNRATE`

### 7.3 VAR fitting

The active scenario generator:

- fits a VAR-like linear model with ridge stabilization
- uses `max_lag=5`
- computes both a normal residual covariance and a crisis covariance
- uses the crisis covariance more heavily for larger shocks

### 7.4 Shock templates and families

The active scenario API supports 6 named families:

- `market-crash`
- `credit-crisis`
- `rate-shock`
- `global-shock`
- `vol-shock`
- `pandemic`

Internally these map to event types such as:

- `market_crash`
- `credit_crisis`
- `rate_shock`
- `global_shock`
- `volatility_shock`
- `pandemic_exogenous`

The generator uses multi-root shock templates rather than a single shocked variable only.

### 7.5 Propagation and persistence

Verified live constants:

- propagation depth: `3`
- propagation decay: `0.4`
- propagation clip: `2.5`
- propagation minimum: `0.12`
- shock persistence days: `5`
- shock persistence decay: `0.72`
- VIX template cap: `3.5`
- VIX propagation scale: `0.65`

### 7.6 Candidate paths vs stored paths

The cleanest accurate way to describe the current canonical flow is:

- the system generates 400 candidate paths
- applies soft plausibility weights
- keeps all generated paths with weights attached

That is important because the current soft filter function returns the full scenario list plus weights; it does not hard-downselect to 200 inside the generator.

So for the current API/backend path:

- `candidateCount` often reflects the full weighted path set
- `displayTargetCount` carries the target display count

### 7.7 Important implementation caveat

`select_var_variables()` does not truly "swap in" a non-core shock variable by displacing the lowest-priority variable. It appends the shock variable and then truncates back to the first 25 variables, so a non-core appended variable can still be dropped.

That means any external explanation should avoid claiming guaranteed temporary inclusion of arbitrary non-core shock variables.

---

## 8. Stress testing

The stress-test API lives in `backend/app/routers/stress_test.py`.

What it does:

- loads the latest or a specified stored scenario set
- maps holdings to variable codes
- computes PnL path-by-path
- weights outcomes using stored soft-filter weights when present
- returns portfolio-level and holding-level downside summaries

Current output metrics include:

- `var95`
- `var99`
- `cvar95`
- `maxDrawdown`
- `lossProbabilities`
- `sectorRisk`
- `holdingRisk`
- `pnlDistribution`
- `topContributors`
- `topAbsorbers`

The stress test persists summary output into `app.stress_test_results`.

Important caveat:

- the asset map includes `"Real Estate": "XLRE"`
- but `XLRE` is not in the canonical 25-variable scenario set
- as a result, some scenario sets can reject real-estate holdings as unsupported at runtime

---

## 9. AI Advisor

The advisor API router is `backend/app/routers/advisor.py`, and the tool-enabled runtime is `ml_pipeline/narrative/advisor_engine.py`.

Verified behavior:

- the real advisor requires `ANTHROPIC_API_KEY`
- without that key, the API returns a polite fallback message
- the runtime model call currently uses `claude-sonnet-4-20250514`
- the frontend renders advisor responses with `ReactMarkdown`

The current tool inventory is 7 tools:

- `get_current_regime`
- `get_regime_history`
- `get_causal_links`
- `trace_causal_path`
- `run_stress_test`
- `get_regulatory_divergences`
- `get_system_stats`

Important nuance:

- arbitrary source-to-target causal tracing is available here through `trace_causal_path`
- this is more capable than the current frontend graph transmission view

---

## 10. API surface

The live API endpoints currently exposed by the backend are:

### Dashboard

- `GET /api/v1/dashboard/summary`
- `GET /api/v1/dashboard/spx-history`
- `GET /api/v1/dashboard/regime-chart`
- `GET /api/v1/dashboard/top-causal-links`

### Causal

- `GET /api/v1/causal/graph`
- `GET /api/v1/causal/regime-comparison`

### Regimes

- `GET /api/v1/regimes/current`
- `GET /api/v1/regimes/timeline`
- `GET /api/v1/regimes/characteristics`
- `GET /api/v1/regimes/transition-matrix`

### Scenarios

- `GET /api/v1/scenarios/metadata`
- `GET /api/v1/scenarios/latest`
- `POST /api/v1/scenarios/generate`
- `GET /api/v1/scenarios/list`

### Stress test

- `POST /api/v1/stress-test/run`

### Advisor

- `POST /api/v1/advisor/chat`
- `GET /api/v1/advisor/suggested-prompts`

### Adversarial

- `POST /api/v1/adversarial/worst-case`
- `POST /api/v1/adversarial/ranked-scenarios`
- `GET /api/v1/adversarial/status`

---

## 11. Frontend behavior notes

For PPT accuracy, these frontend details matter:

- the dev frontend port is `8080`, not `3000`
- the frontend still contains some fallback/mock support code
- Scenario Lab intentionally avoids falling back to hardcoded scenario families if metadata fails
- some UI presets and display helpers still come from `mockData.ts`

So "all values everywhere are always live and non-hardcoded" is too strong as a blanket statement.

---

## 12. Research and extended branches

The repo also contains adjacent work that is real code, but not part of the core causal-stress scenario API path:

- regulatory comparison scripts and data under `ml_pipeline/regulatory/`
- RL-based adversarial scenario search under `ml_pipeline/generative_engine_rl/`
- VECM and copula research engines under `ml_pipeline/generative_engine/`

These can be mentioned in a presentation, but they should be framed as:

- implemented supporting modules
- research / extended capabilities
- not the same thing as the core scenario API path unless specifically demonstrated

---

## 13. Safe summary for presentation use

If you need a short accurate verbal description, this wording is safe:

"CausalStress is a regime-aware financial stress-testing platform. It ingests 56 macro and market variables from FRED and Yahoo Finance, transforms them into a daily modeling panel, detects market regimes with a Gaussian HMM, builds directed causal graphs with a Lasso-based DYNOTEARS-style pipeline plus PCMCI, and generates weighted stressed scenarios over a 60-day horizon using a 25-variable canonical scenario engine. Those scenarios feed portfolio stress testing, causal exploration, and a tool-enabled AI advisor."

---

## 14. Known caveats to avoid overstating

- Do not say the Yahoo inventory is 20 variables; it is currently 21.
- Do not say `DX-Y.NYB` is fetched but excluded from transforms; it is currently transformed.
- Do not say the backend has 8 routers; it currently registers 7.
- Do not say the frontend graph supports arbitrary source-to-target tracing; that capability lives in the advisor tool layer.
- Do not say the "DYNOTEARS" implementation is exact; the code describes it as a practical Lasso-based approximation.
- Do not say every frontend value is purely live with no hardcoded fallback or preset support.
