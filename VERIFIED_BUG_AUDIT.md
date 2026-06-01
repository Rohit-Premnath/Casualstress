# CausalStress Verified Bug Audit

Last audited: 2026-05-27

## Scope

This audit is based on:

- direct code inspection across `backend`, `frontend`, and `ml_pipeline`
- live database inspection against the local PostgreSQL instance
- FastAPI endpoint checks with `TestClient`
- frontend build, test, and lint runs

Important honesty note:

- This is a deep bug audit, not a mathematical proof that the app is bug-free.
- I can confirm many real bugs and deployment blockers.
- I cannot guarantee "100 percent correct" without broader automated coverage, load testing, deployment rehearsal, and business-rule validation.

## Verification Snapshot

- Frontend build: `npm run build` passed
- Frontend tests: `npm test -- --run` passed, but only 1 trivial test exists
- Frontend lint: failed with 44 errors and 16 warnings
- Backend smoke checks: `/`, `/live`, `/health`, major API routes responded
- Python compile check: `backend/app` and `ml_pipeline` compiled successfully
- Database was inspected directly

## Most Serious Confirmed Bugs

### 1. Clean database setup does not match what the running app expects

Severity: Critical

Simple explanation:

- A fresh deployment can break even if this machine works.
- The visible SQL setup file and the real runtime schema are out of sync.

Evidence:

- Visible init SQL defines `models.scenarios.regime_condition` as `INTEGER` in [backend/db_init/01_init.sql](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/db_init/01_init.sql:98)
- Scenario generation writes string values like `"stressed"` into that column in [ml_pipeline/generative_engine/scenario_generator.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine/scenario_generator.py:771)
- Visible init SQL does not create an `event_type` column for `models.scenarios` in [backend/db_init/01_init.sql](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/db_init/01_init.sql:93)
- But the app checks for and expects `event_type` in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:58) and [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:48)
- Visible init SQL does not create the `regulatory` schema or tables, but the regulatory engine reads and writes them in [ml_pipeline/regulatory/regulatory_engine.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regulatory/regulatory_engine.py:171) and [ml_pipeline/regulatory/regulatory_engine.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regulatory/regulatory_engine.py:502)

Verified live DB finding:

- The actual live DB has `regulatory.scenarios` and `regulatory.causal_difference_reports`
- The actual live DB has `models.scenarios.regime_condition` as `character varying`, not integer
- The actual live DB still does not have `models.scenarios.event_type`

Impact:

- Fresh installs can fail during scenario generation
- Regulatory features may fail on clean environments
- Dev, staging, and production can drift silently

Likely fix:

- Add real migrations
- Make `01_init.sql` match the true required schema or replace it with migration-driven setup
- Decide on one final `models.scenarios` schema and update code to use it consistently

### 2. Scenario family filtering is broken on the current live schema

Severity: Critical

Simple explanation:

- If the UI or API asks for the latest `credit_crisis` or `rate_shock` scenario, it still gets the latest scenario overall.
- So family-specific retrieval is lying right now.

Evidence:

- In [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:429), when `event_type` is requested but the column does not exist, the code falls back to:
  - `SELECT ... FROM models.scenarios ORDER BY created_at DESC LIMIT 1`
- I verified live API behavior:
  - `market_crash` returned `market_crash`
  - `credit_crisis` returned `market_crash`
  - `rate_shock` returned `market_crash`
  - `global_shock` returned `market_crash`

Impact:

- Scenario family selection can be wrong
- Any downstream page using family-specific latest scenarios can show incorrect results

Likely fix:

- Stop relying on a missing `event_type` column
- Either add `event_type` properly, or add deterministic mapping columns and query them directly

### 3. Stress-test results are often saved without the scenario that produced them

Severity: High

Simple explanation:

- The app stores many stress test results, but often forgets to store which scenario generated them.
- That makes auditability and reproducibility much weaker.

Evidence:

- `app.stress_test_results` has a `scenario_id` column in [backend/db_init/01_init.sql](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/db_init/01_init.sql:113)
- The FastAPI route insert omits it in [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:367)
- A different older pipeline function stores it correctly in [ml_pipeline/risk_engine/portfolio.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/risk_engine/portfolio.py:403)

Verified live DB finding:

- `app.stress_test_results`: 52 total rows
- linked `scenario_id`: 12
- missing `scenario_id`: 40

Impact:

- You cannot reliably answer "which scenario caused this result?"
- Historical analysis and governance are weakened

Likely fix:

- Include `scenario_id` in the FastAPI insert
- Backfill historical rows where possible

### 4. "Max Drawdown" is not actually max drawdown

Severity: High

Simple explanation:

- The API labels one metric as max drawdown, but it is really just the worst final scenario P&L.
- Those are not the same thing.

Evidence:

- The stress route computes:
  - `max_drawdown = float(np.min(pnls))`
  - in [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:273)
- `pnls` there are final scenario totals, not path-wise peak-to-trough portfolio drawdowns

Impact:

- The UI can overclaim analytical sophistication
- Users may think they are seeing path drawdown risk when they are not

Likely fix:

- Either rename the metric to `Worst Simulated Loss`
- Or compute real path-wise drawdown per scenario using the full scenario trajectory

### 5. The app claims "live" health even when the market data is stale

Severity: High

Simple explanation:

- The UI says `DATA LIVE` when the API is reachable, even if the underlying market data is old.

Evidence:

- Header status only checks `/health` success in [frontend/src/components/layout/Layout.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/components/layout/Layout.tsx:9)
- The status text is derived from API reachability, not data freshness, in [frontend/src/components/layout/Layout.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/components/layout/Layout.tsx:27)

Verified live data finding:

- Current system date during audit: 2026-05-27
- Latest processed market date in DB: 2026-04-15
- That is a 42-day lag

Impact:

- This is a credibility risk in demos and executive reviews
- "Live" can be interpreted as "up to date," which is false here

Likely fix:

- Add a freshness check using `MAX(date)` from `processed.time_series_data`
- Show `live`, `stale`, or `offline` separately

### 6. Causal-link confidence is overstated as 100% when no confidence exists

Severity: High

Simple explanation:

- The causal dashboard shows confidence percentages, but for the main graph those values are not stored.
- Missing confidence is being turned into `100%`.

Evidence:

- Top causal links default missing confidence to `1.0` in [backend/app/routers/dashboard.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/dashboard.py:237)

Verified live DB finding:

- Preferred global graph edges: 1,249
- Edges with stored confidence: 0
- Edges without stored confidence: 1,249

Impact:

- Users can be misled into thinking confidence has been measured when it has not

Likely fix:

- Return `null` or `unknown` for missing confidence
- Show separate weight and confidence concepts in the UI

### 7. Custom holdings can be silently ignored in stress tests

Severity: High

Simple explanation:

- The frontend lets the user add arbitrary holdings.
- The backend only knows how to stress assets that map to known variable codes.
- Unknown assets are skipped with no warning.

Evidence:

- Frontend allows adding `"New Asset"` in [frontend/src/pages/StressTest.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/StressTest.tsx:116)
- Backend skips holdings when their variable code is not in the scenario path in [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:242)

Impact:

- Portfolio losses can be understated
- Risk percentages can look safer than they really are

Likely fix:

- Reject unsupported assets with a validation error
- Or show an explicit warning listing ignored holdings

### 8. Advanced scenario overrides are not faithfully reflected back to the UI

Severity: High

Simple explanation:

- The scenario generator accepts custom anchor variable and magnitude overrides.
- But the response payload shown to the UI rebuilds the shock template from family defaults, not the actual override used.

Evidence:

- Overrides are accepted in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:489)
- But the displayed shock template is rebuilt from family defaults in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:366)

Impact:

- The user can run one scenario and see a different-looking scenario described back to them

Likely fix:

- Pass the actual anchor variable and actual magnitude into the response builder
- Persist them explicitly in the stored row

### 9. The advisor can use stale old-style scenarios instead of the current canonical ones

Severity: High

Simple explanation:

- The advisor's stress-test helper looks up scenarios by `shock_variable = '^GSPC'`.
- Newer canonical rows store family names like `market_crash` in that same column.
- So the advisor can pull an older scenario set even when a newer one exists.

Evidence:

- Advisor helper query in [ml_pipeline/narrative/advisor_engine.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/narrative/advisor_engine.py:288)
- Scenario storage overloads `shock_variable` with `event_type or shock_variable` in [ml_pipeline/generative_engine/scenario_generator.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/generative_engine/scenario_generator.py:769)

Verified live DB finding:

- Latest `^GSPC` row: 2026-04-03, 100 scenarios
- Latest `market_crash` row: 2026-05-13, 400 scenarios

Impact:

- The AI advisor can reason over stale scenario sets
- Cross-feature consistency breaks

Likely fix:

- Separate `event_type` and `anchor_variable`
- Update advisor lookup logic to prefer canonical latest rows

### 10. Production deployment is not ready as-is

Severity: High

Simple explanation:

- The backend container and API config are still dev-oriented.

Evidence:

- Docker runs Uvicorn with `--reload` in [backend/Dockerfile](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/Dockerfile:21)
- Backend code imports `ml_pipeline`, but the Dockerfile only copies `backend`; it relies on a compose volume mount instead in [backend/Dockerfile](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/Dockerfile:18) and [docker-compose.yml](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/docker-compose.yml:57)
- CORS only allows localhost origins in [backend/app/main.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/main.py:45)

Impact:

- Standalone backend images can fail without mounted code
- Production frontend origin will be blocked unless code changes
- `--reload` is not a production process model

Likely fix:

- Build a real production image that includes `ml_pipeline`
- Remove `--reload`
- Move CORS origins to config/env

## Confirmed Medium-Severity Bugs

### 11. Several API routes return error JSON with HTTP 200

Severity: Medium

Simple explanation:

- Some endpoints say "error" in the response body but still return success status code.
- Monitoring, retries, and frontend handling become unreliable.

Evidence:

- Invalid scenario family returns 200 in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:484)
- Missing scenario returns 200 in [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:206)
- I verified both behaviors with `TestClient`

Impact:

- Frontend may treat failed operations as successes
- Logs and alerts can miss real failures

Likely fix:

- Raise `HTTPException` with proper status codes

### 12. Older scenarios can be mislabeled as `market_crash`

Severity: Medium

Simple explanation:

- Some old rows stored only the anchor ticker, like `CL=F` or `DGS10`.
- The current inference logic does not map those correctly and defaults them to `market_crash`.

Evidence:

- Inference default is in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:165)
- Live DB contains older `shock_variable` values like `CL=F`, `DGS10`, `BAMLH0A0HYM2`, `^GSPC`

Impact:

- Scenario history can show the wrong family label

Likely fix:

- Add explicit mapping from old anchor tickers to family IDs
- Backfill data

### 13. `/api/v1/scenarios/latest` hardcodes `horizon = 60`

Severity: Medium

Simple explanation:

- The endpoint says the latest scenario has a 60-day horizon even if the latest stored scenario might be 10 or 30 days.

Evidence:

- Hardcoded horizon in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:474)

Impact:

- Scenario metadata can be wrong

Likely fix:

- Derive horizon from stored path length

### 14. Scenario count semantics are inconsistent

Severity: Medium

Simple explanation:

- The latest scenario response reports `candidateCount = 400` and `scenarioCount = 200`, but the stored row actually contains 400 paths.
- So `scenarioCount` is really a display target, not the actual stored scenario count.

Evidence:

- Response builder sets `candidateCount = len(paths)` and `scenarioCount = displayed_paths` in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:395)
- Latest live row in DB had `n_scenarios = 400` and `jsonb_array_length(scenario_paths) = 400`

Impact:

- Users can misunderstand how many scenarios are really being analyzed

Likely fix:

- Rename fields or return both `storedPathCount` and `displayTargetCount`

### 15. Historical stress-test rows use inconsistent JSON shapes

Severity: Medium

Simple explanation:

- Some stored stress-test rows use arrays.
- Older ones use objects.
- Future history views can break if they assume one shape.

Verified live DB finding:

- 40 rows have array-shaped `portfolio`, `sector_decomposition`, and `marginal_contributions`
- 12 older rows have object-shaped versions instead

Impact:

- Historical reporting is fragile

Likely fix:

- Standardize one JSON schema and backfill older rows

### 16. Current regime streak can be undercounted after 500 days

Severity: Low to Medium

Simple explanation:

- The `/regimes/current` endpoint only inspects the latest 500 rows.
- Very long streaks will be truncated.

Evidence:

- Limit in [backend/app/routers/regimes.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/regimes.py:71)

Impact:

- Current streak can be wrong in extreme cases

Likely fix:

- Use a DB-side streak query like the dashboard route does

### 17. Health only means DB reachable, not app truly ready

Severity: Low to Medium

Simple explanation:

- `/health` does not check data freshness, regulatory tables, or model availability.

Evidence:

- Health logic only tests `SELECT 1` in [backend/app/main.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/main.py:108)

Impact:

- Ops can see "healthy" while important product capabilities are degraded

Likely fix:

- Add readiness checks for data recency and model load status

## Static Quality and Maintainability Issues

### 18. Frontend lint currently fails

Severity: Medium

Verified result:

- `npm run lint` failed with 44 errors and 16 warnings

Main categories:

- heavy `any` usage, especially in [frontend/src/pages/CausalGraph.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/CausalGraph.tsx:204)
- React hook dependency warnings in [frontend/src/pages/CausalGraph.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/CausalGraph.tsx:445)
- API type looseness in [frontend/src/services/api.ts](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/services/api.ts:355)
- tailwind config import style issue in [frontend/tailwind.config.ts](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/tailwind.config.ts:96)

Impact:

- Type safety is weaker than it looks
- Some stale-render bugs in the graph page are plausible because of missing hook deps

### 19. Automated tests are far too shallow

Severity: High as a process gap

Evidence:

- Only one trivial frontend test exists in [frontend/src/test/example.test.ts](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/test/example.test.ts:3)
- Backend `pytest` discovered no meaningful test coverage during audit

Impact:

- You cannot realistically claim the app is fully verified

### 20. Backend async routes still use blocking database connections

Severity: Medium for production scale

Simple explanation:

- Most FastAPI handlers are `async`, but they still use synchronous `psycopg2`.
- That can reduce concurrency under load.

Evidence:

- Example in [backend/app/routers/dashboard.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/dashboard.py:48)
- Example in [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:463)

Impact:

- Performance and responsiveness can degrade under heavier traffic

## Placeholder / Incomplete Features

These are not the same as "fake core outputs," but they are unfinished:

- Settings page is a stub in [frontend/src/App.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/App.tsx:33)
- Advisor attachment button has no upload flow in [frontend/src/pages/AIAdvisor.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/pages/AIAdvisor.tsx:181)
- Notification bell is decorative in [frontend/src/components/layout/Layout.tsx](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/frontend/src/components/layout/Layout.tsx:61)

## Database Findings Worth Cleaning Up

- `models.scenarios` has 102 valid array payload rows
- 64 scenario rows are missing `soft_filter_weight` entries inside path objects
- `app.stress_test_results` has mixed historical JSON shapes
- Live DB has extra runtime schema not represented in visible setup SQL

## Highest-Value Fix Order

1. Fix schema drift first
2. Fix scenario family filtering
3. Fix stress-test result linkage to `scenario_id`
4. Fix the fake `maxDrawdown` metric or rename it
5. Fix stale-data detection and stop calling stale data "live"
6. Fix causal confidence reporting
7. Fix custom holding validation
8. Fix advanced scenario override reporting
9. Fix advisor scenario lookup consistency
10. Make deploy config production-safe
11. Make API errors return proper HTTP status codes
12. Add real tests before claiming verification

## Bottom Line

The app is not fake overall. The core causal/regime/scenario/stress/adversarial flows are real.

But it is not yet "100 percent correct, verified, and deployable."

The main blockers are:

- schema drift
- misleading API/UI semantics in a few important places
- stale-data presentation
- incomplete deployment hardening
- weak test coverage
