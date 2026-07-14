# Forensic Audit: `Causalstress_final.pdf`

## Source Identity

- The PDF is a rendered print of [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:358).
- The rendered pages show `file:///C:/Users/rohit.premnath/Desktop/Casualstress/CausalStress_Pipeline_Documentation.html`, so the HTML file in this repo is the actionable source for verification.

## Method Used

- Read the document by rendering PDF pages and matching them to the HTML source.
- Verified runtime claims against:
  - live backend API on `http://localhost:8000`
  - live Postgres tables in Docker
  - stored experiment artifacts in `models.paper_experiments`
  - repo artifacts such as `regime_graphs_data.json`, `precision_at_k_v2.json`, and `canonical_paper_numbers.py`
- Ran backend regression tests:
  - `docker exec causalstress-api sh -lc "cd /app/backend && PYTHONPATH=/app/backend pytest tests/test_regressions.py -q"`
  - Result: `5 passed`

## Overall Verdict

- The document is **partly accurate but not forensically clean**.
- The **headline backtest numbers are supported** by stored experiment artifacts.
- Several **operational and scale numbers are stale or wrong** versus the current database and runtime.
- The document also **mixes multiple measurement surfaces**:
  - older HMM paper artifacts
  - newer GMM/live-DB regime reporting
  - current runtime API behavior
  - older static counts embedded in the narrative

## Confirmed Claims

### Headline scenario-performance claims: confirmed

- `90.0%` test coverage, `77.6%` direction, `100%` pairwise consistency are supported by the latest stored `Exp4_Canonical_Ablation` artifact.
- `p=0.008` vs Historical Replay and `p=0.031` vs Unconditional VAR are supported more precisely as `0.0078` and `0.0312` by the latest stored `Exp8_Significance` artifact.
- Source lines in document:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:358)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:395)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:399)

### Causal-graph scale claims: confirmed

- `1,249` discovered edges across `56` variables is supported by:
  - [ml_pipeline/causal_graph_data.json](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/causal_graph_data.json:1)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:1046)
- `25 / 25` ground-truth recovered is supported by the latest stored `Exp1_Causal_Validation` artifact.
- `Precision@10 = 30%` and `15x` lift over random are supported by [ml_pipeline/precision_at_k_v2.json](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/precision_at_k_v2.json:1) and the table at [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:1050).

### Stress-only edge claims: confirmed

- `211` stress-only edges, `97` calm-only edges, `134` shared edges, `345` stressed edges are directly present in [ml_pipeline/regime_graphs_data.json](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regime_graphs_data.json:1).
- Document lines:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:1074)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:1075)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:1078)

### Regime hit-rate claim `14/15 = 93.3%`: confirmed

- Recomputed from the live `models.regimes` table and from `ml_pipeline/regime_data.json`.
- The `14/15` number is real.
- The document line is supported:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:959)

### Ingestion-universe counts: confirmed

- `35` FRED series: confirmed from live DB `raw_fred.observations`.
- `21` Yahoo tickers: confirmed from live DB `raw_yahoo.daily_prices`.
- Runtime stress-test universe supports `16` user-facing assets:
  - [backend/app/routers/stress_test.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/stress_test.py:76)

### Scenario-family count: confirmed, with nuance

- User-facing API exposes `6` scenario families:
  - [backend/app/routers/scenarios.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/backend/app/routers/scenarios.py:74)
- Live metadata endpoint confirms `displayedPaths=200` and `candidateCount=400`.

## Contradicted or Stale Claims

### Trading days: document says `5,581`, live data says `5,548`

- Document:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:364)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:434)
- Live DB:
  - `processed.time_series_data where source != 'engineered'`: `5548` distinct dates
  - `processed.time_series_data` total panel: `5548` distinct dates
- Source-of-truth code also says `5548`:
  - [ml_pipeline/canonical_paper_numbers.py](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/canonical_paper_numbers.py:34)

### Active regime count: document says `6`, live system has `5`

- Document:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:420)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:574)
- Live DB regime distribution only contains:
  - `elevated`, `calm`, `stressed`, `normal`, `crisis`
- Artifact also says `n_states = 5`:
  - [ml_pipeline/regime_data.json](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/ml_pipeline/regime_data.json:1)

### Current elevated-regime streak: document says `275+`, live runtime says `242`

- Document:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:379)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:420)
- Live DB / API:
  - latest regime `elevated`
  - streak `242`
  - latest date `2026-04-15`

### Stressed-regime training rows: document says `3,105`, current production loader yields `3,374`

- Document:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:383)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:612)
- Production code path, executed inside backend container:
  - total joined rows: `5528`
  - filtered rows for `['elevated','stressed','high_stress','crisis']`: `3374`
- The list still includes `high_stress`, but that label does not appear in the current data.

### Raw-ingestion counts are stale

- Document claims:
  - `45,438` FRED observations
  - `109,370` Yahoo OHLCV rows
  - `368,346` processed records
- Live DB says:
  - FRED rows: `92,946`
  - Yahoo rows: `108,717`
  - processed rows: `366,168`
- Relevant document lines:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:564)

### “11/11 on the same HMM events” is not supported by current stored classifications

- Document says:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:964)
- Recomputed from the current stored regime timeline:
  - 15-event set: `14/15 = 93.3%`
  - 11-event canonical subset: `10/11 = 90.9%`
- Current miss on the canonical 11-event subset is `2020 Tech Selloff`, not zero misses.

### The report’s explanation of the one GMM miss is contradicted by live data

- Document says the only miss is `China Stock Market Crash 2015`, labeled `normal`:
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:964)
  - [CausalStress_Pipeline_Documentation.html](/c:/Users/megha/OneDrive/Desktop/Casual%20Stress/Casualstress/CausalStress_Pipeline_Documentation.html:988)
- Live DB and `regime_data.json` both classify that event window dominantly as `stressed`.
- The actual miss in the current stored classifications is `September 2020 Tech Selloff`, which comes out `normal`.

## Mixed / Ambiguous Claims

### “20 years”

- The statement is directionally reasonable but not exact.
- Current live date range is `2005-01-04` to `2026-04-15`, which is more than 21 calendar years of span.
- I would classify this as marketing shorthand, not a precise measurement.

### Scenario-path totals are conflated

- The document uses several different path totals:
  - `400 weighted scenario paths`
  - `6 shock families × 400 = 2,400`
  - `1,600 scenarios generated`
- Current runtime contract:
  - API metadata: `200 displayed`, `400 candidate`
  - latest stored scenario batches in DB: `n_scenarios = 400`
- So the report mixes:
  - stored/generated candidate paths
  - user-facing displayed paths
  - “currently generated” family subsets

### HMM vs GMM content mixes old and new evaluation surfaces

- Old stored `Exp2_Regime_Detection` artifact still reports `72.7%`.
- Newer live-DB GMM narrative reports `93.3%`.
- Both exist in the repo, but the document does not clearly separate:
  - older paper artifact
  - newer live classification audit

## External / Regulatory Claims

### DFAST threshold: supported

- The claim that the supervisory stress test applies to firms with `$100 billion or more` in total consolidated assets is supported by official Federal Reserve 2021 DFAST materials.
- Official source:
  - https://www.federalreserve.gov/publications/2021-june-dodd-frank-act-stress-test-preface.htm

### EBA stress-testing framework: supported

- The EBA does run official EU-wide banking stress tests and publishes methodology/scenarios.
- Official source:
  - https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-stress-testing

### UK “equivalent framework” wording: over-broad

- The document cites Bank of England `CBES`, which is a climate exploratory exercise, not a simple one-to-one equivalent of DFAST.
- I would rewrite that line to avoid implying strict equivalence.

## Final Assessment

- **Reliable as a high-level product story:** yes
- **Reliable as a forensic technical reference:** no
- **Headline model-performance numbers:** supported
- **Operational counts, current-state claims, and some regime-language details:** need correction

## Minimum Corrections Needed

1. Replace `5,581` trading days with `5,548`.
2. Replace `6 regimes` with `5 active regimes`, unless you explicitly mean a theoretical label set that includes unused `high_stress`.
3. Replace `275+ consecutive trading days` with `242` as of `2026-04-15`, or make it dynamic.
4. Reconcile `3,105` stressed-regime days with the current production loader output `3,374`, or clearly label `3,105` as an older locked paper subset.
5. Update ingestion counts (`FRED`, `Yahoo`, processed rows).
6. Remove or rewrite the `11/11` and `China Stock Market Crash` regime-detection explanation.
7. Clarify scenario counts as `400 candidate / 200 displayed` if that is the intended runtime contract.
