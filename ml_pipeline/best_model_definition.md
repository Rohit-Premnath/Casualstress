# Best Model Definition

**Canonical lock date:** April 19, 2026
**Source of truth:** `canonical_paper_numbers.py` and `canonical_best_model.py`

This document describes the locked canonical scenario generation model used for
all paper experiments, headline results, and figures. If any number in this
document conflicts with `canonical_paper_numbers.py`, the Python file wins.

---

## Final canonical winner

- **Internal model name:** `causal_regime_multi_root_soft_filtered_ttails_datafit`
- **Paper label:** `Canonical Soft Filtered (Student-t, data-fit df)`
- **Full signature:**
  `causal_regime_multi_root_soft_filtered_ttails_datafit | graph=stressed_full | filter=soft | multi_root=yes | train_regimes=elevated,stressed,high_stress,crisis | innov=student_t_data_fit(df_n=5.97,df_c=3.84,df_mid=4.79)`

---

## Core components

- **Regime-conditioned training** on the four stress regimes:
  `elevated`, `stressed`, `high_stress`, and `crisis`
- **Stressed-regime causal graph** loaded from `regime_causal_graphs.json`
  (graph mode = `full`, 330 edges in the canonical stressed-graph variant)
- **Multi-root shock initialization** drawn from event-family templates
  (credit_crisis, sovereign_crisis, global_shock, volatility_shock, rate_shock,
  pandemic, pandemic_exogenous, market_crash)
- **Student-t innovations with data-fit degrees of freedom**, replacing the
  earlier Gaussian innovations. Tail behavior is calibrated empirically per
  regime rather than assumed.
- **Soft plausibility filtering** instead of hard top-k filtering.
  Scenarios are weighted by score raised to power 6.0 rather than discarded.

---

## Student-t innovation parameters (data-fit on pre-2020 VAR residuals)

The degrees-of-freedom values were estimated via Student-t MLE
(`scipy.stats.t.fit`) on VAR residuals from the pre-2020 calibration window
(2005-01-01 to 2019-12-31). See `calibrate_df_from_residuals.py`.

| Parameter | Value | Calibration source |
|-----------|-------|--------------------|
| `df_normal` | 5.97 | calm + normal regime residuals (median) |
| `df_crisis` | 3.84 | stress + crisis regime residuals (median) |
| `df_mid` | 4.79 | geometric mean of df_normal and df_crisis |
| Extreme-regime noise scale | 1.2 | applied during stressed/crisis sampling |
| Mid-regime noise scale | 1.1 | applied during elevated sampling |

**Validation:** Student-t beats Gaussian on the KS test for 20/20 variables in
both regime sets.

---

## Canonical filter and sampling settings

- Target scenarios per event: `200`
- Candidate multiplier: `2` (i.e. `400` candidate scenarios generated, then
  soft-filtered to 200)
- Filter mode: `soft_plausibility`
- Soft-filter weight transform: `score ** 6.0`
- Soft-filter minimum weight floor: `1e-6`
- Weighted quantiles used for backtest evaluation: `5th`, `50th`, `95th`

---

## Locked headline results (5-seed averaged, 2026-04-19 run)

Seeds: `[20260407, 20260408, 20260409, 20260410, 20260411]`

**Test events (n=7, out-of-sample):**

| Metric | Value |
|--------|-------|
| Coverage | 90.0% |
| Direction | 77.6% |
| Pairwise | 100.0% |
| Plausibility | 0.7706 |
| Bootstrap 95% CI (coverage) | [76.7%, 99.5%] |

**Validation events (n=4, used for canonical model selection only):**

| Metric | Value |
|--------|-------|
| Coverage | 94.2% |
| Direction | 87.5% |
| Pairwise | 100.0% |

**Statistical significance** (paired Wilcoxon, one-sided, test events only):

- vs Historical Replay: p = 0.0078 (significant)
- vs Unconditional VAR: p = 0.0312 (significant)
- vs Gaussian Monte Carlo: not significant
- vs Regime VAR (no graph): not significant
- vs Canonical Soft Filtered (Gaussian): not significant

The model selection improvement over the Gaussian variant is methodologically
motivated (better tail calibration) rather than statistically significant on
the 7-event test set. This is acknowledged in the paper.

---

## Why this model won

- **Higher validation coverage** than every comparison method: 94.2% vs
  87.5% (Gaussian canonical) vs 92.5% (Regime VAR no-graph) vs lower for all
  remaining baselines.
- **Direction accuracy gain** from data-fit Student-t tails (77.6% test
  direction vs 75.7% for the Gaussian canonical variant).
- **Perfect pairwise consistency** (100%) preserved on every test event.
- **Empirically calibrated tails** — degrees of freedom come from data, not
  from an assumption. This is a defensible methodology choice for reviewers.
- **Soft plausibility filtering** retains more scenario diversity than hard
  top-k filtering while still penalizing implausible paths through the
  weighting transform.

---

## Alignment with codebase

- `canonical_best_model.py` is the runtime configuration source of truth.
- `canonical_paper_numbers.py` is the paper-numbers source of truth.
  Includes a `_self_check()` invariant pass that runs on import.
- `all_paper_experiments.py` produces all 8 experiment artifacts using this
  canonical model. The canonical row in the main ablation table is tagged
  `role="canonical"`.
- `multi_backtest_v3.py` uses this canonical configuration for all backtest
  runs.
- `scenario_generator.py` is the production generation path. It still runs
  on regime-conditional VAR; VECM is implemented as a parallel research
  branch (see `generative_engine_vecm/` once Phase 2A scaffolding lands) and
  is not part of the current canonical path.

---

## Known limitations

- **2020 COVID coverage is 53.3%** at canonical settings — the system's
  weakest event. Pandemic dynamics are out-of-distribution for a model
  trained on 2005–2019 financial-market crises. This is acknowledged in the
  paper rather than masked.
- **Wilcoxon non-significance** vs the Gaussian canonical variant on test
  events is honestly reported. The Student-t selection is justified on
  methodology and validation-set coverage, not on test-set significance.
- **VECM cointegration evidence** is presented as supporting analysis
  (Experiment 6, 6 cointegrating vectors across 4 groups) but is not yet
  wired into the canonical scenario generator. A parallel branch under
  `generative_engine_vecm/` is the planned location for that experiment.