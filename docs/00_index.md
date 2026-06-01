# CausalStress Documentation Index

## What Is CausalStress

CausalStress is a financial stress-testing system that generates causally-grounded crisis scenarios by learning how financial variables influence each other under market stress. It combines causal discovery (DYNOTEARS + PCMCI), regime-conditional Hidden Markov Models, and VAR-based scenario generation with Student-t innovations to produce scenarios that respect empirically estimated causal structure — not just historical correlations.

The system spans data ingestion from FRED and Yahoo Finance, causal graph learning, regime classification, scenario generation, portfolio risk metrics, regulatory comparison (DFAST 2026), adversarial RL-based worst-case finding, and an LLM-powered narrative explanation engine. A FastAPI backend and React frontend expose the full pipeline.

---

## Canonical Model Snapshot (Locked 2026-04-19)

| Property | Value |
|----------|-------|
| Internal name | `causal_regime_multi_root_soft_filtered_ttails_datafit` |
| Paper label | `Canonical Soft Filtered (Student-t, data-fit df)` |
| Training data | 5,548 trading days, 2005-01-04 to 2026-04-14 |
| Variables | 56 total; 25 core for scenario generation |
| Causal edges (ensemble) | 1,249 |
| Causal edges (stressed canonical) | 330 |
| Regime model | 5-state Gaussian HMM |
| Scenario horizon | 60 days |
| Scenarios per event | 200 (soft-filtered from 400) |
| Innovation model | Student-t, df_normal=5.97, df_crisis=3.84 |
| Seeds | [20260407, 20260408, 20260409, 20260410, 20260411] |

---

## Headline Results (Locked, Test Events n=7)

| Metric | Value |
|--------|-------|
| Coverage | **90.0%** |
| Direction | **77.6%** |
| Pairwise | **100.0%** |
| Plausibility | **0.7706** |
| Bootstrap 95% CI | [76.7%, 99.5%] |
| vs Historical Replay (Wilcoxon) | +29.0 pts, **p=0.0078** |
| vs Unconditional VAR (Wilcoxon) | +27.6 pts, **p=0.0312** |

---

## Document Index

| Document | Contents |
|----------|---------|
| [01_system_overview.md](01_system_overview.md) | 4-layer pipeline, component map, data flow, infrastructure, API surface, design decisions |
| [02_data_layer.md](02_data_layer.md) | All 56 variables (tables by category), sources, transforms, 25 core variables, database schema |
| [03_causal_discovery.md](03_causal_discovery.md) | DYNOTEARS, PCMCI, ensemble merge, forbidden/required edges, regime-conditional graphs, amplification factors, Exp 1 results |
| [04_regime_detection.md](04_regime_detection.md) | 5-state Gaussian HMM, 7 feature indicators, regime distribution, crisis validation, Exp 2 results |
| [05_scenario_generation.md](05_scenario_generation.md) | Canonical model, VAR fitting, 8 shock templates, causal propagation, Student-t innovations, soft filtering, all parameters |
| [06_risk_engine.md](06_risk_engine.md) | VaR/CVaR/drawdown formulas, sector decomp, marginal contribution, 4 portfolio profiles |
| [07_regulatory_engine.md](07_regulatory_engine.md) | DFAST 2026 verified, EBA illustrative, divergence findings (BBB +17-26%, Treasury +10-24%), paper language |
| [08_rl_adversarial.md](08_rl_adversarial.md) | 1-step MDP, reward function, 3 critical bug fixes, PPO v2-v5, bandit v1-v2, portfolio worst-case results, REST API |
| [09_experiments_and_results.md](09_experiments_and_results.md) | All 8 experiments, per-event table, ablation table, per-variable coverage, Wilcoxon tests, VECM, copula |
| [10_limitations_and_future_work.md](10_limitations_and_future_work.md) | COVID failure (53.3%), Brexit direction (33.3%), Wilcoxon non-significance, EBA not verified, RL gap vs beam search |

---

## Source of Truth Files

These files define the locked system state. Do not change a paper number without changing these first:

| File | Role |
|------|------|
| [ml_pipeline/canonical_paper_numbers.py](ml_pipeline/canonical_paper_numbers.py) | Every number in the paper. Includes self-check invariants run on import. |
| [ml_pipeline/canonical_best_model.py](ml_pipeline/canonical_best_model.py) | Runtime model configuration (df values, filter mode, training regimes, seeds). |
| [ml_pipeline/action_space/action_space.yaml](ml_pipeline/action_space/action_space.yaml) | Action space spec (locked 2026-05-05). Run `verify_action_space.py` before modifying. |
| [ml_pipeline/best_model_definition.md](ml_pipeline/best_model_definition.md) | Human-readable canonical model documentation. If it conflicts with canonical_paper_numbers.py, the .py wins. |
| [ml_pipeline/EBA_PAPER_FRAMING.md](ml_pipeline/EBA_PAPER_FRAMING.md) | Agreed paper language for DFAST (verified) vs EBA (illustrative) distinction. |

---

## Key Figures (for Paper)

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 1 | [ml_pipeline/figure_1_architecture.py](ml_pipeline/figure_1_architecture.py) | System architecture 4-layer diagram |
| Figure 2 | [ml_pipeline/figure_2_regime_timeline.py](ml_pipeline/figure_2_regime_timeline.py) | Regime classification timeline 2005-2026 |
| Figure 3 | [ml_pipeline/figure_3_causal_graph.py](ml_pipeline/figure_3_causal_graph.py) | Stressed-regime causal graph (330 edges) |
| Figure 4 | [ml_pipeline/figure_4_regime_graphs.py](ml_pipeline/figure_4_regime_graphs.py) | Calm vs stressed causal graph comparison |
| Figure 5 | [ml_pipeline/figure_5_precision_recall.py](ml_pipeline/figure_5_precision_recall.py) | Causal discovery precision@k curve |
| Figure 6 | [ml_pipeline/figure_6_covid_fan_chart.py](ml_pipeline/figure_6_covid_fan_chart.py) | COVID 2020 scenario fan chart |
| Figure 8 | [ml_pipeline/figure_8_per_event_heatmap.py](ml_pipeline/figure_8_per_event_heatmap.py) | Coverage × direction heatmap (11 events × 6 methods) |
| Figure 9 | [ml_pipeline/figure_9_dfast_divergence.py](ml_pipeline/figure_9_dfast_divergence.py) | DFAST 2026 divergence heatmap |

---

## Running the Experiments

```bash
# Run all 8 paper experiments (5-seed averaging, ~2-4 hours)
python ml_pipeline/all_paper_experiments.py --seed-runs 5

# Generate all paper figures
python ml_pipeline/generate_paper_figures.py

# Verify action space consistency (12 invariants)
python ml_pipeline/action_space/verify_action_space.py

# Verify DFAST ingestion
python ml_pipeline/dfast_verification.py

# Check canonical numbers self-check
python -c "import ml_pipeline.canonical_paper_numbers; print('All checks passed')"
```

---

## Abstract Draft (from canonical_paper_numbers.py)

> Financial stress testing relies on correlation-based models that underestimate tail risk during crises. We present CausalStress, an integrated system combining ensemble causal discovery (1,249 edges across 56 variables), 5-state regime-conditional causal graphs (211 contagion edges appear only during stress, with credit cascade amplification of 4.2× and bank-lending contagion of 6.9×), and a generative engine using data-fit Student-t innovations (df from MLE on pre-2020 VAR residuals, calm=5.97, crisis=3.84). On 7 held-out test events (2016-2023), our canonical model achieves 90.0% out-of-sample coverage with 100% pairwise consistency and 77.6% directional accuracy, significantly outperforming Historical Replay (p=0.008) and Unconditional VAR (p=0.031). Applied to DFAST 2026, our model projects 17-26% higher BBB credit stress than the Fed's correlation-based methodology. Code and data pipeline are publicly released.
