# Best Model Definition

Final canonical winner:

- Model name: `causal_regime_multi_root_soft_filtered`
- Paper label: `Full Model (Soft Filtered)`
- Signature: `causal_regime_multi_root_soft_filtered | graph=stressed_full | filter=soft | multi_root=yes | train_regimes=elevated,stressed,high_stress,crisis`

Core components:

- Regime-conditioned training on `elevated`, `stressed`, `high_stress`, and `crisis`
- Stressed-regime causal graph loaded from `regime_causal_graphs.json`
- Graph mode: `full`
- Multi-root shock initialization from event-family templates
- Crisis covariance blending for larger shocks
- Soft plausibility filtering instead of hard top-k filtering

Canonical filter settings:

- Target scenarios: `200`
- Candidate multiplier: `2`
- Candidate scenarios generated: `400`
- Filter mode: `soft_plausibility`
- Soft-filter weights: `score ** 6.0`
- Weighted quantiles used for backtest evaluation: `5th`, `50th`, `95th`

Why this model won:

- It preserved high coverage while improving directional accuracy versus the non-filtered and hard-filtered graph variants.
- It kept perfect or near-perfect pairwise crisis consistency in backtests.
- It was more stable than the hard-filtered winner family because soft filtering retained more useful scenario diversity.

Alignment note:

- `multi_backtest_v3.py` uses this as the canonical winner variant.
- `scenario_generator.py` uses this as the production generation path.
- `all_paper_experiments.py` reports this model as `Full Model (Soft Filtered)`.
