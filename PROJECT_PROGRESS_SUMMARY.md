# Project Progress Summary

## Overview
This phase focused on understanding the current app, improving the scenario engine, and proving which configuration works best through backtesting.

The biggest conclusion is:

- the project already has a strong quantitative core in `ml_pipeline`
- the engine is now much more realistic than before
- the best-performing setup is backed by repeated backtests
- the frontend/backend product layer still needs to be wired cleanly to this engine

## Initial App Assessment
We reviewed the app structure and found:

- `backend/app` is still minimal scaffolding
- `frontend` is still mostly mockup-stage
- the real substance is in `ml_pipeline`

The strongest existing modules were:

- causal discovery
- regime detection
- scenario generation
- portfolio/risk logic
- regulatory and narrative support layers

So the project is best described as:

- a strong research/quant prototype
- not yet a fully productized app

## Main Goal
The main objective of this phase was to improve the crisis scenario engine and determine, with evidence, which configuration best reproduces historical market stress events.

Key questions:

- Does the causal graph really help?
- Does regime-aware training help?
- Does starting from multiple simultaneous shocks help?
- Which configuration should become the default engine?

## Early Finding
One major issue was discovered early:

- the original `multi_backtest.py` was not actually using the causal graph in its core backtest path

That meant the backtest was mostly validating a VAR-style simulator, not the full causal-stress thesis.

Fixing that was the first important step.

## Changes Made

### 1. Backtest Framework Upgrades
File:

- `ml_pipeline/multi_backtest.py`

We turned the backtest into a real ablation framework.

Changes included:

- added true causal graph usage in backtesting
- added stressed-regime graph testing
- added regime-conditioned training
- added event-family multi-root shock templates
- added extra holdout crises
- added stronger benchmark metrics
- added filtered-scenario variants

Variants tested across the process included:

- `baseline_no_causal`
- `causal_global`
- `causal_stressed`
- `baseline_regime_stress`
- `causal_regime_stress`
- `baseline_regime_multi_root`
- `causal_regime_multi_root`
- `causal_regime_multi_root_pruned`
- `causal_regime_multi_root_filtered`
- `causal_regime_multi_root_pruned_filtered`

### 2. Scenario Generator Upgrades
File:

- `ml_pipeline/generative_engine/scenario_generator.py`

We brought the winning ideas from backtesting into the real generator.

Changes included:

- default stressed-regime training
- multi-root event-family shock templates
- stressed causal graph propagation
- shock persistence so crisis impulse does not disappear after day 1
- event-specific guardrails
- VIX damping to reduce unrealistic blowups
- cleaner scenario labeling/storage

Currently supported scenario families:

- `global_shock`
- `market_crash`
- `rate_shock`
- `credit_crisis`

## Core Modeling Improvements

### Regime-Aware Training
Instead of fitting equally on all history, the engine now focuses on:

- `elevated`
- `stressed`
- `high_stress`
- `crisis`

Why this matters:

- stressed markets behave differently from calm markets
- crisis simulation should learn mainly from crisis-like periods

### Multi-Root Design
Instead of starting a crisis from one variable only, each scenario family starts from a realistic bundle of simultaneous shocks.

Examples:

- market crash: equities down, VIX up, financials down, spreads wider
- credit crisis: equities down, bank stocks down, spreads wider, yields lower
- global shock: equities down, oil down, VIX up

Why this matters:

- real crises usually hit multiple parts of the system at once

### Causal Graph Usage
The stressed causal graph is now used in both backtest variants and the scenario generator.

Why this matters:

- it helps propagate the shock through related variables in a more coherent way

Important nuance:

- causal graphs help
- but the largest performance gains came from regime-aware training and multi-root shocks

### Shock Persistence
We found the crisis impulse was being washed out too quickly.

Fix:

- the initial shock now decays over several days instead of disappearing immediately

This made outputs much more believable, especially for:

- market crash
- credit crisis
- global shock

### Guardrails
We added event-specific guardrails to keep variables from behaving too unrealistically.

Examples:

- reduced VIX overshoot
- limited excessive oil drift in some families
- added constraints for rates/spreads in certain templates

## Stronger Evaluation Metrics
We expanded the benchmark beyond just coverage and direction.

New metrics:

- `Coverage`: whether actual outcomes landed inside the predicted range
- `Direction`: whether the median scenario got the sign right
- `MedAE`: median absolute error between predicted median and actual move
- `TailMiss`: severity of misses when actual outcomes fall outside the range
- `Pairwise`: whether scenarios preserve the expected crisis shape
- `Benchmark Score`: combined score across the above

This gave us much stronger evidence for choosing the winning engine design.

## Experiments That Did Not Win

### Learned Shock Templates
We tested learned shock templates derived from prior same-family events.

Result:

- interesting idea
- but overall worse than the hand-tuned multi-root templates

Main reasons:

- some families had too few prior examples
- some learned templates became too weak
- family medians were noisy and sometimes under-represented severe crises

This was rolled back as the default direction.

## Best Current Backtest Result
The best current research configuration is:

- `causal_regime_multi_root_filtered`

That means:

- regime-conditioned training
- multi-root shock initialization
- stressed causal graph propagation
- plausibility-based scenario filtering

Latest best-variant backtest result:

- `87.9%` coverage
- `80.3%` direction accuracy
- `80.5` benchmark score

Interpretation:

- very strong on classic financial crises
- strong on volatility, banking, and many rate-driven stress events
- still weaker on unusual political shocks like Brexit
- improved on COVID, but not fully solved

## Latest Best-Variant Event Results
For `causal_regime_multi_root_filtered`, recent backtest performance looked like this:

- `2008 Global Financial Crisis`: `83%` coverage, `83%` direction
- `2010 Flash Crash / Euro Stress`: `100%` coverage, `100%` direction
- `2011 US Debt Downgrade / Euro Crisis`: `100%` coverage, `100%` direction
- `2015 China Devaluation / Oil Crash`: `67%` coverage, `67%` direction
- `2016 Brexit Shock`: `83%` coverage, `33%` direction
- `2018 Volmageddon`: `100%` coverage, `83%` direction
- `2018 Q4 Fed Tightening Selloff`: `100%` coverage, `83%` direction
- `2020 COVID Crash`: `33%` coverage, `100%` direction
- `2020 September Tech Selloff`: `100%` coverage, `67%` direction
- `2022 Rate Hike Selloff`: `100%` coverage, `67%` direction
- `2023 SVB Banking Crisis`: `100%` coverage, `100%` direction

Simple interpretation:

- excellent on many endogenous financial crises
- good on several banking and rate events
- weaker on Brexit-like fast political shocks
- COVID is directionally strong now but still under-captured in range

## Current Scenario Generator Behavior
The real generator now:

- loads processed data and regime labels
- fits on stressed-regime history
- loads the stressed causal graph
- creates multi-root scenario templates
- simulates 100 paths per event family
- stores scenarios in the database

Recent real run output looked like:

- `global_shock`
  - median `^GSPC`: `-3.9%`
  - median `CL=F`: `-18.6%`
  - median `XLF`: `-9.4%`

- `market_crash`
  - median `^GSPC`: `-10.2%`
  - median `^VIX`: `+64.9%`
  - median `XLF`: `-25.6%`

- `rate_shock`
  - median `DGS10`: `+15bps`
  - median `^GSPC`: `-3.2%`
  - median `XLF`: `-12.3%`

- `credit_crisis`
  - median `^GSPC`: `-10.8%`
  - median `^VIX`: `+59.3%`
  - median `XLF`: `-27.7%`
  - median `BAMLH0A0HYM2`: `+53bps`

Simple interpretation:

- `market_crash` and `credit_crisis` are currently the strongest-looking families
- `global_shock` is credible
- `rate_shock` is directionally right but still milder than the other templates

## Current Product Status
What exists now:

- a strong engine
- clear best-performing research configuration
- credible scenario generation
- database storage for generated scenarios and backtest results

What is not yet fully implemented:

- a clean FastAPI endpoint for the scenario page
- user-facing request/response models
- frontend-to-engine wiring for scenario generation
- clean support for user inputs like severity, horizon, and portfolio selection

So the engine is ready to power the app, but the product/API layer still needs to be implemented cleanly.

## Current Best Summary
The project has moved from:

- a promising research prototype

to:

- a backtest-supported crisis scenario engine with a clear best-performing design

## Recommended Next Steps
Suggested next steps from here:

1. Expose the winning scenario engine through a backend API endpoint
2. Connect the frontend scenario page to real engine inputs
3. Make filtering softer rather than hard top-path selection
4. Continue improving weak families like Brexit/COVID-style shocks
5. Connect scenario generation with portfolio stress outputs for user-facing results

## Final Plain-English Wrap-Up
Where the project stands now:

- the engine is real
- it is materially better than where it started
- we know which configuration works best
- we have evidence from backtesting
- the next phase is productization and final polish, not rebuilding from scratch
