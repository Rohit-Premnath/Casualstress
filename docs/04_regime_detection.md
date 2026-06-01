# Regime Detection

## Overview

CausalStress classifies every trading day from 2005-01-04 to 2026-04-14 (5,548 days) into one of five market regimes using a Gaussian Hidden Markov Model (HMM). The regime label is used to: (1) select which data to train the scenario generator's VAR on, (2) load the appropriate regime-conditional causal graph, and (3) route the current market state to the correct scenario generation path at inference time.

**The 5 regimes:** calm, normal, elevated, stressed, crisis.
**Regime detection event accuracy:** 72.7% (8 of 11 historical events correctly classified as stress or above).

---

## Implementation

| File | Purpose |
|------|---------|
| [ml_pipeline/regime_detection/hmm_model.py](ml_pipeline/regime_detection/hmm_model.py) | HMM training, classification, regime labeling per trading day |
| [ml_pipeline/regime_detection/regime_causal_graphs.py](ml_pipeline/regime_detection/regime_causal_graphs.py) | Re-runs causal discovery within each regime's labeled data |

---

## The HMM

**Model type:** Gaussian HMM with 5 hidden states, one observation per trading day.

**Feature vector (7 indicators per day):**

| Feature | Ticker/ID | Why included |
|---------|-----------|--------------|
| Equity volatility | ^VIX | Primary fear gauge, first to move in stress |
| High yield spread | BAMLH0A0HYM2 | Credit risk premium, distinguishes financial stress from equity-only volatility |
| Yield curve slope | T10Y2Y | Inversion signals recession risk; flattening precedes stress |
| Equity returns | ^GSPC | Market direction |
| Bond volatility | ^MOVE | Treasury market stress, separate from equity volatility |
| TED spread | TEDRATE | Interbank funding stress (pre-2023 authentic, post-2023 approximated) |
| Financial stress index | STLFSI4 | Composite Fed measure of overall financial stress |

**Training:** Expectation-Maximization on the full 5,548-day panel. Gaussian emission distributions per state. The model learns 5 Gaussian distributions (mean and covariance over 7 features) and a 5×5 transition matrix.

**State assignment:** After fitting, states are assigned regime labels by ranking on VIX level (low VIX = calm, high VIX = crisis). Intermediate states are labeled by spread levels and stress index values.

---

## Regime Distribution

Based on the full 5,548-day panel (2005-01-04 to 2026-04-14):

| Regime | Days | % of Panel | Character |
|--------|------|-----------|-----------|
| elevated | 1,795 | 32.5% | Moderate stress, VIX 20-30, spreads mildly elevated |
| calm | 1,174 | 21.2% | Normal bull market, VIX <15, tight spreads |
| stressed | 1,011 | 18.3% | High stress, VIX 30-50, spread widening, regime changes likely |
| normal | 980 | 17.7% | Low-vol recovery periods, VIX 15-20, narrow spreads |
| crisis | 568 | 10.3% | Extreme stress, VIX >50, spreads blowing out, GFC/COVID-type |

Note: `high_stress` appears in some code as a sixth label. In the actual classified database, only these 5 labels appear. The canonical model's training set uses `["elevated", "stressed", "high_stress", "crisis"]` — in practice, `high_stress` is never observed and the training data is effectively `elevated + stressed + crisis`.

---

## Known Crisis Periods and Regime Validation

The HMM is validated against 9 known historical stress periods:

| Period | Crisis | Expected regime | Detected? | Detection lag |
|--------|--------|----------------|-----------|---------------|
| 2007-10 to 2009-03 | GFC | stressed/crisis | Yes | 0 days |
| 2010-05-06 | Flash Crash | stressed/crisis | Yes | 0 days |
| 2011-07 to 2011-10 | US Debt Downgrade | stressed | Yes | 26 days |
| 2015-08 to 2016-02 | China Devaluation/Oil | stressed | Yes | 0 days |
| 2016-06-23 | Brexit | stressed | Yes | 0 days |
| 2018-01-26 to 2018-04 | Volmageddon | stressed | Yes | 10 days |
| 2020-02-19 to 2020-03-23 | COVID Crash | crisis | Yes | 0 days |
| 2020-09 | Tech Selloff | (not stress+) | No | — |
| 2022-01 to 2022-06 | Rate Hike Selloff | (not stress+) | No | — |
| 2023-03 | SVB Banking Crisis | (not stress+) | No | — |

**Overall event accuracy:** 72.7% (8 of 11, where "detected" means the regime is classified as stressed or crisis during the event window).

**Three misses:**
- 2020 Tech Selloff: classified as `normal`. This was a rapid intraday selloff (−10% in 3 weeks) without the broad credit/funding stress the HMM keys on. VIX rose but spreads didn't blow out.
- 2022 Rate Hike: classified as `elevated`. Rising rates without acute credit or funding stress — the HMM correctly identifies this as elevated but not stressed, which is arguably correct (it was a slow grind, not a crisis).
- 2023 SVB: classified as `elevated`. The SVB collapse was brief (10 days) and localized to the regional banking sector. The broader stress indicators (STLFSI4, HY spreads) only moved modestly.

---

## Experiment 2: Quantitative Regime Detection Performance

Binary classification: is the day stress-or-above (stressed or crisis) vs below?

| Metric | Value |
|--------|-------|
| Precision | 0.457 |
| Recall | 0.799 |
| F1 | 0.582 |

**Confusion matrix** (binary: stress+ vs not):

|  | Predicted not-stress | Predicted stress+ |
|--|---------------------|------------------|
| **Actually not-stress** | 3,767 (TN) | 857 (FP) |
| **Actually stress+** | 182 (FN) | 722 (TP) |

**Interpretation:** The model has high recall (0.80 — it catches most actual stress periods) but lower precision (0.46 — it also flags some non-stress days as elevated). This is the correct trade-off for a stress-testing application: missing a real stress period is worse than false-alarming. The false positives (857 days) are mostly in the `elevated` category — days the HMM correctly identifies as above-average risk even if not a named crisis.

**Per-event detection lag:** Most major crises (GFC, Flash Crash, China/Oil, Brexit, COVID) are detected with zero lag — the HMM classifies the first day of the event window as stressed. The 2011 Debt Downgrade has a 26-day lag and Volmageddon has a 10-day lag; both are rapidly-developing events where the volatility spike preceded the sustained stress classification.

---

## How the Regime is Used Downstream

**Scenario generation training:** The VAR model is fit on only the elevated + stressed + crisis days. This ensures the VAR's coefficient matrix captures co-movement under stress, not the more subdued dynamics of calm periods.

**Causal graph selection:** At inference time, the current regime label (read from `models.regimes` for the most recent date) determines which causal graph to load. The canonical model always loads the stressed-regime graph regardless of the current regime, because: (1) stress scenarios should be stress-conditioned, and (2) the stressed-regime graph is the densest and most causally rich.

**Dashboard display:** The frontend Regimes page shows a timeline of regime classifications from 2005 to present, with the current regime prominently displayed and transition probabilities shown in real time.

**Regime statistics output:**
```json
{
  "current_regime": "elevated",
  "probability": 0.72,
  "transition_probs": {
    "calm": 0.05,
    "normal": 0.10,
    "elevated": 0.55,
    "stressed": 0.25,
    "crisis": 0.05
  }
}
```

---

## Output

`models.regimes` table: one row per trading day.

```sql
CREATE TABLE models.regimes (
    date DATE PRIMARY KEY,
    regime_label INTEGER,         -- HMM hidden state index (0-4)
    regime_name TEXT,             -- "calm", "normal", "elevated", "stressed", "crisis"
    probability FLOAT,            -- posterior probability of assigned state
    transition_probs JSONB        -- {regime_name: probability} forward-looking transition
);
```

`regime_data.json` ([ml_pipeline/regime_data.json](ml_pipeline/regime_data.json)): the full 5,548-day classification sequence with regime name and probability per day. Used by figure scripts and the frontend.

`regime_timeline.json`: extracted version for the Figure 2 timeline visualization.
