# Benchmark Table — Locked Numbers
**Locked:** 2026-05-21  
**Eval protocol:** 16 held-out seeds (20000–20015), beam width 6, UCB β=0.5  
**Metric:** % of exhaustive beam search portfolio loss (higher = better)  
**Gate:** ≥ 85% qualifies for production use

---

## Table 1: Adversarial Search Quality — % of Beam Search Portfolio Loss

| Method | balanced | bond_heavy | tech_heavy | credit_heavy |
|---|:---:|:---:|:---:|:---:|
| **Random** | 30.0% | 33.8% | 30.0% | 29.1% |
| **Heuristic (max-magnitude)** | 64.0% | 62.0% | 68.0% | 65.1% |
| **Bandit v1 (1-step)** | 53.0% | 74.1% | 60.9% | 46.2% |
| **Bandit v2 (2-step)** | 62.6% | **86.4% ✓** | 77.1% | 69.6% |
| **Beam search (oracle)** | 100% | 100% | 100% | 100% |

### v1 → v2 Gain (percentage points)
| Profile | v1 | v2 | Gain |
|---|:---:|:---:|:---:|
| balanced | 53.0% | 62.6% | +9.6 pp |
| bond_heavy | 74.1% | 86.4% | +12.3 pp |
| tech_heavy | 60.9% | 77.1% | +16.2 pp |
| credit_heavy | 46.2% | 69.6% | +23.4 pp |

2-step MDP consistently improves over 1-step across all profiles (avg +15.4 pp).

---

## Table 2: Best Adversarial Sequence Found (v2 Greedy)

| Profile | Bandit top sequence | Beam top sequence |
|---|---|---|
| balanced | XLK −5.0σ → XLF −5.0σ | XLE −5.0σ → XLY −5.0σ |
| bond_heavy | TLT −5.0σ → LQD −5.0σ | BAMLC0A0CM +4.5σ → ^RUT −5.0σ |
| tech_heavy | XLK −5.0σ → ^NDX −5.0σ | FEDFUNDS +5.0σ → UNRATE +5.0σ |
| credit_heavy | XLF −5.0σ → XLY −5.0σ | XLV −5.0σ → XLF −5.0σ |

Note: bandit and beam find different sequences — the bandit finds financially
interpretable sequences (sector-to-sector contagion) while beam occasionally
finds macro-financial chains. Both are valid adversarial pathways; the
bandit's sequences are more directly useful to a practitioner.

---

## Table 3: Supporting Metrics (v2, Greedy, mean over 16 seeds)

| Profile | Portfolio Loss | Causal Fidelity | DFAST Breach Rate | Diversity |
|---|:---:|:---:|:---:|:---:|
| balanced | 0.696 | 0.634 | 0.348 | 0.922 |
| bond_heavy | 0.594 | 0.679 | 0.568 | 0.905 |
| tech_heavy | 1.020 | 0.660 | 0.450 | 0.944 |
| credit_heavy | 0.660 | 0.653 | 0.311 | 0.934 |

Causal fidelity ≈ 0.63–0.68 across all profiles: shocks follow the
empirically estimated causal graph at ~2/3 of steps, consistent with
the structural constraint imposed during training.

---

## Source Files

All numbers extracted from (not modified):
```
ml_pipeline/runs/bandit_v1_{profile}/heldout_results_1step.json   — v1 native eval
ml_pipeline/runs/bandit_v2_{profile}/heldout_results.json          — v2 native eval
```

Random and heuristic baselines computed as:
`method.mean_portfolio_loss / beam.mean_portfolio_loss × 100`
using v2 beam as the reference (consistent denominator across all rows).

v1 best = max(greedy_vs_beam_pct, ucb_vs_beam_pct) from heldout_results_1step.json  
v2 best = max(greedy_vs_beam_pct, ucb_vs_beam_pct) from heldout_results.json
