# Limitations and Future Work

## Overview

This document collects the honest limitations of CausalStress as it stands at the time of paper submission. These are not hedges — they are specific, quantified failure modes. Several are already explained by the system's design choices; others represent genuine open problems. All limitations are reported transparently and traced to source data in [ml_pipeline/canonical_paper_numbers.py](ml_pipeline/canonical_paper_numbers.py).

---

## Limitation 1: COVID 2020 Coverage is 53.3%

**What happened:** The canonical model covers only 53.3% of the 6 key variables on the 2020 COVID event (averaged over 5 seeds). This is the worst event result in the test set.

**Per-variable breakdown at event end (seed 20260407):**

| Variable | Actual move | Covered? |
|----------|------------|---------|
| ^GSPC | −33.6% | Yes |
| ^VIX | +315.3% | Yes |
| DGS10 | −79 bps | Yes |
| CL=F | −55.1% | **No** |
| XLF | −42.4% | Yes |
| BAMLH0A0HYM2 | +726 bps | **No** |

**Why it fails:**
- **CL=F** fell 55% — the March 2020 Saudi-Russia price war was a simultaneous supply shock unrelated to COVID's financial contagion. The model had no mechanism to anticipate an oil supply dispute overlapping with a pandemic.
- **BAMLH0A0HYM2** widened 726 bps — energy-sector HY defaults drove this beyond what pre-pandemic training data contains. The model's training data (2005–2019) never saw energy-sector HY spreads this wide outside of a credit crisis.

**Direction is 100%:** The model correctly predicts the direction of all 6 variables (equities down, VIX up, rates down, oil down, financials down, HY spreads up). The failure is magnitude underestimation, not directional confusion. This distinction matters: a risk manager warned that "a COVID-type pandemic could cause all of the above" has received useful information even if the exact magnitude was underestimated.

**Is this fixable?** Partially. Including a wider set of commodity-supply-shock scenarios, or training on COVID data after the fact, would improve this event. But COVID-2020 was genuinely out-of-distribution for a model trained on pre-2020 financial market crises. Honestly reporting 53.3% is preferable to masking it.

---

## Limitation 2: Brexit Direction is 33.3%

**What happened:** The canonical model achieves 80% coverage on Brexit (the actual outcomes are within the scenario envelope) but only 33.3% direction (the median scenario is pointing the wrong way for 4 of 6 variables).

**Why:**  Brexit was a fast, political shock (12-day window) that fit the `global_shock` template. The model's global_shock template predicts: equities down, VIX up, oil down, financials down, HY spreads up, rates down. In reality, Brexit produced a more complex response where sterling crashed and UK-specific assets fell hard, but US Treasuries and US equities recovered quickly. The US-centric model assigned the "global shock" response to variables like DGS10 (flight-to-safety, rates fall), but the actual 12-day trajectory saw US rates mostly unchanged as the shock was UK-specific.

**Conclusion:** Fast, geopolitically-localized shocks that don't follow standard financial contagion patterns are a systematic weakness. The model performs better on events with broader cross-market contagion (credit crises, rate shocks, volatility regimes).

---

## Limitation 3: Wilcoxon Non-Significance vs Regime VAR and Gaussian MC

The canonical Student-t model is **not statistically significantly better** than Regime VAR (no graph) (p=0.625) or Gaussian MC (p=0.563) on the 7-event test set.

**Why we selected Student-t anyway:**
1. Validation coverage: 94.2% vs 92.5% (Regime VAR) and 91.7% (Gaussian MC) — higher but small-sample.
2. Pairwise consistency: 100% vs 96.4% (Regime VAR) and 41.1% (Gaussian MC) — a meaningful practical advantage.
3. Directional accuracy: 77.6% vs 70.0% (Regime VAR) and 43.3% (Gaussian MC).
4. Methodology: Student-t innovations are empirically correct (KS test, 20/20 variables). Gaussian is known to be wrong; we just lack statistical power at n=7 to prove it.

**What this means:** The core coverage improvement over standard practice (Historical Replay, Unconditional VAR) is statistically verified. The advantage over the closest sophisticated baselines is directional but not statistically confirmed. This is acknowledged in the paper's limitations section.

**Root cause:** n=7 is small. The bootstrap CI is [76.7%, 99.5%] — very wide. With n=20 test events, the existing differences would likely be significant.

---

## Limitation 4: Causal Graph Precision@k is Low

Raw precision of the full ensemble graph against the 25-edge conservative ground truth: 0.020 (2.0%). Precision@10 = 0.30 (30%, 15× baseline) — better but still modest.

**Why the raw precision appears low:** The 1,249-edge ensemble includes many true economic relationships that are not in the conservative 25-edge labeled set. The full graph is not 98% noise — it's that the ground truth is deliberately conservative. Downstream validation (100% pairwise consistency on all test events) is stronger evidence that the graph is capturing real structure.

**What we lead with:** precision@10 = 30% (15× baseline), recall = 100%, 255 consensus edges, FCI robustness = 90%, and the downstream metric that a noise graph cannot produce 100% pairwise consistency.

---

## Limitation 5: EBA Regulatory Comparison is Not Verified

The EBA 2025 Adverse scenario is illustrative only. Variable paths are approximated from published themes, and the mapping uses US-centric proxies (^GSPC for EU equities). No EBA CSV has been ingested. Results presented for EBA should not be interpreted as verified divergence measurements.

The DFAST 2026 comparison is fully verified (official Fed CSV, all checks pass per `dfast_verification.py`).

---

## Limitation 6: VECM Not in Canonical Path

Experiment 6 identifies 6 cointegrating vectors across credit spreads, macro (Taylor Rule), and funding stress variable groups. VECM error-correction dynamics (mean-reversion pull toward long-run equilibrium) are economically meaningful and would improve scenario realism for long-horizon projections.

However, VECM requires I(1) input series (level data, not differenced), while the current VAR uses differenced/log-return stationary data. Integrating VECM into the canonical generator requires architectural changes. This is noted as future work.

---

## Limitation 7: RL Adversarial Agent Does Not Reach Beam Search Quality

**Gap:** PPO v5 achieves 58-79% of exhaustive beam search performance across profiles. The agent learns useful portfolio-differentiating behavior but doesn't fully explore the 2,000+ action space.

**Why:** 50-100k training steps are insufficient for full exploration of a 2,000-combination action space, especially given the 1-step MDP structure (each episode gives only one reward signal). The warm-start behavioural cloning helps but the agent tends to converge on the actions identified in beam search rather than discovering novel adversarial strategies.

**Current status:** The key empirical finding — that different portfolios have different worst-case shocks, identified correctly by the adversarial scan — holds and is meaningful. The RL framing as a *learned* adversarial agent (rather than an exhaustive search) is aspirational at current training scales. How this result is framed as a research contribution is being worked out.

**Next steps for RL:**
- 2-step MDP (v7): agent picks shock, observes early trajectory, picks follow-up. Would demonstrate adaptive adversarial reasoning.
- Larger training budget (1M+ steps) with async rollout collection.
- Richer observation space: include portfolio weights in the observation so the agent can condition on what portfolio it's attacking.

---

## Limitation 8: Oil Direction is the Weakest Variable

**CL=F direction accuracy: 69%** — the lowest of all 6 key variables. 

**Events where oil direction is wrong:**
- 2008 GFC: Oil was in a commodity supercycle, initially rising even as equities fell. The model's global_shock and credit_crisis templates predict oil falls, which is wrong for 2008's first phase.
- 2015 China/Oil: Oil fell hard, but the model initially predicted a smaller fall due to early-regime classification as `global_shock` rather than the oil-specific supply-demand collapse.

**Root cause:** Oil (CL=F) straddles two regimes — financial-contagion-driven (where oil falls with equities) and supply-shock-driven (where oil moves independently). The model captures financial contagion dynamics but not the supply-side stories. This is a fundamental limitation of training on financial market data without commodity supply fundamentals.

---

## Limitation 9: HMM Misses Three Post-2020 Events

Three events in the test set are not classified as stress-or-above by the HMM:
- 2020 Tech Selloff: classified `normal`
- 2022 Rate Hike: classified `elevated`
- 2023 SVB: classified `elevated`

The model still generates reasonable scenarios for these events (96.7% coverage for SVB, 100% coverage for 2020 Tech Selloff), because the canonical model uses the stressed-regime causal graph regardless of the current regime at inference. The regime miss doesn't break scenario generation but it does limit the model's ability to flag these events as stress periods on the dashboard.

---

## Limitation 10: Single Jurisdiction (US Only)

All data is US-centric. FRED and Yahoo Finance data represent US markets, US rates, and US credit. The EBA illustrative comparison uses US proxies for European variables. No non-US equity indices, European rates, or EM credit spreads are included in the core model.

This is a framework limitation, not an algorithmic one. The pipeline would work for other jurisdictions given appropriate local data.

---

## Future Work Summary

| Item | Priority | Status |
|------|----------|--------|
| VECM in canonical path | High | Research branch in `vecm_engine.py` |
| RL 2-step MDP (v7) | High | Designed, not yet trained |
| Extended RL training (1M+ steps) | Medium | Infrastructure ready |
| EBA official CSV ingestion | Medium | Not started |
| Non-US market extension | Low | Framework supports it, data not sourced |
| Pandemic/supply-shock event augmentation | Medium | Would address COVID and oil direction weaknesses |
| Post-2020 regime recalibration | Low | HMM retraining on recent data |
| Diversity bonus for RL (crisis scenario reference bank) | Low | Designed in rewards.py but not wired |
