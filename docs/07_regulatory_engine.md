# Regulatory Engine

## Overview

The regulatory engine compares CausalStress scenario projections against official regulatory stress tests. The primary benchmark is the DFAST 2026 Severely Adverse scenario, loaded directly from the Federal Reserve's official CSV release. A cross-jurisdictional example using EBA 2025 Adverse themes is included as an illustrative comparison only.

The key output is a **Causal Difference Report** — a document that shows not just where the causal model's projections diverge from the regulatory scenario, but traces the causal paths that explain those divergences.

---

## Implementation

| File | Purpose |
|------|---------|
| [ml_pipeline/regulatory/regulatory_engine.py](ml_pipeline/regulatory/regulatory_engine.py) | Scenario loading, causal model projection, divergence computation, difference reports |
| [ml_pipeline/dfast_verification.py](ml_pipeline/dfast_verification.py) | Validates DFAST CSV ingestion (run 2026-04-21, all checks passed) |
| [ml_pipeline/dfast_figure_extract.py](ml_pipeline/dfast_figure_extract.py) | Extracts DFAST divergence data for Figure 9 heatmap |
| [ml_pipeline/EBA_PAPER_FRAMING.md](ml_pipeline/EBA_PAPER_FRAMING.md) | Agreed-upon paper language for DFAST vs EBA scoping |

---

## DFAST 2026 Severely Adverse — Verified Benchmark

**Source:** Federal Reserve official CSV, final release.
**Scenario ID (in database):** `bc54571b-c7c7-4040-80db-4d23d30d1bb8`
**Verification script:** `dfast_verification.py`, run 2026-04-21 — all checks passed.
**Provenance tag:** `"Federal Reserve (Official CSV - Final)"`

The DFAST scenario publishes 9-quarter paths (quarterly resolution) for 5 variables:

| Variable | Mapping | DFAST Peak |
|----------|---------|-----------|
| GDP growth | (derived from macro variables) | −8% then recovery |
| Unemployment rate | UNRATE | 5.5% → 10.0% → 8.5% |
| BBB corporate spread | BAMLC0A4CBBB | 4.5% → 6.0% → 3.5% |
| S&P 500 | ^GSPC | 75 → 50 → 68 (index level) |
| VIX | ^VIX | 45 → 50 → 18 |

**Verified peak levels:** BBB = 8.20%, 10-Year Treasury = 3.10%, VIX = 72.00 (internal DB verified values, not rounded).

---

## Causal Difference Report

The CausalStress projection is run using the DFAST scenario starting conditions as the initial shock to the causal model. The causal model then propagates the shock through the graph over 9 quarters (approximated as sequential 60-day windows).

**Divergence computation:** For each variable in each quarter, the divergence is:

```
divergence_pct = (causal_projection - dfast_assumption) / |dfast_assumption| × 100
```

Positive divergence = causal model projects more severe path than Fed's assumption.
Negative divergence = causal model projects less severe path.

**Divergence severity classification:**
- `"more_severe"`: Causal model more stressed, divergence > +10%
- `"less_severe"`: Causal model less stressed, divergence < -10%
- `"ELEVATED"`: Large positive divergence (> +25%)
- `"DIVERGENT"`: Sign reversal (causal model predicts opposite direction)

**Report ID (in database):** `42d005d6-9f6f-4329-a1e1-1e81f7daa37b`
**Report date:** 2026-04-04

---

## Key Divergence Findings (Verified against Official DFAST CSV)

**Total significant divergence cells:** 34 (filtered to quarters where divergence exceeds ~10% threshold)

**BBB Corporate Yield:**
- All 13 quarters exceed the divergence threshold.
- Causal model projects BBB spreads **16.8% to 26.3% higher** than Fed's assumption across the 9-quarter horizon.
- Paper rounds to: **17–26% higher**.
- Interpretation: The causal model, trained on stressed-regime data where credit spread amplification is 4.2×, projects more severe credit stress than the Fed's correlation-based methodology.

**Treasury Yields:**
- 10-Year Treasury: 12 of 13 quarters exceed threshold. Range: **10.4% to 23.9% higher** than Fed assumption.
- 5-Year Treasury: 9 of 13 quarters. Range: 10.0% to 12.8% higher.
- 3-Month Treasury: 0 of 13 quarters (Fed directly controls short rates; no significant divergence).
- Total: 21 Treasury data points exceed the threshold.
- Paper rounds to: **10–24% higher** (aggregate across 5Y and 10Y tenors).

**Per-variable maximum divergences:**

| Variable | Max divergence |
|----------|---------------|
| BBB Corporate Yield | +26.3% |
| 10-Year Treasury Yield | +23.9% |
| 5-Year Treasury Yield | +12.8% |

**Why the causal model diverges upward:** The Fed's DFAST scenarios are designed as regulatory benchmarks, not as predictions. They apply correlation-based shock paths calibrated to be severe but not extreme. The CausalStress model, conditioned on stressed-regime causal amplification (bank lending contagion 6.9×, credit cascade 4.2×), projects that credit markets in a truly severe scenario would deteriorate faster than the DFAST path assumes. This is a feature, not a bug — it's the system providing evidence that the regulatory floor may underestimate certain tail risks.

---

## EBA 2025 Adverse — Illustrative Only

**Source:** Approximated from published EBA 2025 stress test themes. NOT loaded from official EBA CSV.

**Explicit caveats (verbatim from [ml_pipeline/EBA_PAPER_FRAMING.md](ml_pipeline/EBA_PAPER_FRAMING.md)):**

> The EBA 2025 Adverse scenario is not loaded from an official CSV in our system; variable paths are approximated from the published EBA themes and the variable mapping uses US-centric proxies (e.g. ^GSPC for EU equity prices). EBA results are presented for illustrative comparison only and are not used as a verified regulatory benchmark. A full EBA integration with the official EBA scenario data is noted as future work.

**What it shows:** An illustrative cross-jurisdictional comparison demonstrating that the CausalStress framework can, in principle, be applied to other regulatory scenarios. The variable approximations mean no numerical claim in the paper should be attributed to the EBA comparison.

---

## Paper Language

**One-sentence claim for methodology section:**

> Our regulatory comparison is anchored on the DFAST 2026 Severely Adverse scenario, loaded directly from the official Federal Reserve CSV; we additionally include an illustrative cross-jurisdictional example based on EBA 2025 adverse themes, with the explicit caveat that variable paths are approximated and the mapping uses US-centric proxies.

**Limitations paragraph for discussion:**

> Our regulatory validation focuses on a single jurisdiction (US, via DFAST 2026). The EBA cross-jurisdictional comparison is illustrative and uses approximated variable mappings; we do not claim verified equivalence to the official EBA 2025 stress test. Extending the verified-CSV ingestion pipeline to additional regulators (EBA, BoE, APRA) is a natural next step for the framework.

---

## What Changed in the Codebase to Support This Framing

1. `regulatory_engine.py` top docstring explicitly states DFAST-verified, EBA-illustrative scoping with public URLs for both.
2. `regulatory_engine.py` `EBA_2025_ADVERSE` dict updated: name = `"EBA 2025 Adverse Scenario (Illustrative)"`, source = `"European Banking Authority (Illustrative — Approximated)"`.
3. `figure_1_architecture.py` tightened `"DFAST/EBA alignment"` to `"DFAST 2026 verified"` in the architecture diagram, removing the implicit claim of EBA verification.

---

## Narrative Engine Integration

When the advisor endpoint (`/api/v1/advisor/chat`) is asked about regulatory scenarios, the narrative engine:

1. Calls `explain_regulatory_divergence(dfast_report, causal_model)` from [ml_pipeline/narrative/narrative_engine.py](ml_pipeline/narrative/narrative_engine.py).
2. Uses Claude (Anthropic API) to generate a plain-English explanation of WHERE and WHY the causal model diverges from the Fed's assumptions.
3. Traces the causal path: e.g., "The causal model predicts higher BBB spreads because the lending-standards contagion channel, which amplifies by 6.9× in crisis, transmits credit tightening from large corporates to small businesses faster than the Fed's model assumes."

---

## Output

**Database:**
```
regulatory.scenarios         (id, name, source, paths jsonb, quarters, variables)
regulatory.causal_difference_reports   (id, scenario_id, divergences jsonb,
                                        report_date, summary_text)
```

**JSON export:** `dfast_figure_data.json` — used by [ml_pipeline/figure_9_dfast_divergence.py](ml_pipeline/figure_9_dfast_divergence.py) to generate the divergence heatmap figure.
