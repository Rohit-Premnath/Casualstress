# EBA Scoping Statement — Paper-Ready Snippets

## Purpose
This file documents the agreed-upon scoping language for DFAST vs EBA in the
ICAIF paper. The codebase has been updated to match this framing
(`regulatory/regulatory_engine.py`, `figure_1_architecture.py`).

---

## One-sentence claim (for abstract or related-work)

> Our regulatory comparison is anchored on the DFAST 2026 Severely Adverse
> scenario, loaded directly from the official Federal Reserve CSV; we additionally
> include an illustrative cross-jurisdictional example based on EBA 2025 adverse
> themes, with the explicit caveat that variable paths are approximated and the
> mapping uses US-centric proxies.

---

## Methodology paragraph (for Experiments section, Sub-section: Regulatory Validation)

> We evaluate our scenario engine against the DFAST 2026 Severely Adverse
> scenario, ingested directly from the Federal Reserve's official CSV release.
> The variable mapping is one-to-one for the five quarterly paths the Fed
> publishes (GDP growth, unemployment, BBB corporate spread, equity prices,
> VIX). We report divergences using ranges over [report_id], where positive
> divergence indicates the causal model projects a more severe path than the
> Fed assumption.
>
> We additionally include the EBA 2025 Adverse scenario as an illustrative
> cross-jurisdictional example. Unlike DFAST, the EBA scenario is not loaded
> from an official CSV in our system; variable paths are approximated from
> the published EBA themes and the variable mapping uses US-centric proxies
> (e.g. ^GSPC for EU equity prices). EBA results are presented for
> illustrative comparison only and are not used as a verified regulatory
> benchmark. A full EBA integration with the official EBA scenario data is
> noted as future work.

---

## Limitations paragraph (for Discussion section)

> Our regulatory validation focuses on a single jurisdiction (US, via DFAST
> 2026). The EBA cross-jurisdictional comparison is illustrative and uses
> approximated variable mappings; we do not claim verified equivalence to
> the official EBA 2025 stress test. Extending the verified-CSV ingestion
> pipeline to additional regulators (EBA, BoE, APRA) is a natural next step
> for the framework.

---

## What changed in the codebase to support this framing

1. `regulatory/regulatory_engine.py` — top docstring rewritten to explicitly
   state the DFAST-verified, EBA-illustrative scoping, including the public
   URL for both regulator publications.
2. `regulatory/regulatory_engine.py` — FALLBACK_SCENARIOS section comment
   tightened to clarify that DFAST has an official-CSV path and EBA does not.
3. `regulatory/regulatory_engine.py` — `EBA_2025_ADVERSE` dict updated:
   - `name` now `"EBA 2025 Adverse Scenario (Illustrative)"`
   - `source` now `"European Banking Authority (Illustrative — Approximated)"`
   - `description` now explicitly states this is not a verified comparison
4. `figure_1_architecture.py` — the AI Risk Narratives bullet
   `"DFAST/EBA alignment"` was tightened to `"DFAST 2026 verified"`,
   removing the implicit claim of EBA verification from the architecture
   figure.