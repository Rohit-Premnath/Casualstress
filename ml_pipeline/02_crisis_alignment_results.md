# Crisis Alignment Results
**Generated:** 2026-05-21 06:21
**Seeds scanned:** 50000–50499 (500 total)
**Profiles:** bond_heavy, tech_heavy, balanced, credit_heavy

---

## Summary

For each crisis period, the 3 seeds whose historical starting date is
closest to the crisis date are selected. The bandit is then run on those
seeds. If the pathway it discovers matches the historically observed
transmission mechanism, that is evidence of historical-crisis alignment.

## GFC_peak — Oct 2008

**Historical context:** Lehman Brothers filed Ch.11 on Sep 15 2008. By Oct 10, VIX hit 80, interbank credit markets frozen. Expected bandit: credit spread shocks (HYG/LQD/BAMLH), financial sector contagion (XLF). Key variables: BAMLH0A0HYM2 (HY spreads), XLF, ^GSPC, TLT (flight-to-safety bid).

### Closest seeds

| Seed | Sampled Date | Gap (days) |
|---|---|---|
| 50199 | 2009-05-27 | 229 |
| 50287 | 2009-05-27 | 229 |
| 50022 | 2009-05-28 | 230 |

### Bandit pathways by profile

| Profile | Bandit sequence | Portfolio loss | Causal fidelity |
|---|---|:---:|:---:|
| bond_heavy | TLT -5.0σ → XLU -5.0σ | 0.453 | 0.664 |
| tech_heavy | ^NDX -5.0σ → XLY -5.0σ | 0.842 | 0.475 |
| balanced | XLK -5.0σ → XLY -4.5σ | 0.906 | 0.770 |
| credit_heavy | XLF -5.0σ → TLT -5.0σ | 0.539 | 0.631 |

### Alignment verdict

<!-- Fill this in manually after reviewing the table above -->
<!-- Does the bond_heavy bandit fire TLT-type shocks for Rate_shock_22? -->
<!-- Does tech_heavy fire XLK/QQQ shocks for COVID_crash? -->

---

## COVID_crash — Mar 2020

**Historical context:** S&P 500 fell 34% in 33 days (Feb 19 – Mar 23 2020). Volatility spike: VIX >80. Broad equity selloff, credit spreads blew out, then massive Fed intervention. Expected bandit: broad equity shocks (^GSPC, XLK, XLY), VIX spike, credit spread widening.

### Closest seeds

| Seed | Sampled Date | Gap (days) |
|---|---|---|
| 50435 | 2020-05-11 | 52 |
| 50419 | 2020-05-19 | 60 |
| 50113 | 2020-05-20 | 61 |

### Bandit pathways by profile

| Profile | Bandit sequence | Portfolio loss | Causal fidelity |
|---|---|:---:|:---:|
| bond_heavy | TLT -5.0σ → LQD -5.0σ | 0.532 | 0.566 |
| tech_heavy | ^NDX -5.0σ → XLY -5.0σ | 1.287 | 0.451 |
| balanced | XLF -5.0σ → XLK -5.0σ | 0.247 | 0.557 |
| credit_heavy | XLF -5.0σ → TLT -5.0σ | 0.435 | 0.484 |

### Alignment verdict

<!-- Fill this in manually after reviewing the table above -->
<!-- Does the bond_heavy bandit fire TLT-type shocks for Rate_shock_22? -->
<!-- Does tech_heavy fire XLK/QQQ shocks for COVID_crash? -->

---

## Rate_shock_22 — Oct 2022

**Historical context:** Fed hiked 425bp in 2022 — fastest cycle in 40 years. 20Y Treasury (TLT) fell ~40% YTD by Oct 2022. Both bonds AND equities down (unusual). Expected bandit (bond_heavy portfolio): TLT shock → bank/financial contagion (XLF), because rising rates compress bank NII and mark bond portfolios to market. Expected for tech_heavy: rate-sensitive growth stocks (QQQ/XLK) re-rated sharply.

### Closest seeds

| Seed | Sampled Date | Gap (days) |
|---|---|---|
| 50483 | 2022-05-23 | 144 |
| 50295 | 2022-05-19 | 148 |
| 50280 | 2022-05-18 | 149 |

### Bandit pathways by profile

| Profile | Bandit sequence | Portfolio loss | Causal fidelity |
|---|---|:---:|:---:|
| bond_heavy | TLT -5.0σ → LQD -5.0σ | 0.503 | 0.557 |
| tech_heavy | XLK -5.0σ → ^NDX -5.0σ | 0.710 | 0.484 |
| balanced | XLK -4.5σ → XLY -5.0σ | 0.701 | 0.582 |
| credit_heavy | XLF -5.0σ → TLT -5.0σ | 0.742 | 0.615 |

### Alignment verdict

<!-- Fill this in manually after reviewing the table above -->
<!-- Does the bond_heavy bandit fire TLT-type shocks for Rate_shock_22? -->
<!-- Does tech_heavy fire XLK/QQQ shocks for COVID_crash? -->

---
