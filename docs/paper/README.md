# Paper Materials — Adversarial RL Layer
**Status:** Locked for writing  
**Last updated:** 2026-05-21

## Files

| File | Contents | Status |
|---|---|---|
| `00_benchmark_table.md` | All benchmark numbers (random / heuristic / v1 / v2 / beam), all 4 profiles | **LOCKED** |
| `01_crisis_alignment_scan.py` | Script that scans seeds → finds crisis dates → runs bandit | Reproducible |
| `02_crisis_alignment_results.md` | Output of scan + alignment verdicts for GFC/COVID/2022 | **LOCKED** |
| `03_cross_profile_narrative.md` | Paper-ready text for Experiments section 5.4 | **LOCKED** |
| `04_sections_3_and_4_draft.md` | Section 3 (Problem Formulation) + Section 4 (Method) — forensically verified (v2 corrected) | Draft |
| `05_section_5_draft.md` | Section 5 (Experiments & Results) — forensically verified against JSON result files | Draft |
| `06_sections_1_and_2_draft.md` | Section 1 (Introduction) + Section 2 (Related Work) | Draft v2 |
| `07_section_6_conclusion.md` | Section 6 (Conclusion) — synthesises Sections 1–5, no new claims | Draft |

## Do not modify locked files without re-running the eval

Benchmark numbers come from:
- `ml_pipeline/runs/bandit_v1_{profile}/heldout_results_1step.json`
- `ml_pipeline/runs/bandit_v2_{profile}/heldout_results.json`

Crisis alignment results come from running `01_crisis_alignment_scan.py`
inside the Docker container (`docker exec causalstress-api bash -c "cd /ml_pipeline && python 01_crisis_alignment_scan.py"`).

## Key numbers to remember

- **Gate threshold:** 85% of beam search quality
- **Only profile clearing the gate:** bond_heavy at **86.4%**
- **v1→v2 average gain:** +15.4 pp across all profiles
- **Best historical alignment:** Rate shock 2022 (bond_heavy TLT→LQD ✓, tech_heavy XLK→^NDX ✓)

## One-sentence paper claim

> Neural contextual bandit achieves 86.4% of exhaustive beam-search quality on
> bond-heavy portfolios, with demonstrated alignment to three historical crisis
> transmission mechanisms, at sub-second inference time.
