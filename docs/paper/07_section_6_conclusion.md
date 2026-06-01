# Section 6 Draft — Conclusion
**Status:** Draft — draws only from verified claims in Sections 1–5; no new numbers introduced  
**Grounded in:**
- `docs/paper/04_sections_3_and_4_draft.md` (v4)
- `docs/paper/05_section_5_draft.md` (v3)
- `docs/paper/06_sections_1_and_2_draft.md` (v2)  
**Last updated:** 2026-05-25

---

## 6  Conclusion

We have presented an adversarial stress testing system that inverts the standard
conditional-scenario paradigm: rather than asking what losses a fixed scenario
causes, the system searches for the shock sequence that causes the greatest loss
for a specific portfolio, subject to an empirically estimated causal graph
constraint.

The central technical contribution is BanditRewardNet, a neural contextual bandit
trained by supervised reward regression on exhaustively labelled action-state
triplets. Framing adversarial search as a 2-step MDP and replacing PPO
policy-gradient training with dense reward supervision resolves the degenerate
collapse observed across four successive PPO iterations: the bandit outperforms
the domain-expert heuristic on three of four portfolio profiles and clears the
$\geq 85\%$ deployment gate on the bond\_heavy profile (86.4\% of beam-search
reference quality), with an average v1→v2 gain of +15.4 percentage points across
profiles.

Historical crisis alignment — tested without any crisis-specific supervision — shows
strong-to-partial correspondence with the observed transmission mechanisms of the
2022 rate shock, the COVID-19 crash, and the 2008 Global Financial Crisis,
providing evidence that the reward-shaped adversarial search recovers financially
grounded pathways rather than statistical artefacts.

**Limitations.** Three profiles do not clear the deployment gate: balanced (62.6\%),
tech\_heavy (77.1\%), and credit\_heavy (69.6\%). The beam-search reference itself
covers only 36 of 62,500 possible 2-step sequences, so the gate threshold is
relative to a tractable reference, not a global optimum. The training distribution
spans 1990–2024; novel crisis types without historical precedent may not be
well-represented. Portfolio profile routing uses an argmax of three exposure
dimensions, which does not handle hybrid concentration well.

**Future work.** The most direct extensions are: (1) extending the 2-step MDP to
longer horizons ($T > 2$) to capture multi-stage contagion chains; (2) training
profile-agnostic models conditioned on continuous exposure fingerprints rather than
discrete profiles; (3) adding an explicit confidence interval over the
$\%_{\mathrm{beam}}$ metric from bootstrap resampling of the 16 held-out seeds;
and (4) integrating the verified-CSV regulatory pipeline for additional
jurisdictions (EBA, BoE) alongside the existing DFAST comparison.

The broader takeaway is methodological: for short-horizon adversarial search over
a structured discrete action space with a computable reward, dense supervised
reward regression from exhaustive action labelling consistently outperforms sparse
policy-gradient methods, and the resulting policy admits interpretable causal
attribution of discovered worst-case pathways.

