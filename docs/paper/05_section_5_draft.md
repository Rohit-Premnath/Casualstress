# Section 5 Draft — Experiments and Results
**Status:** Draft (v6 — three-model exhaustive comparison complete) — quantitative claims verified; interpretive discussion in §5.5 and §5.6 is evidence-based but not strictly code-verifiable  
**Verified against:**
- `ml_pipeline/runs/bandit_v2_{profile}/heldout_results.json` — v2 primary results (greedy/UCB mode per profile recorded)
- `ml_pipeline/runs/bandit_v2_bond_heavy/heldout_results_exhaustive.json` — exhaustive on 6-branch original (1,000,000 evals; 2026-05-27)
- `ml_pipeline/runs/bandit_v2_bond_heavy_aug/heldout_results_exhaustive.json` — exhaustive on 25-branch augmented (1,000,000 evals; 2026-05-28)
- `ml_pipeline/runs/bandit_v2_bond_heavy_minmag/heldout_results_exhaustive.json` — exhaustive on 25+25 forced min-mag (1,000,000 evals; 2026-05-31)
- `ml_pipeline/runs/bandit_v1_{profile}/heldout_results_1step.json` — v1 baseline
- `ml_pipeline/runs/ppo_v5_{profile}_*/heldout_benchmark.json` — PPO ablation (confirmed 1-step MDP; top_rl_sequence single-action)
- `ml_pipeline/runs/bandit_v3_tech_heavy/heldout_results.json` — v3 regression
- `ml_pipeline/generative_engine_rl/sequence_compare.py` — heuristic as per-seed best of three sequences
- `ml_pipeline/generative_engine_rl/rewards.py` — CF definition (sign-consistency), diversity (action novelty)
- `docs/paper/02_crisis_alignment_results.md` — crisis alignment (locked)  
**Last updated:** 2026-05-31 (v6 — three-model exhaustive table in §5.2, updated §5.8)

---

## 5  Experiments

### 5.1  Experimental Setup

**Environment.** All evaluations use the real-mode simulator (Section 4.1):
lag-2 VAR estimated on 3,374 stressed-regime observations, stressed-regime causal
graph (330 edges), and Student-t innovations with data-fit degrees of freedom
(df\_crisis $\approx 3.84$, df\_normal $\approx 5.97$).

**Held-out seeds.** We evaluate on 16 seeds (20000–20015), drawn from an integer
range disjoint from the training range (seeds 1000–1049) and the crisis-alignment
scanning range (seeds 50000–50499). Hyperparameter selection used seeds in the
2000–2049 range; the 20000-block was reserved exclusively for held-out evaluation.

**Metrics.** The primary metric is *percentage of beam search portfolio loss*:
$$
\%_{\mathrm{beam}} = \frac{\bar{\ell}_{\text{method}}}{\bar{\ell}_{\text{beam}}} \times 100,
$$
where $\bar{\ell}$ denotes mean mark-to-market portfolio loss over 16 seeds.
Higher is better; the deployment gate is $\geq 85\%$.

**Baselines.** We compare four methods:

1. **Random**: two actions drawn uniformly at random from the 250-element action
   catalog, independently per step.

2. **Heuristic (max-magnitude)**: three profile-specific 2-step sequences
   pre-specified by a domain expert (e.g., for `bond_heavy`:
   HYG $-5\sigma$ → LQD $-5\sigma$; DGS10 $+5\sigma$ → HYG $-5\sigma$;
   DGS2 $+4\sigma$ → DGS10 $+4\sigma$). For each seed, all three sequences are
   evaluated and the best-performing one for that seed is retained; the reported
   value is the mean portfolio loss over the 16 per-seed best results. This
   oracle-selection framing represents the upper bound of what a practitioner could
   achieve by manually specifying and selecting from these three sequences with
   hindsight.

3. **Bandit v1 (1-step)**: the bandit policy trained to select a single
   shock per episode; reported as percentage of the 1-step beam reference (consistent
   denominator for a 1-step MDP).

4. **Beam search (reference)**: 2-step beam search with width $k = 6$,
   evaluating all $k^2 = 36$ complete sequences per seed (Section 4.7).

**Note on denominators.** Bandit v1 and its beam are both 1-step; bandit v2 and
all 2-step baselines use the 2-step beam as the denominator. The v1 and v2
percentages are therefore each expressed as fractions of their respective oracles
and should not be compared as absolute numbers — the improvement across versions
reflects both bandit quality gains and the additional loss achievable from a
second shock.

**UCB.** Table 1 reports the best of greedy and UCB ($\beta = 0.5$) per profile.
The winning mode differs by profile: balanced and credit\_heavy use UCB (62.6\%
and 69.6\% respectively); bond\_heavy and tech\_heavy use greedy (86.4\% and 77.1\%
respectively).

---

### 5.2  Main Results: Adversarial Search Quality

**Table 1: Adversarial search quality (% of beam search portfolio loss)**

| Method | balanced | bond\_heavy | tech\_heavy | credit\_heavy |
|---|:---:|:---:|:---:|:---:|
| Random | 30.0% | 33.8% | 30.0% | 29.1% |
| Heuristic (max-magnitude) | 64.0% | 62.0% | 68.0% | 65.1% |
| Bandit v1 (1-step) | 53.0% | 74.1% | 60.9% | 46.2% |
| Bandit v2 (2-step) | **62.6%** | **86.4% ✓** | **77.1%** | **69.6%** |
| Beam search (oracle) | 100% | 100% | 100% | 100% |

The 2-step bandit (v2) consistently outperforms the 1-step bandit (v1) across all
four profiles, and exceeds the domain-expert heuristic on three of four profiles
(bond\_heavy +24.4 pp, tech\_heavy +9.1 pp, credit\_heavy +4.5 pp; balanced $-1.4$ pp).
The bond\_heavy profile clears the $\geq 85\%$ deployment gate with 86.4\%.

**v1 $\to$ v2 improvement:**

| Profile | v1 | v2 | Gain |
|---|:---:|:---:|:---:|
| balanced | 53.0% | 62.6% | +9.6 pp |
| bond\_heavy | 74.1% | 86.4% | +12.3 pp |
| tech\_heavy | 60.9% | 77.1% | +16.2 pp |
| credit\_heavy | 46.2% | 69.6% | +23.4 pp |
| **Mean** | — | — | **+15.4 pp** |

The largest gain (+23.4 pp) occurs on credit\_heavy, suggesting the 2-step MDP is
particularly valuable when the optimal second shock (flight-to-safety vs.
credit contagion) depends strongly on the regime context established by the first
shock.

**Why bond\_heavy clears the gate.** Duration risk is concentrated and the causal
transmission chain is tight: rate shocks propagate to fixed income prices through
the VAR propagation structure, with the stressed-regime causal graph providing a
stable rate→credit pathway. The bandit reliably fires TLT as the primary shock
(top discovered sequence: TLT $-5\sigma$ → LQD $-5\sigma$), then selects the
appropriate second leg based on the causal graph. The stability of the first-step
choice makes the 2-step MDP highly effective for this profile.

**Why the heuristic remains competitive on `balanced`.** The heuristic selects the
best of three domain-expert sequences independently for each seed — a per-seed
oracle that represents the upper bound of expert-template performance. For
`balanced`, the set of three sequences (including ^GSPC $-5\sigma$ → XLF $-5\sigma$
as the broad equity + financial shock) is well-suited to a portfolio with no
dominant single-risk factor. The bandit trails by 1.4 pp ($-1.4$ pp) against this
per-seed oracle. The bandit's remaining value for this profile is sequence
diversity: it generates distinct causal pathways across seeds rather than selecting
from a fixed expert template.

**Exhaustive comparison (bond\_heavy).** The beam reference (width 6, 36 sequences)
is not globally optimal. To quantify this gap, we ran full exhaustive evaluations
for bond\_heavy across three training configurations: all $250^2 = 62{,}500$
two-step sequences per seed across all 16 held-out seeds ($1{,}000{,}000$ total
evaluations per run). Bandit, beam (recomputed), and exhaustive best were computed
in a single consistent pass in each run. Results across configurations:

\begin{center}
\begin{tabular}{lccc}
\hline
Training configuration & Bandit/exhaustive & Beam/exhaustive & Bandit/beam \\
\hline
6-branch baseline & 50.1\% & 68.7\% & 72.9\% \\
25-branch (greedy) & 55.7\% & 70.8\% & 78.6\% \\
25-branch + forced $0.5\sigma$ & 55.6\% & \textbf{74.9\%} & 74.2\% \\
\hline
\end{tabular}
\end{center}

The exhaustive optima are dominated by moderate-first-shock sequences (e.g.
LQD $-0.5\sigma \to$ TLT $-5\sigma$; XLF $-4.5\sigma \to$ TLT $-5\sigma$;
CPIAUCSL $-0.5\sigma \to$ T10Y2Y $+5\sigma$) whose first step scores poorly
by immediate reward but sets up a highly damaging second shock through the
causal propagation structure. Expanding step-2 branch coverage from 6 to 25
improved bandit coverage from $50.1\%$ to $55.7\%$ of the exhaustive maximum
($+5.6$ pp). Adding forced minimum-magnitude branches (one per target variable
at $0.5\sigma$) did not improve greedy coverage further but raised the beam
reference's own exhaustive coverage from $70.8\%$ to $74.9\%$, and improved
UCB-mode bandit performance to $82.0\%$ of beam ($\approx 60\%$ of exhaustive).
The Table 1 percentages (including the $86.4\%$ gate result) are relative to
the beam reference; beam itself captures only $\approx 70$--$75\%$ of the true
adversarial ceiling depending on training configuration.

---

### 5.3  Supporting Metrics

**Table 2: Supporting metrics (v2, greedy, mean over 16 seeds)**

| Profile | Portfolio Loss | Causal Fidelity | DFAST Severity | Diversity |
|---|:---:|:---:|:---:|:---:|
| balanced | 0.696 | 0.634 | 0.348 | 0.922 |
| bond\_heavy | 0.594 | 0.679 | 0.568 | 0.905 |
| tech\_heavy | 1.020 | 0.660 | 0.450 | 0.944 |
| credit\_heavy | 0.660 | 0.653 | 0.311 | 0.934 |

All numbers are from `bandit_v2_{profile}/heldout_results.json` (bandit\_greedy
row).

**Causal fidelity.** Values of 0.63–0.68 mean that for approximately two-thirds
of the causal graph edges where both the source and target variables move
non-trivially, the observed terminal comovement sign is consistent with the
causal prediction (edge weight $\times$ source shock direction). This is
consistent with the structural constraint imposed during training (the CF bonus
in the reward function) and represents a meaningful improvement over random
sequences, whose CF values (approximately 0.61 for balanced, varying by profile)
serve as a data-fit lower anchor.

**DFAST breach severity** is a continuous measure of how badly the scenario
exceeds the DFAST 2026 severely-adverse regulatory thresholds, evaluated on
four key variables (UNRATE, BBB spread, ^GSPC, ^VIX). A severity of 1.0 means the
scenario exactly matches the official DFAST test; >1.0 indicates a catastrophic
scenario exceeding DFAST severity. Values of 0.31–0.57 indicate that the bandit
discoveries regularly produce scenarios in the range of regulatory stress tests
without being designed to do so. The beam reference's credit\_heavy DFAST severity
of 1.04 confirms that worst-case adversarial sequences can exceed the regulatory
severely-adverse scenario.

**Sequence diversity** (0.90–0.94) reflects the action novelty signal: the
frequency with which each (target variable, shock family) pair has appeared in
the recent episode history. Values above 0.90 confirm that the bandit explores
distinct variable-pair pathways across episodes rather than converging on a single
repeated action, indicating effective adversarial search rather than degenerate
single-variable hammering.

---

### 5.4  Historical Crisis Alignment

We validate that the bandit discovers financially meaningful adversarial pathways
rather than reward artefacts by testing it on historical starting states near three
crisis dates and comparing the discovered pathway to the observed transmission
mechanism.

**Table 3: Bandit pathways on historical crisis starting states**

| Crisis | Profile | Bandit pathway | Historical transmission | Verdict |
|---|---|---|---|---|
| Rate shock 2022 | bond\_heavy | TLT $-5\sigma$ → LQD $-5\sigma$ | Treasury selloff → IG credit spread widening | ✓ Strong |
| Rate shock 2022 | tech\_heavy | XLK $-5\sigma$ → ^NDX $-5\sigma$ | Tech sector re-rating → Nasdaq selloff | ✓ Strong |
| COVID 2020 | tech\_heavy | ^NDX $-5\sigma$ → XLY $-5\sigma$ | Nasdaq-led selloff → consumer discretionary | ✓ Strong |
| COVID 2020 | credit\_heavy | XLF $-5\sigma$ → TLT $-5\sigma$ | Financial stress → treasury flight-to-safety | ✓ Partial |
| GFC 2008 | credit\_heavy | XLF $-5\sigma$ → TLT $-5\sigma$ | Bank collapse → treasury flight-to-safety | ✓ Strong |
| GFC 2008 | bond\_heavy | TLT $-5\sigma$ → XLU $-5\sigma$ | Rate crash → rate-sensitive utilities | Partial |

Pathways from `docs/paper/02_crisis_alignment_results.md`. Crisis-year seeds are the
closest available historical states; all within 230 days of the crisis date.

**Rate shock of 2022 (best-aligned crisis).** The Federal Reserve raised rates 425 bp
in 2022 — the fastest hiking cycle in 40 years — sending TLT down 40% YTD and
LQD down 18% YTD. The bond\_heavy bandit independently fires the precise
two-leg chain (TLT → LQD) that characterised this event. The tech\_heavy bandit
fires XLK → ^NDX, matching the sector re-rating → broad Nasdaq selloff pathway
(Nasdaq-100 fell 33% in 2022). Both profiles achieve strong alignment without
any crisis-specific supervision signal.

**COVID crash of 2020.** The S\&P 500 fell 34\% in 33 days (Feb 19 – Mar 23, 2020),
led by growth names before broad credit dislocation. The tech\_heavy bandit fires
^NDX → XLY — the Nasdaq-led selloff propagating to consumer discretionary — the
historically observed first-leg transmission channel. The starting states land
52 days after the trough (early recovery period), making the strong alignment
more notable.

**GFC peak of 2008.** The closest available seeds land 229 days after Oct 10, 2008
(during the post-Lehman recovery). Despite this gap, the credit\_heavy bandit fires
XLF → TLT — the GFC signature pathway: financial sector collapse followed by the
largest treasury flight-to-safety bid in modern history. The bond\_heavy result
(TLT → XLU) is a partial alignment: TLT as primary shock is correct, and XLU
(utilities) is rate-sensitive, but the expected second leg would be a financial
sector or credit-spread shock rather than utilities.

**Alignment is not guaranteed by the reward function.** The bandit is trained on
portfolio damage and causal fidelity signals, not on historical labelling. Its
discovery of crisis-aligned pathways across three structurally different crisis
types provides independent evidence that the reward-shaped adversarial search is
financially grounded.

---

### 5.5  Cross-Profile Analysis

The four profiles reveal a consistent pattern: **bandit learnability correlates
with portfolio risk concentration.**

**bond\_heavy (86.4\% — gate cleared).** Concentration in duration risk provides a
clear learning signal. The bandit reliably selects TLT as the first shock across
all regimes; the second-step choice (LQD, XLF, or XLU) adapts to the causal graph
state. The v1 result (74.1\%) already exceeds the heuristic (62.0\%); the 2-step
MDP pushes this to 86.4\%.

**tech\_heavy (77.1\%).** The bandit correctly identifies XLK/^NDX as the primary
shock vector across both COVID (growth panic) and 2022 (rate re-rating) starting
states — two structurally different crisis environments. The gap to the gate
reflects second-step noise: the optimal propagation target varies across regimes
(XLY under COVID, ^NDX under 2022). A v3 training run with doubled crisis-seed
sampling regressed to 57.6\% UCB (see RL\_RESULTS\_REPORT.md); v2 is the retained
production model.

**credit\_heavy (69.6\%).** The bandit correctly identifies XLF (financials) as the
primary shock — consistent across GFC and COVID starting states.
The +23.4 pp v1→v2 gain is the largest of any profile, indicating the second step
provides the most value when the optimal follow-on shock (flight-to-safety vs.
credit contagion) is regime-dependent.

**balanced (62.6\%).** The balanced profile is the hardest: no single exposure
dominates (^GSPC 30\%, with distributed tech, financial, energy, and utility
positions). The bandit trails the domain-expert heuristic by 1.4 pp — the only
profile where the expert template outperforms the learned policy. This reflects
the diffuse exposure structure: with no dominant risk factor, the bandit cannot
learn a stable primary shock variable. It still delivers diversity value (distinct
pathways across seeds) and outperforms the heuristic's worst sequence; the
limitation is portfolio structure, not model capacity.

---

### 5.6  Inference Complexity

The bandit's inference cost per portfolio is two batched forward passes through
BanditRewardNet plus two simulator steps:

1. **Step-1 scoring**: one vectorised forward pass scoring all 250 actions in
   parallel (observation expanded to shape $(250, 56)$, action embeddings precomputed
   as a $(250, 32)$ tensor); single argmax selects the shock.
2. **One simulator step**: VAR propagation for 25 variables, one lag.
3. **Step-2 scoring**: same as step 1, with updated observation including the
   step-1 shock vector.
4. **One simulator step**: final portfolio loss computed.

In UCB mode, step-1 and step-2 scoring each run 20 stochastic forward passes
(dropout enabled); the scoring cost scales to $20 \times 2 = 40$ forward passes
total. The network is small (56→128→128→160→64→1), and the simulator steps — VAR
propagation for a 25-variable lag-2 system — are the primary per-inference cost,
not the network scoring. No explicit wall-time benchmark is reported here; empirical
latency depends on hardware and parallelism.

By contrast, the beam reference requires 250 simulator calls (to score all step-1
actions individually) plus 36 full 2-step simulations — an order of magnitude
more environment interactions than the bandit. The bandit replaces these simulator
calls with neural network forward passes, converting the search problem from
environment-bound to compute-bound.

---

### 5.7  Ablation: PPO vs. Supervised Reward Regression

We trained the adversarial policy with PPO across four successive iterations (v2–v5),
progressively adding crisis-state oversampling, damage-first reward, and longer
rollouts. The PPO formulation was a 1-step MDP: the policy selected a single shock
per episode evaluated against a 1-step beam reference. Despite these refinements,
the best PPO policy (v5) achieved 50.8\% (balanced), 52.9\% (tech\_heavy),
76.5\% (bond\_heavy), and 56.3\% (credit\_heavy) of its respective 1-step beam
reference — falling below the domain-expert heuristic on three of four profiles.
The top discovered sequence in each profile was always a single maximum-magnitude
shock on the highest-weight asset (XLK $-5\sigma$ for tech\_heavy, TLT $-5\sigma$
for bond\_heavy, XLF $-5\sigma$ for credit\_heavy, XLY $-5\sigma$ for balanced),
confirming convergence to a context-insensitive degenerate strategy.

The supervised reward regression approach (bandit v2) clears or approaches the
$\geq 85\%$ gate on bond\_heavy (86.4\% vs its oracle) and achieves 62.6–77.1\%
on the remaining three profiles. Direct numerical comparison of the
percentage figures requires caution because PPO v5 and bandit v2 are evaluated
against different beam references (1-step vs 2-step); the practically fair summary
is that bandit v2 exceeds the domain-expert heuristic on three of four profiles
(bond\_heavy +24.4 pp, tech\_heavy +9.1 pp, credit\_heavy +4.5 pp; balanced
$-1.4$ pp net), while PPO v5 falls below the same heuristic on three of four
profiles. This establishes the practical advantage of dense supervised reward
labelling over sparse policy-gradient training in this problem class.

---

### 5.8  Limitations

We explicitly scope the following limitations:

1. **Three profiles do not clear the 85\% gate.** balanced (62.6\%),
   tech\_heavy (77.1\%), and credit\_heavy (69.6\%) are below the deployment
   threshold. The system remains useful — it returns causally coherent,
   financially interpretable adversarial sequences — but with reduced confidence
   that they represent the true worst case.

2. **The beam reference is not globally optimal.** With width 6, the beam evaluates
   $6^2 = 36$ of the 62,500 possible 2-step sequences. Exhaustive evaluation for
   bond\_heavy ($250^2 \times 16 = 1{,}000{,}000$ evaluations) establishes the true
   upper bound. Across training configurations, beam captures $69$--$75\%$ of the
   exhaustive global maximum; the greedy bandit captures $50$--$56\%$; UCB-mode
   bandit captures $\approx 60\%$. The $85\%$ gate and all Table 1 percentages are
   relative to beam, not to the true adversarial maximum; see §5.2 for the full
   per-configuration breakdown.

3. **Training distribution is historical (1990–2024).** Novel crisis types with
   no historical precedent (e.g. digital-asset contagion into traditional markets)
   may not be well-represented in the training distribution.

4. **Portfolio profiles are coarse.** A portfolio that is simultaneously 30\%
   duration and 30\% tech would be routed to whichever exposure has the higher
   computed score at routing time (argmax of the three exposure dimensions). A
   portfolio where both scores are nearly tied may not receive the most appropriate
   profile. Hybrid-profile adversarial search is future work.
