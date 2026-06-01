# Sections 1 & 2 Draft — Adversarial RL Paper
**Status:** Draft (v2 — corrected) — Section 1 quantitative claims aligned with verified drafts; Section 2 citations are draft-level and not source-verified  
**Grounded in:**
- `docs/paper/04_sections_3_and_4_draft.md` (v4) — method, verified against source code
- `docs/paper/05_section_5_draft.md` (v3) — results, verified against JSON artifacts
- `docs/paper/README.md` — one-sentence paper claim
- `ml_pipeline/best_model_definition.md` — canonical scenario generator results (90% coverage)
- `ml_pipeline/EBA_PAPER_FRAMING.md` — regulatory comparison scope
- `ml_pipeline/RL_RESULTS_REPORT.md` — system description, portfolio profiles  
**Last updated:** 2026-05-25 (v2 — 12 forensic patches applied)

---

## 1  Introduction

Financial stress testing is a cornerstone of risk management practice and
regulatory compliance. Standard practice asks a conditional question: *given a
pre-specified scenario, what losses does this portfolio incur?* Under this
paradigm, the quality of stress testing is bounded by the quality of the
scenario library. Scenarios are authored by domain experts or mandated by
regulators (e.g., the Federal Reserve's DFAST Severely Adverse trajectory),
and a portfolio is exposed only to the scenarios that humans have thought to
design.

This leaves a structurally important question unanswered: *what is the
worst-case shock sequence for this specific portfolio?* A stress tester who
cannot answer this question cannot know whether the regulatory scenario is
genuinely severe for the portfolio at hand, or whether a distinct pathway —
one that exploits the portfolio's particular concentration of duration,
technology, or credit risk — would cause substantially greater damage.

We address this gap with an adversarial stress testing system that inverts the
standard question. Given a portfolio and a causal market model, the system
searches for the shock sequence that maximises mark-to-market portfolio loss,
subject to the constraint that the sequence respects empirically estimated
causal transmission mechanisms. This is not a human-authored scenario but a
machine-discovered worst case, grounded in the causal structure of historical
stressed markets.

**Technical approach.** We frame adversarial search as a two-step Markov
Decision Process over a discrete action space of 250 (variable, magnitude)
pairs. A neural contextual bandit — BanditRewardNet — is trained by supervised
reward regression on exhaustively labelled (observation, action, reward) triplets:
at step 1, all 250 actions are evaluated for each training seed; at step 2, the
top-$K$ step-1 actions are expanded into branches and all 250 actions are
evaluated from each resulting state. Given an observation of the current market
state (regime, z-scored variables, prior shock vector, step progress), the trained
model predicts the portfolio loss that would result from each candidate action,
then selects the action with the highest predicted value (greedy) or the highest
upper-confidence-bound (UCB). The reward combines portfolio loss, causal
fidelity (fraction of propagation edges sign-consistent with the causal graph),
and sequence diversity. Hard action constraints eliminate causally implausible
targets and adverse-direction violations before scoring. Separate policies are
trained for four portfolio vulnerability profiles: balanced, bond\_heavy,
tech\_heavy, and credit\_heavy.

**Results.** We evaluate across 16 held-out starting states and four portfolio
profiles. The 2-step bandit (v2) improves over the 1-step baseline by
+9.6 to +23.4 percentage points across all profiles, averaging +15.4 pp.
The bond\_heavy profile clears the $\geq 85\%$ deployment gate, reaching
86.4\% of beam-search-reference quality.
Validated against three structurally distinct historical crises — the 2022
rate shock, the COVID-19 crash, and the 2008 Global Financial Crisis — the
bandit shows strong-to-partial alignment with historically observed transmission
mechanisms across multiple profile–crisis pairings, providing independent
evidence of financial grounding beyond the reward signal.

The causal scenario generator that underlies the simulator achieves 90\%
coverage on held-out stress events, significantly outperforming unconditional
VAR and historical-replay baselines, with Student-t innovations calibrated to
stressed-regime residuals.

**Contributions.** We contribute:

1. A two-step adversarial stress testing MDP with a causal graph constraint and
   a portfolio-specific reward signal that addresses the degenerate policy collapse
   observed under PPO policy-gradient training.

2. BanditRewardNet: a neural contextual bandit trained by dense supervised
   reward regression, which outperforms the domain-expert heuristic on three of
   four profiles and represents a stronger adversarial search formulation than
   PPO policy-gradient, which falls below the same heuristic on three of four profiles.

3. An empirical demonstration that the bandit discovers financially meaningful
   adversarial pathways — aligned with GFC, COVID, and 2022 crisis transmission
   mechanisms — without any crisis-specific supervision signal.

4. Transparent quality disclosure: the bandit reports its result as a
   percentage of the beam-search reference, so practitioners know whether the
   discovered sequence carries production-grade confidence or exploratory
   confidence.

---

## 2  Related Work

### 2.1  Regulatory Stress Testing and Scenario Design

Regulatory stress testing frameworks — DFAST (Dodd-Frank Act Stress Test) in
the United States and equivalent frameworks in Europe (EBA) and the United
Kingdom (BoE) — require financial institutions to evaluate portfolio resilience
under prescribed severely-adverse macroeconomic paths
\cite{frb_dfast_2026,eba_methodology_2023}. These scenarios are centrally
designed and applied uniformly across institutions, which limits their
effectiveness for identifying institution-specific worst cases: a scenario
severe for a balanced multi-asset book may be only moderately severe for a
duration-concentrated or credit-concentrated book.

Internal stress testing literature has explored scenario amplification
\cite{rebonato_2010}, reverse stress testing (finding the scenario that causes
ruin \cite{fsr_2008_reverse}), and historical scenario replay
\cite{mcneil_frey_embrechts_2015}. These approaches either rely on expert
judgment to define search directions or are confined to historically observed
scenarios, both of which limit coverage of tail risks that have not yet
materialised.

### 2.2  Causal Discovery in Finance

A growing body of work applies causal discovery methods — including
Granger-causality testing \cite{granger_1969}, DYNOTEARS \cite{pamfil_2020},
and PCMCI \cite{runge_2019} — to financial time series to recover the directed
graph of propagation relationships among asset prices, credit spreads, and
macroeconomic variables. \citet{billio_2012} use pairwise Granger causality to
measure systemic connectedness among financial institutions. \citet{hurd_2016}
model contagion through causal graphs of interbank exposures.

We adopt the DYNOTEARS/PCMCI approach for causal graph estimation on
stressed-regime time series (1990–2024), obtaining a 330-edge stressed-regime
graph that constrains which propagation directions are structurally supported.
Unlike pairwise approaches, DYNOTEARS and PCMCI recover time-lagged
directed acyclic graph (DAG) structure while controlling for multiple
comparisons, making the estimated edges actionable as hard constraints rather
than correlational suggestions.

### 2.3  Generative Scenario Models

Generative approaches to stress scenario construction use machine learning to
produce scenarios that are statistically consistent with historical market
dynamics while being capable of extrapolating to tail regions. Methods include
copula-based joint tail models \cite{embrechts_2002}, regime-switching
models \cite{ang_bekaert_2002}, variational autoencoders for market data
\cite{buehler_2020}, and GANs trained on financial time series
\cite{wiese_2020,deep_hedging_2019}.

Our causal scenario generator belongs to the regime-conditioned VAR family:
a lag-2 Vector Autoregression estimated on stressed-regime observations, with
Student-t innovations calibrated empirically from VAR residuals (df $\approx 3.84$
for crisis regimes). This is a deliberate choice of interpretability over
expressiveness: the VAR propagation dynamics and causal graph edges are
inspectable and directly linked to the reward signal. Generative neural
approaches (VAE, GAN) would require interpretable attribution of "why this
scenario caused portfolio loss" — a requirement not met by decoder-space generation.

The closest related work in generative stress testing is \citet{golub_2021},
who use constrained Monte Carlo with regime conditioning for VaR estimation, and
\citet{mcneil_2019}, who augment historical replay with copula-fitted tail
scenarios. Neither work poses the adversarial search problem or uses a policy to
search for portfolio-specific worst cases.

### 2.4  Adversarial Machine Learning in Finance

Adversarial attacks in machine learning — perturbations designed to maximise a
loss function — have been extensively studied in image classification
\cite{goodfellow_2014_fgsm,madry_2017_pgd} and natural language processing
\cite{jia_2017,wallace_2019}. Financial applications of adversarial methods are
more limited: \citet{bao_2019} apply adversarial training to time-series
forecasting; \citet{zhang_2020} use GAN-based adversarial scenarios for VaR
backtesting. These works treat adversarial perturbations as input transformations,
whereas our setting requires adversarial search over a semantically meaningful
action space (which variable to shock, by how much) subject to causal plausibility
constraints.

Closest to our work is \citet{cont_2020}, who formulate worst-case regulatory
scenario search as a convex optimisation problem over stress scenario parameters.
The key distinction is their approach requires a differentiable portfolio valuation
function and optimises a continuous relaxation, whereas our method (a) operates
directly on the discrete (variable, magnitude) action space, (b) incorporates the
causal graph as a hard constraint on the action mask, and (c) requires only
portfolio mark-to-market, not a differentiable valuation function.

### 2.5  Reinforcement Learning and Bandits in Finance

Reinforcement learning has been applied across a range of financial tasks:
portfolio optimisation \cite{moody_1998,jiang_2017}, execution optimisation
\cite{almgren_2001,nevmyvaka_2006}, and market making \cite{spooner_2018}.
Bandit methods have been used for hyper-parameter optimisation in
high-frequency trading \cite{chan_2019} and for exploration-exploitation
tradeoffs in limit-order-book strategies \cite{ning_2021}.

Our use of a neural contextual bandit for adversarial search is distinct from
these applications in several respects. First, the reward is a supervised
regression target derived from environment evaluations over exhaustively labelled
action-state pairs, not a live trading signal — eliminating the non-stationarity
and delayed feedback that plague online bandit algorithms in financial environments.
Second, training is entirely offline; at deployment time the trained model scores
all 250 actions in a single vectorised forward pass (greedy) or 20 stochastic
passes (UCB), with no online exploration.

The use of supervised reward regression to convert an RL problem into a bandit
problem was demonstrated in the context of language model alignment by
\citet{christiano_2017} (reward modelling from human preferences) and
\citet{ziegler_2019} (RLHF). Our setting differs in that the reward labels come from exhaustive environment
evaluations (not human feedback), and the policy is a non-sequential contextual
bandit rather than a language model.
PPO policy-gradient training \cite{schulman_2017} was evaluated across four
successive iterations (v2–v5) as a 1-step MDP and found to converge to a
context-insensitive degenerate strategy, falling below the domain-expert
heuristic on three of four profiles and motivating the switch to supervised
reward regression.

### 2.6  Bandit Formulations for Structured Action Spaces

Contextual bandits with structured action spaces have been studied in
recommendation systems \cite{li_2010_linucb}, drug dosing \cite{bastani_2020},
and safety-constrained sequential decision-making \cite{amani_2019}. In these
settings, the challenge is to select from a large but semantically organised
action set given a context vector, without the full Markov structure of an MDP.

Our action space (250 (variable, magnitude) pairs per step) is modest in
size but highly structured: the hard action mask eliminates causally implausible
actions before scoring, the observation encodes the prior shock history (allowing
step-2 selection to condition on step-1 outcome), and the UCB exploration bonus
incentivises coverage of distinct variable-family pairs. This structure makes
the contextual bandit a natural fit: sufficient expressivity to learn
portfolio-specific patterns, sufficient interpretability to report the causal
pathway to a risk practitioner.

---

*References to be completed in camera-ready version. Placeholder citations
above use author-year style; final paper will use numbered ICAIF format.*
