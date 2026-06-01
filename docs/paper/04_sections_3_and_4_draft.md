# Section 3 & 4 Draft — Adversarial RL Paper (v5 — corrected)
**Status:** Draft — verified against source code, run artifacts, and checkpoint metadata  
**Verified against:**
- `causal_stress_env.py` — observation space, hard action constraints, progress term
- `rewards.py` — reward formula, CF definition, diversity definition
- `action_wrapper.py` — action space dimensions
- `neural_bandit.py` — architecture, dropout=0.15
- `train_bandit.py` / `train_v7_bandit_2step.py` — dataset construction
- `real_mode_loader.py` — VAR fitting, regime handling, graph loading
- `adversarial_serve.py` — profile routing (argmax logic)
- `portfolio_model.py` — portfolio compositions
- `action_space/action_space.yaml` — 25 variable list
- `best_model_definition.md` — canonical model: VAR (not VECM), Student-t data-fit df
- `RL_RESULTS_REPORT.md` — DYNOTEARS/PCMCI graph source, 330 edges
- Live bandit checkpoint metadata: obs_dim=56, n_targets=25, n_families=1, n_mags=10, dropout=0.15  
**Last updated:** 2026-05-25 (v5 — Point 1: MDP/bandit reconciliation in §3.3 + §4.4; Point 2: formal VAR+graph equations in §3.2)

---

## 3  Problem Formulation

### 3.1  Portfolio and Market State

Let a portfolio $\mathcal{P}$ be a set of $n$ holdings, each a (asset, weight) pair
$\{(a_i, w_i)\}_{i=1}^{n}$ with $\sum_i w_i = 1$.
The market state at time $t$ is a vector $\mathbf{x}_t \in \mathbb{R}^{25}$ of
z-scored values for 25 core financial variables spanning equity indices, interest
rates, credit spreads, commodity prices, and macroeconomic indicators (see Table A.1).

Each historical observation carries an HMM-inferred regime label drawn from the
$K = 5$ regime categories present in the historical state support.
The policy observation encodes
this label as a $K$-dimensional one-hot vector. The VAR is estimated exclusively on
the four stressed-side regimes (elevated, stressed, high\_stress, crisis); the
stressed-regime causal graph is used throughout adversarial search regardless of
the episode's historical starting regime.

### 3.2  Causal Market Simulator

We model market dynamics under a shock using a two-layer simulator:

**Layer 1 — Regime-conditioned VAR.**
A lag-2 Vector Autoregression (VAR) is estimated on 3,374 observations drawn
exclusively from elevated, stressed, high\_stress, and crisis regimes (1990–2024).
This provides the baseline multivariate propagation dynamics for the 25 core
variables. The VAR is fitted once and shared across all environment instances.

**Layer 2 — Stressed-regime causal graph.**
A directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with
$|\mathcal{V}| = 25$ nodes and $|\mathcal{E}| = 330$ edges constrains which
propagation directions are structurally supported. The graph is estimated by
DYNOTEARS/PCMCI causal discovery on stressed-period time series and loaded from
the canonical regime-conditional causal graph artifact. The stressed-regime variant is used
throughout adversarial search (worst-case assumption: we do not assume a more
permissive causal structure).

**Student-t innovations.** The canonical scenario generator uses Student-t
distributed innovations with degrees of freedom calibrated empirically from VAR
residuals: df $\approx 3.84$ for crisis regimes and df $\approx 5.97$ for
near-normal regimes. This data-fit parameterisation captures fat-tailed behaviour
during extreme periods without imposing an arbitrary distributional assumption.

**Simulator mechanics.** Let $\mathbf{B} \in \mathbb{R}^{(1+pd)\times d}$ be the
ridge-OLS coefficient matrix of the lag-$p$ VAR estimated on stressed-regime data
($d = 25$, $p = 2$). The two layers interact as follows.

**Step 0 — Shock initialisation via causal graph.** Given a root shock of
magnitude $\sigma_v$ on variable $v$, the initial shock vector
$\boldsymbol{\delta} \in \mathbb{R}^{25}$ is computed by breadth-first propagation
through $\mathcal{G}$ up to 3 hops. At hop $k$, for each edge
$(s, u, w_{su}) \in \mathcal{E}$ where $s$ was visited at hop $k-1$ and $u$ has
not yet been visited:
$$
\boldsymbol{\delta}_u = \boldsymbol{\delta}_s \cdot w_{su} \cdot \gamma^k,
\qquad \gamma = 0.4,
$$
with $\boldsymbol{\delta}_u$ set to zero if $|\boldsymbol{\delta}_u| < 0.12$.
The causal graph $\mathcal{G}$ appears only here — it constrains the initial
condition, not the VAR transition dynamics.

**Steps 1–$H$ — Reduced-form VAR rollout.** From the shocked initial state
$\tilde{\mathbf{x}}_0 = \boldsymbol{\delta}$ (in z-score units), each subsequent
day is:
$$
\tilde{\mathbf{x}}_t
= \bigl[\mathbf{1},\;\tilde{\mathbf{x}}_{t-1}^\top,\;\tilde{\mathbf{x}}_{t-2}^\top\bigr]
  \mathbf{B}
\;+\;\boldsymbol{\varepsilon}_t
\;+\;\boldsymbol{\delta}\cdot\rho^{t-1}\cdot\mathbf{1}[t \leq \tau],
$$
where $\boldsymbol{\varepsilon}_t = \kappa\cdot\mathbf{L}\,\mathbf{z}_t$,
$\mathbf{z}_t \stackrel{\text{i.i.d.}}{\sim} \mathrm{Student\text{-}t}(\nu)$,
$\mathbf{L} = \mathrm{chol}(\boldsymbol{\Sigma})$; the shock-persistence parameters
are $\rho = 0.72$ and $\tau = 5$ days; and the scale $\kappa \in \{1.0, 1.1, 1.2\}$
increases with shock severity. The trajectory is denormalised as
$\mathbf{x}_t = \tilde{\mathbf{x}}_t \odot \boldsymbol{\sigma} + \boldsymbol{\mu}$
before portfolio loss evaluation.

**Distinction from SVAR.** This is a reduced-form VAR with causal-graph-guided
shock initialisation, not a Structural VAR (SVAR). In an SVAR, $\mathcal{G}$ would
appear in $\mathbf{B}$ via a structural identification restriction (e.g., Cholesky
or long-run constraints on the contemporaneous impact matrix). Here, $\mathbf{B}$
is estimated by unconstrained ridge OLS; $\mathcal{G}$ affects only
$\boldsymbol{\delta}$ at $t=0$ and the causal fidelity reward signal.

### 3.3  Adversarial Search Problem

**Standard stress testing** asks: *given scenario* $S$, *compute portfolio loss.*
We invert this: *find the sequence of shocks that maximises portfolio loss for this
specific portfolio*, subject to structural plausibility constraints.

Let $\mathcal{A}$ be the feasible action space (target variable, shock magnitude)
pairs, and let $T$ denote the search horizon (we fix $T = 2$). The adversarial
search problem is:

$$
\mathbf{a}^* = \arg\max_{\mathbf{a}_{1:T} \in \mathcal{A}^T}
\; \ell(\mathbf{x}_T, \mathcal{P}),
$$

where $\ell(\mathbf{x}_T, \mathcal{P})$ is the mark-to-market portfolio loss after
$T$ simulation steps, and $\mathcal{A}$ is restricted by two hard structural
constraints:

1. **Causal plausibility mask**: an action targeting variable $v$ is only valid
   if $v$ has at least one outgoing edge in the stressed-regime causal graph
   $\mathcal{G}$; otherwise the action is rejected and receives a penalty reward.
2. **Adverse direction**: shocks to price-return variables must be negative
   (downward); shocks to rate/spread variables must be positive (upward).

A soft causal coherence objective is incorporated into the reward function
(Section 4.5), encouraging sequences whose VAR propagation follows the causal
graph's predicted sign structure. This soft term rewards but does not enforce
causal coherence at every propagation edge.

**MDP formulation and solution method.** The above is formally a finite-horizon
Markov Decision Process (MDP) with $T = 2$ steps, state space $\mathbb{R}^{56}$
(the policy observation vector $\mathbf{o}_t$), action space $\mathcal{A}$ (250
elements), and transition dynamics given by the causal VAR simulator. We solve it
via per-step supervised reward regression rather than policy-gradient RL: a reward
model $\hat{R}(\mathbf{o}_t, \mathbf{a})$ is trained for each step to predict the
terminal episode reward from the current observation-action pair. At deployment,
the policy selects $\mathbf{a}_t^* = \arg\max_{\mathbf{a} \in \mathcal{A}}
\hat{R}(\mathbf{o}_t, \mathbf{a})$ — a greedy decomposition of the MDP through
its per-step reward approximation.

We use the term *contextual bandit* throughout to describe this per-step argmax
mechanism. This terminology is informal: the classical contextual bandit setting
assumes i.i.d. (context, action) pairs with no state transition, whereas here
$\mathbf{o}_2$ includes the post-step-1 market state and prior shock vector,
making the problem inherently sequential. The term is retained because the
solution mechanism — offline reward regression followed by single-step greedy
selection — does not require online exploration or value iteration. Readers should
treat "contextual bandit" as shorthand for "per-step reward model with argmax
deployment in a finite-horizon MDP."

**Causal fidelity** $\mathrm{CF} \in [0, 1]$ measures the degree to which the
VAR-simulated trajectory respects the causal graph's predicted comovement
directions. Formally, for each edge $(u, v, w) \in \mathcal{E}$, the predicted
sign of $v$'s terminal move is $\mathrm{sgn}(w \cdot \Delta u)$, where $\Delta u$
is the cumulative move of source variable $u$ over the trajectory. CF is the
fraction of edges where the observed sign of $\Delta v$ matches the predicted sign,
among edges where both variables move non-trivially:
$$
\mathrm{CF} = \frac{\#\{(u, v, w) \in \mathcal{E} : \mathrm{sgn}(\Delta v)
              = \mathrm{sgn}(w \cdot \Delta u),\; |\Delta u| > \epsilon,
              \; |\Delta v| > \epsilon \}}
{\#\{(u, v, w) \in \mathcal{E} : |\Delta u| > \epsilon,\; |\Delta v| > \epsilon \}}.
$$

**Combinatorial scale.** With $|\mathcal{V}| = 25$ variables and 10 discrete
adverse magnitude levels (0.5$\sigma$ to 5.0$\sigma$), the per-step feasible
action space has 250 elements. For $T = 2$, the full search space contains
$250^2 = 62{,}500$ candidate sequences. Exhaustive evaluation at inference time is
intractable; our beam-search reference (width 6) covers $6^2 = 36$ complete
sequences per seed. We learn a policy that approximates this argmax via a single batched
network forward pass over the 250-element action space.

---

## 4  Method

### 4.1  Causal Market Simulator (Implementation)

The simulator is implemented as an OpenAI Gym-compatible episode environment
(`causal_stress_env.py`). Each episode starts from a sampled historical market
state; the episode terminates after $T = 2$ shocks are applied.

The policy observation vector $\mathbf{o}_t \in \mathbb{R}^{56}$ is structured as:

$$
\mathbf{o}_t = \bigl[\underbrace{\mathbf{e}_r}_{K} \;\|\;
\underbrace{\mathbf{z}_t}_{25} \;\|\;
\underbrace{\boldsymbol{\delta}_{t-1}}_{25} \;\|\;
\underbrace{p_t}_{1} \bigr],
\quad K + 25 + 25 + 1 = 56,
$$

where $K = 5$ is the number of HMM regime categories present in the data;
$\mathbf{e}_r \in \{0,1\}^K$ is the one-hot encoding of the episode's starting
regime (stored from the sampled historical observation, not re-inferred during
bandit rollout); $\mathbf{z}_t$ is the vector of z-scored market variable values
at the current simulation step; $\boldsymbol{\delta}_{t-1} \in \mathbb{R}^{25}$
is the shock magnitude vector applied at the previous step (zeros at $t = 1$);
and $p_t = (t-1)/T \in \{0, 0.5\}$ is the episode progress fraction
($p_1 = 0$ at step 1, $p_2 = 0.5$ at step 2 in a $T = 2$ episode).

The inclusion of $\boldsymbol{\delta}_{t-1}$ is critical: it allows the step-2
policy to condition on what was already shocked, enabling causally sequenced
multi-step pathways rather than independent single-variable decisions.

The stressed-regime causal graph is fixed throughout the episode. The VAR is
fitted with lag $p = 2$ on 3,374 observations from stressed training regimes
and cached once at process start.

### 4.2  Action Space and Constraints

The action space is MultiDiscrete$([25, 1, 10])$ with 250 elements per step,
implemented in `action_wrapper.py`:

- **Target variable** ($|\mathcal{V}| = 25$): the variable to shock.
- **Shock family** (1 active in production): in `use_family_templates=False` mode,
  the family dimension is collapsed to a single identity entry.
- **Magnitude** (10 adverse levels): $\{0.5, 1.0, \ldots, 5.0\}\,\sigma$ for
  price variables (applied as negative shocks) or $\{+0.5, \ldots, +5.0\}\,\sigma$
  for rate/spread variables, drawn from a linspace of 21 values $[-5, 5]$ with
  the zero entry excluded and the adverse half selected.

**Hard constraints** are enforced at the environment level before any reward
computation:

1. **Causal plausibility mask**: targets rejected by the stressed-regime causal
   plausibility mask receive `reward = -1.0` and a no-op trajectory.
   The policy learns to avoid invalid targets through this penalty signal.
2. **Adverse direction**: the environment rejects a shock if its sign is not
   adverse for the selected variable (e.g., a positive magnitude on an equity
   variable is rejected).

**Soft causal coherence**: in addition to these hard constraints, the reward
function includes a causal fidelity bonus (Section 4.5) that rewards trajectories
where VAR-propagated comovements follow the causal graph's sign predictions. This
soft term shapes the policy toward causally coherent sequences beyond the binary
plausibility mask.

### 4.3  Portfolio Fingerprinting and Zero-Shot Profile Routing

Because training a separate model per arbitrary portfolio is infeasible at
deployment time, we define four canonical *vulnerability profiles* and route each
portfolio to its nearest profile at inference time.

**Exposure fingerprinting.** Given holdings $\{(a_i, w_i)\}$, each holding is
classified into one of four exposure types — technology, credit, duration
(fixed income), or equity — by matching on ticker symbols, asset category labels,
and asset name keywords (in that priority order). Credit is matched before
duration to correctly route positions such as high-yield bonds (which carry a
`fixed-income` category label but are credit exposures). The four aggregate
exposures are weighted sums over matching holdings.

**Profile assignment** takes the argmax over the three non-equity exposures:

$$
\text{profile} =
\begin{cases}
\text{balanced} & \text{if } \max(\text{tech}, \text{duration}, \text{credit}) < 25\% \\
\arg\max_{k \in \{\text{tech}, \text{duration}, \text{credit}\}} e_k & \text{otherwise}
\end{cases}
$$

This argmax routing means a portfolio with 30% duration and 26% tech routes to
`bond_heavy` (duration wins), not `tech_heavy`. All three non-equity exposures
are scored simultaneously and the dominant one wins.

**Canonical portfolio compositions** (used for training; verified from `portfolio_model.py`):

| Profile | Primary holding (weight) | Secondary holdings |
|---|---|---|
| bond\_heavy | TLT 40% | LQD 27%, HYG 12%, XLU 9%, GC=F 6%, \^GSPC 6% |
| tech\_heavy | XLK 35% | \^NDX 25%, \^GSPC 15%, XLY 10%, XLF 5%, EEM 5%, TLT 3%, GC=F 2% |
| credit\_heavy | HYG 36% | LQD 28%, XLF 16%, \^GSPC 10%, TLT 6%, GC=F 4% |
| balanced | \^GSPC 30% | XLF 15%, XLK 15%, XLE 10%, XLV 10%, XLY 10%, XLU 5%, TLT 5% |

Zero-shot routing means a practitioner with an arbitrary portfolio receives
adversarial scenarios from the nearest canonical model without retraining.

### 4.4  BanditRewardNet Architecture

We learn a reward model $\hat{R}(\mathbf{o}, \mathbf{a}; \theta)$ that estimates
the expected portfolio loss for taking action $\mathbf{a}$ from observation
$\mathbf{o}$. The architecture is a two-branch fusion network (`neural_bandit.py`):

**Observation encoder** $f_\phi$: projects the 56-dimensional observation to a
128-dimensional context embedding through two linear layers with LayerNorm, GELU
activations, and dropout ($p = 0.15$):
$$
f_\phi(\mathbf{o}) = \mathrm{LN}(\mathrm{GELU}(\mathrm{Linear}_{128}(
    \mathrm{Dropout}_{0.15}(\mathrm{GELU}(\mathrm{LN}(\mathrm{Linear}_{128}(\mathbf{o}))))
)))
$$

**Action embedder** $g_\psi$: encodes the discrete action tuple
$(\text{target}, \text{family}, \text{magnitude})$ as three learned embeddings
of dimensions 16, 8, and 8 respectively, concatenated to a 32-dimensional
action representation.

**Fusion head**: concatenates the 128-dim context and 32-dim action to a 160-dim
vector, then projects through Linear$(160 \to 64)$ → GELU → Dropout$(0.15)$
→ Linear$(64 \to 1)$ to produce the scalar reward estimate $\hat{R}$.

During inference, the action that maximises $\hat{R}(\mathbf{o}, \cdot)$ is selected
greedily by scoring all 250 catalog actions in a single batched forward pass
(observation expanded to $(250, 56)$). During exploration (UCB mode), 20
stochastic forward passes are run with Dropout active; the selected action
maximises $\mu + \beta \cdot \sigma$ with $\beta = 0.5$. This is the per-step
argmax over the reward model described in Section 3.3; because $\mathbf{o}_2$
includes the post-step-1 state, the step-2 model implicitly conditions on the
MDP transition without requiring explicit dynamic programming.

### 4.5  Reward Function

The training reward for a completed $T$-step episode is:

$$
R = \ell \cdot \bigl(1 + \lambda_{\mathrm{CF}} \cdot \mathrm{CF}
    + \lambda_{\mathrm{DV}} \cdot \mathrm{DV}\bigr),
$$

where $\ell$ is the mark-to-market portfolio loss (maximum drawdown over the
60-day simulation horizon), $\mathrm{CF} \in [0,1]$ is causal fidelity
(Section 3.3), $\mathrm{DV} \in [0,1]$ is a diversity term, and
$\lambda_{\mathrm{CF}} = 0.3$, $\lambda_{\mathrm{DV}} = 0.1$.

The multiplicative structure enforces a *damage-first* property: if the portfolio
loss is zero, the causal fidelity and diversity bonuses are also zero, preventing
the policy from generating causally elegant but financially harmless scenarios.

**Diversity.** The primary diversity signal is *action novelty*: $\mathrm{DV} = 1
- f_k$, where $f_k$ is the fraction of the 100 most recent actions in which the
same (target variable, shock family) pair was chosen. This encourages the policy
to explore distinct variables across episodes rather than converging to a single
repeated sequence. When no action history exists, $\mathrm{DV} = 1.0$ (fully
novel). As a fallback when action history is unavailable, diversity falls back to
trajectory-vs-historical-bank distance: $1 - \exp(-d_{\min})$ where $d_{\min}$
is the mean absolute distance to the nearest reference crisis trajectory.

**DFAST breach severity** is computed for logging and interpretability (measuring
how severely the scenario exceeds DFAST 2026 regulatory thresholds on UNRATE,
BBB spread, \^GSPC, and \^VIX) but is excluded from the reward signal in
`portfolio_adversarial` mode.

### 4.6  Training Data Construction

Training uses supervised reward regression rather than RL policy gradient.
For each profile, a dataset of $(\mathbf{o}, \mathbf{a}, R)$ triplets is
constructed by exhaustively evaluating all 250 actions at each step for a set of
training seeds, using the environment to compute the reward.

**Dataset size and structure** (from `train_v7_bandit_2step.py`):

| Split | Seeds | Actions per seed | Triplets |
|---|---|---|---|
| Step-1 | 50 | 250 | 12,500 |
| Step-2 | 50 × 6 branches | 250 | 75,000 |
| **Total** | — | — | **87,500** |

At step 2, 6 branches are created per seed by taking the top-6 step-1 actions
and applying each; the step-2 reward is then evaluated for all 250 actions from
each of the 6 resulting states. This ensures the step-2 model learns reward
landscapes that arise from plausible step-1 choices, not arbitrary intermediate
states.

**Crisis oversampling.** Seeds are drawn preferentially from historical crisis
states to ensure tail-risk environments are represented. The sampling probability
for crisis seeds is $p_{\mathrm{crisis}} = 0.3$ for bond\_heavy and 0.5 for
credit\_heavy, reflecting the higher importance of credit-stress dynamics for
those profiles.

**Training labels are environment rewards**, not beam demonstrations.
The beam reference is used only to compute the evaluation metric $\%_{\mathrm{beam}}$
after training. This choice avoids distributional dependency on the beam's
approximation errors and ensures the bandit is optimised directly for portfolio
damage.

### 4.7  Two-Step Rollout and Evaluation

At inference time, the policy executes a stateful 2-step greedy rollout:

1. **Step 1**: observe $\mathbf{o}_1 = [\mathbf{e}_r, \mathbf{z}_1, \mathbf{0}, 0]$
   (prior shock vector zero, progress zero); score all 250 actions;
   select $\mathbf{a}_1^* = \arg\max_\mathbf{a} \hat{R}(\mathbf{o}_1, \mathbf{a})$;
   apply shock to simulator; observe resulting state $\mathbf{z}_2$.

2. **Step 2**: construct $\mathbf{o}_2 = [\mathbf{e}_r, \mathbf{z}_2, \boldsymbol{\delta}_1, 0.5]$
   where $\boldsymbol{\delta}_1$ encodes the step-1 shock and $p_2 = 0.5 = 1/T$;
   score all 250 actions;
   select $\mathbf{a}_2^* = \arg\max_\mathbf{a} \hat{R}(\mathbf{o}_2, \mathbf{a})$.

The prior shock vector $\boldsymbol{\delta}_{t-1}$ in $\mathbf{o}_2$ distinguishes
this from a pure contextual bandit: the step-2 model observes what was already
shocked, enabling it to select a complementary second shock rather than repeating
the first.

**Beam search reference.** To establish a tractable upper reference, we run a
two-pass beam search with width $k = 6$. First, all 250 step-1 actions are
evaluated using a one-step environment; the top-$k$ actions by reward are
retained. Second, all $k^2 = 36$ combinations of (first action, second action)
from that top-$k$ set are evaluated as complete 2-step sequences; the sequence
with maximum portfolio loss is the reference output. This covers 36 of 62,500
possible sequences and is a tractable reference, not an exhaustive oracle.
All benchmark percentages are reported as the bandit's portfolio loss divided by
this beam reference's portfolio loss on the same seed.

**Evaluation protocol.** Both bandit and beam are evaluated on 16 held-out seeds
(20000–20015, unused during training or hyperparameter selection) with UCB
$\beta = 0.5$. The production metric is mean $\%_{\mathrm{beam}}$ across seeds;
the deployment gate is $\geq 85\%$.

### 4.8  PPO Ablation and the Case for Bandit Supervision

We initially trained the adversarial policy using Proximal Policy Optimisation
(PPO) across four successive iterations (v2–v5), progressively adding crisis-state
oversampling, a damage-first reward structure, and increased rollout length.

Despite these refinements, PPO converged to a degenerate context-free strategy on
all four profiles: the top discovered sequence was always a single maximum-magnitude
shock applied to the portfolio's highest-weight asset (XLY for balanced, XLK for
tech\_heavy, XLF for credit\_heavy, TLT for bond\_heavy). Measured against the
beam reference on 16 held-out seeds, PPO v5 achieved 50.8\% (balanced),
52.9\% (tech\_heavy), 76.5\% (bond\_heavy), and 56.3\% (credit\_heavy) —
falling below the domain-expert heuristic on three of four profiles. Bond\_heavy
is the exception: TLT as the primary shock is near-optimal for that profile, so
the degenerate strategy coincidentally produced a reasonable result despite
ignoring regime context.

Post-hoc analysis identified two causes:

1. **Sparse reward per episode.** With $T = 2$, the PPO critic receives one scalar
   per episode. Policy-gradient variance at this horizon requires far more rollouts
   than the environment can efficiently provide.

2. **Action space size relative to episode length.** With 250 actions and $T = 2$,
   the training signal was insufficiently dense for the policy to reliably
   distinguish profitable action sequences from uninformative ones within a
   practical rollout budget.

Switching to supervised reward regression — treating the problem as learning
$\hat{R}(\mathbf{o}, \mathbf{a})$ from exhaustively labelled $(\mathbf{o},
\mathbf{a}, R)$ triplets — resolved both issues. The reward model receives a
dense signal (one label per $(\mathbf{o}, \mathbf{a})$ pair), and the 87,500-sample
dataset covers the full action space at each step. The bandit's greedy rollout
recovers the policy by scoring all 250 actions at inference time.

This design choice — supervised reward regression over policy gradient — is the
key architectural decision that enabled training convergence.

---

## Appendix A — Action Space Variables

**Table A.1: The 25 core market variables in the action space**

| # | Ticker | Description | Type |
|---|---|---|---|
| 1 | \^GSPC | S&P 500 Index | Equity index |
| 2 | \^VIX | CBOE Volatility Index | Volatility |
| 3 | \^NDX | Nasdaq-100 Index | Equity index |
| 4 | \^RUT | Russell 2000 Index | Equity index |
| 5 | DGS10 | 10-Year Treasury Yield | Rate |
| 6 | DGS2 | 2-Year Treasury Yield | Rate |
| 7 | T10Y2Y | 10Y–2Y Yield Spread | Spread |
| 8 | FEDFUNDS | Federal Funds Rate | Rate |
| 9 | CL=F | Crude Oil (WTI Futures) | Commodity |
| 10 | GC=F | Gold Futures | Commodity |
| 11 | BAMLH0A0HYM2 | ICE BofA HY Master II Spread | Credit spread |
| 12 | BAMLH0A3HYC | ICE BofA HY CCC Spread | Credit spread |
| 13 | BAMLC0A0CM | ICE BofA IG Corporate Spread | Credit spread |
| 14 | XLF | Financial Select Sector SPDR | Equity sector |
| 15 | XLK | Technology Select Sector SPDR | Equity sector |
| 16 | XLE | Energy Select Sector SPDR | Equity sector |
| 17 | XLV | Health Care Select Sector SPDR | Equity sector |
| 18 | XLY | Consumer Discr. Select Sector SPDR | Equity sector |
| 19 | XLU | Utilities Select Sector SPDR | Equity sector |
| 20 | TLT | iShares 20+ Year Treasury Bond ETF | Fixed income |
| 21 | LQD | iShares iBoxx IG Corporate Bond ETF | Fixed income |
| 22 | HYG | iShares iBoxx HY Corporate Bond ETF | High yield |
| 23 | EEM | iShares MSCI Emerging Markets ETF | Emerging markets |
| 24 | CPIAUCSL | CPI All Urban Consumers (YoY) | Macro |
| 25 | UNRATE | US Unemployment Rate | Macro |

Magnitude bounds: $[-5.0\sigma, +5.0\sigma]$ in steps of $0.5\sigma$, with zero excluded.
Adverse direction enforced: price assets take negative shocks; spread/rate assets take positive shocks.
