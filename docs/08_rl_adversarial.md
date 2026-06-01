# RL Adversarial Stress Testing

## Overview

The RL adversarial module asks a different question than the rest of CausalStress. Rather than "what happens under scenario X?", it asks: "what is the **worst causally-grounded scenario** for this specific portfolio?" An RL agent learns to search over the causal scenario space and find the shock that maximizes portfolio loss, subject to the constraint that the shock must respect the causal graph.

This gives risk managers a portfolio-specific adversarial scenario — not hand-picked by a human analyst, but discovered by an agent that has learned from the data which shocks do the most damage to which portfolios.

**Status:** Phase 2B. PPO v2-v5 and bandit v1-v2 are trained and evaluated. Results are meaningful (PPO v2: mean_reward = 0.963 vs random = 0.451, +96.5%) but RL is early-stage relative to the canonical scenario generator. How RL integrates as a research contribution is being framed.

---

## Implementation

| File | Purpose |
|------|---------|
| [ml_pipeline/generative_engine_rl/causal_stress_env.py](ml_pipeline/generative_engine_rl/causal_stress_env.py) | Gymnasium environment wrapping the canonical scenario generator |
| [ml_pipeline/generative_engine_rl/rewards.py](ml_pipeline/generative_engine_rl/rewards.py) | Multi-component reward function |
| [ml_pipeline/generative_engine_rl/portfolio_model.py](ml_pipeline/generative_engine_rl/portfolio_model.py) | Portfolio profile definitions and adverse direction mapping |
| [ml_pipeline/generative_engine_rl/dfast_breach.py](ml_pipeline/generative_engine_rl/dfast_breach.py) | DFAST breach severity computation (diagnostic, not in reward) |
| [ml_pipeline/generative_engine_rl/adversarial_serve.py](ml_pipeline/generative_engine_rl/adversarial_serve.py) | REST inference server, pre-loaded models |
| [ml_pipeline/generative_engine_rl/train_ppo.py](ml_pipeline/generative_engine_rl/train_ppo.py) | PPO training loop (Stable-Baselines3) |
| [ml_pipeline/generative_engine_rl/neural_bandit.py](ml_pipeline/generative_engine_rl/neural_bandit.py) | BanditRewardNet: neural contextual bandit |
| [ml_pipeline/action_space/action_space.yaml](ml_pipeline/action_space/action_space.yaml) | Action space specification (single source of truth, locked 2026-05-05) |
| [ml_pipeline/action_space/verify_action_space.py](ml_pipeline/action_space/verify_action_space.py) | 12-invariant spec consistency checker |
| [ml_pipeline/RL_RESULTS_REPORT.md](ml_pipeline/RL_RESULTS_REPORT.md) | Full training log and results (1,186 lines) |

---

## Problem Framing

Traditional stress testing:
```
Input: scenario X (fixed by human analyst)
Output: portfolio P&L under scenario X
```

CausalStress adversarial:
```
Input: portfolio profile (balanced, tech_heavy, bond_heavy, credit_heavy)
Output: worst-case scenario X* = argmax_{X ∈ causal_space} portfolio_loss(X)
        subject to: X respects causal graph constraints
```

The RL agent learns to maximize portfolio loss rather than coverage or plausibility. It finds the shock combination that exploits the portfolio's specific vulnerabilities — identified from the causal structure rather than from hand-crafted templates.

---

## The 1-Step MDP

The environment is a 1-step Markov Decision Process (Gymnasium interface):

**Observation space:** 25-dimensional vector — the initial market state (current values of all 25 core variables, normalized).

**Action space (MultiDiscrete or Dict):**
```python
action = {
    "target_var": Discrete(25),          # which variable to shock
    "shock_magnitude": Box(-5.0, 5.0),   # shock in σ units
    "event_family": Discrete(8),         # which shock family template to use
}
```
With default settings (`use_family_templates=False`), only 1 family is active → effectively 25 × 10 = 250 combinations. With all 8 families: 25 × 8 × 10 = 2,000 combinations.

**Episode:** One episode = one (observation, action, reward) tuple. The agent picks a shock, the environment calls `generate_scenarios()` with that shock, computes the reward from the resulting trajectories, and the episode ends.

**Reset:** Each episode samples a new initial market state from the set of historical stressed-regime days (or crisis-seed days during curriculum training). The observation is the 25-variable snapshot for that day.

---

## Causal Plausibility Mask

Before any shock is executed, a plausibility mask checks whether the chosen target variable has outgoing edges in the stressed-regime causal graph. Variables with no outgoing edges (no causal influence on other variables) are rejected — shocking them produces no propagation and no adversarial damage. The mask is loaded from `action_space.yaml`.

---

## Reward Function

The production reward function (portfolio_adversarial mode):

```python
total = w_pl × portfolio_loss + w_cf × causal_fidelity + w_dv × diversity

where:
  portfolio_loss = -min(cumsum(daily_portfolio_returns))   # max drawdown, always ≥ 0
  causal_fidelity = fraction of causal edges respected (0-1, ≈0.61 for constrained shocks)
  diversity = normalized entropy of shocked variables (encourages exploration)

  w_pl = 1.0   (dominant: maximize damage)
  w_cf = 0.3   (secondary: stay causally coherent)
  w_dv = 0.1   (tertiary: don't collapse to one action)
```

**DFAST breach is NOT in the reward.** It is logged as a diagnostic metric (`logged_dfast_breach`) but excluded from the learning signal. See Bug Fix #1 below.

---

## Three Critical Bug Fixes

Before the production training run, three bugs were identified that each fundamentally broke the learning signal:

### Fix 1: DFAST Exploit (Reward Gaming)

**Bug:** The reward included `w_dfast × dfast_breach`. The VAR model showed that shocking UNRATE (unemployment) upward causes the Fed to cut rates, which causes equities to **rally**. The agent learned to trigger DFAST breaches via unemployment spikes — earning high reward while `portfolio_loss ≈ 0`.

**Fix:** Remove `dfast_breach` entirely from the reward. It is now only logged.

```python
# Before
total = w_pl × portfolio_loss + w_dfast × dfast_breach + w_cf × causal_fidelity

# After
total = w_pl × portfolio_loss + w_cf × causal_fidelity + w_dv × diversity
```

### Fix 2: Max Drawdown vs Terminal Return

**Bug:** `portfolio_loss = max(-sum(path), 0)` — computed at the terminal day 60. A scenario where the portfolio crashes 60% then fully recovers by day 60 showed `portfolio_loss ≈ 0`.

**Fix:** Use the minimum cumulative return over the full 60-day horizon:

```python
# Before
portfolio_loss = max(-sum(path), 0)

# After
portfolio_loss = -min(cumsum(path))   # always ≥ 0 when path has any negative cumulative return
```

This is also the correct definition for risk management: a portfolio manager who is 60% down at day 30 faces margin calls, forced liquidation, and potential ruin — regardless of subsequent recovery.

### Fix 3: Support Gate Killing Gradient

**Bug:** A gating term `clip(portfolio_loss, 0, 1)` was multiplied onto the causal fidelity and diversity terms:

```python
total = gate × (w_cf × causal_fidelity + w_dv × diversity)
where gate = clip(portfolio_loss, 0, 1)
```

In 60-80% of episodes (before Fix 2 corrected `portfolio_loss`), `portfolio_loss = 0` → `gate = 0` → `total = 0` → zero gradient everywhere. The policy couldn't learn because most episodes produced no signal.

**Fix:** Unconditional sum with no gate:

```python
total = w_pl × portfolio_loss + w_cf × causal_fidelity + w_dv × diversity
```

**After all three fixes: 21/21 unit tests passing.**

---

## Training Progression

### PPO v2 (balanced portfolio, 50k steps)

**Config:** Stable-Baselines3 PPO, learning_rate=3e-4, batch_size=256, n_steps=1024, gae_lambda=0.95.

**Warm-start:** Beam search over 2,000 action combinations × 12 seeds → behavioural cloning pre-training (8 epochs, accuracy 17% → 57%).

**Results:**
- Mean reward: 0.963 vs random baseline 0.451 (+96.5%)
- Max reward seen: 2.037 vs random max 0.968
- Training time: 6 minutes (CPU)
- Action concentration: 100% convergence to single best action discovered in beam search warm-start

### PPO v3 (all 4 profiles, over-specialization fix)

Multi-profile training identified an over-specialization bug: the v2 agent, when evaluated on other profiles, still picked the balanced-profile optimal shock. v3 trains separate policies per profile with profile-conditioned observations.

### PPO v4 (damage-first reward + crisis seeds)

- Reward reweighted to make portfolio_loss dominant (w_pl = 1.0, others reduced)
- Crisis-seeded rollouts: training episodes always start from historical crisis days (GFC, COVID, Volmageddon) rather than random days. Forces agent to learn from hard examples.
- Better convergence on bond_heavy and credit_heavy profiles.

### PPO v5 (100k steps, crisis seeds in rollouts)

- Extended training to 100k steps (doubled from v4).
- Crisis seeds maintained during rollout phase (not just warm-start).
- Most stable across all 4 profiles.

### Bandit v1 and v2 (Neural Contextual Bandit)

Alternative architecture: instead of PPO's policy gradient, train a `BanditRewardNet` — a neural network that maps `(state, action) → predicted_reward`. The agent selects the action maximizing the predicted reward (with UCB exploration).

**BanditRewardNet architecture:** Feed-forward network, 2 hidden layers. Input: concatenated [25-dim state, action_features]. Output: scalar reward prediction.

**v2 training (2-step MDP variant):** Agent picks first shock, observes the trajectory after day 5, picks a follow-up shock. Allows adaptive responses to early market dynamics.

---

## Held-Out Benchmark Results

Trained on 80% of historical stress events, evaluated on 20% held-out:

| Profile | RL mean_reward | Random mean_reward | Beam search (exhaustive) | RL/Beam |
|---------|---------------|-------------------|--------------------------|---------|
| balanced | 0.968 | 0.630 | 1.66 | 58% |
| tech_heavy | 1.048 | 0.509 | 1.696 | 62% |
| bond_heavy | 0.801 | 0.528 | 1.010 | 79% |
| credit_heavy | 0.836 | 0.509 | 1.244 | 67% |

RL consistently outperforms random (by 53-106%) but doesn't match the exhaustive beam search. The gap reflects that 50-100k training steps are insufficient for the RL agent to fully explore the 2,000+ action space. Bandit v2 shows smaller generalization gaps on bond_heavy and credit_heavy.

---

## Portfolio-Specific Worst Shocks

The key empirical result: different portfolios have different worst-case shocks, and the RL/brute-force agent finds them correctly.

| Profile | Worst shock (brute-force scan) | Max drawdown |
|---------|-------------------------------|-------------|
| balanced | XLE −5.0σ, pandemic_exogenous | Commodity-led recession hits diversified book broadly |
| tech_heavy | ^NDX −5.0σ, global_shock | Concentrated Nasdaq exposure maximally damaged |
| bond_heavy | ^NDX via contagion | Equity crash transmits through credit to TLT/LQD |
| credit_heavy | CL=F −5.0σ, credit_crisis | Oil crash → energy HY defaults → HYG/LQD blowout |

**Bandit v2 best sequences (balanced):**
- Beam search best: XLE −5.0σ → XLY −5.0σ (mean_reward = 1.82)
- Greedy bandit: XLK −5.0σ → XLF −5.0σ (mean_reward = 0.89)

---

## Action Space Specification

The formal action space is defined in [ml_pipeline/action_space/action_space.yaml](ml_pipeline/action_space/action_space.yaml) (locked 2026-05-05). This YAML is the single source of truth for:

- 25 core variable taxonomy and semantic categories
- Per-variable magnitude bounds (e.g., VIX capped at ±3.5σ)
- Shock family templates (matching scenario_generator.py exactly)
- Causal plausibility mask rules
- RL action space bounds (MultiDiscrete dimensions)
- Reward component weights

**Consistency verification:** [ml_pipeline/action_space/verify_action_space.py](ml_pipeline/action_space/verify_action_space.py) checks 12 invariants between the YAML spec and the production codebase. Exit 0 = consistent. Run before any modification to the action space or scenario generator.

---

## REST API (Production)

Pre-trained models are loaded at API startup by the lifespan context manager in [backend/app/main.py](backend/app/main.py):

```python
async def lifespan(app: FastAPI):
    app.state.adversarial_models = load_all_bandit_models()
    yield
```

**Endpoint:** `POST /api/v1/adversarial/worst-case`

**Request:**
```json
{
  "portfolio_profile": "balanced",     // or tech_heavy, bond_heavy, credit_heavy
  "n_seeds": 4,                       // random market starting states to evaluate
  "ucb_beta": 0.5                     // UCB exploration (0=greedy, 2=high exploration)
}
```

**Response:**
```json
{
  "profile": "balanced",
  "worst_sequence": [
    { "target_var": "XLE", "family_name": "pandemic_exogenous", "magnitude": -5.0 }
  ],
  "portfolio_loss": 0.187,            // max drawdown
  "causal_fidelity": 0.61,
  "diversity": 0.33,
  "dfast_breach": 0.45,               // logged only, not in reward
  "seeds_tried": 4,
  "seed_used": 2,
  "quality_note": "RL policy, 58% of beam quality"
}
```

**`GET /api/v1/adversarial/status`** returns which portfolio bandit models are loaded and their version tags.

---

## Runs Directory

Trained model artifacts are in [ml_pipeline/runs/](ml_pipeline/runs/):

```
runs/
├── ppo_fixed_v1_20260510_021405/        # fixed PPO (all 3 bugs corrected)
├── ppo_smoke_20260507_*/                # 10 smoke test runs
├── bandit_v1_{balanced,tech_heavy,...}/ # bandit v1 per profile
├── bandit_v2_{balanced,tech_heavy,...}/ # bandit v2 (2-step MDP)
├── portfolio_comparison/               # brute-force worst-shock comparison
└── benchmark_summary.json              # aggregate benchmark across all variants
```

Each run directory contains `config.json` (training hyperparameters) and `eval_results.json` (held-out evaluation metrics).
