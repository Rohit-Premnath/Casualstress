# generative_engine_rl — Phase 2B RL Adversarial Scenario Generator

**Status:** Scaffolding complete (v0.1.0). Training not yet implemented.

This package implements the adversarial reinforcement-learning scenario
generator described in Phase 2B of the CausalStress rebuild plan. It is
**fully isolated** from the canonical `generative_engine/` — if Phase 2B
training is unstable or descoped, this entire folder can be deleted with
zero impact on the canonical 90.0% test coverage headline.

## What's in this scaffold

| File | Purpose |
|---|---|
| `action_space_loader.py` | Loads `action_space.yaml` into runtime objects, exposes plausibility masks and clamping helpers |
| `portfolio_model.py` | Equal-weight equity-ETF portfolio (~30% ^GSPC, sectors, defensives). Used for the drawdown reward signal |
| `rewards.py` | Multi-component adversarial reward: portfolio_loss + dfast_breach + causal_fidelity + diversity |
| `causal_stress_env.py` | Gymnasium environment. 1-step MDP wrapping the canonical `generate_scenarios()` |
| `rollout.py` | Random-policy rollout test. Validates env contract end-to-end |
| `__init__.py` | Package init |

## Architecture summary

The environment is a **1-step MDP**. Each episode:

1. Reset returns the initial-state observation (25-dim variable vector at t=0)
2. Agent picks `(target_var, shock_magnitude, event_family)` — a single shock spec
3. Env clamps magnitude to spec bounds (with VIX special-cap)
4. Env checks causal-plausibility mask; rejects if invalid
5. Env calls `generate_scenarios()` to produce a 60-day trajectory
6. Env computes multi-component reward
7. Episode terminates

This design lets the canonical scenario generator stay untouched while making
the RL contract clean. The agent learns *what shock specifications produce
worst-case adversarial scenarios* — not how to step through 60 days one at a
time.

## Running the scaffolding tests

```bash
cd ml_pipeline
python3 -m generative_engine_rl.rollout
```

Expected output: 50 random-policy episodes complete with valid rewards,
plus targeted behavior tests for VIX clamping, determinism, and the
causal-plausibility rejection path.

## Integration with the canonical pipeline

The env consumes (read-only):

- `action_space.yaml` — the action space spec from Phase 1 Task 4
- `regime_causal_graphs.json` — for the per-regime plausibility mask
- `canonical_best_model.load_canonical_graph()` — for the production causal graph
- `generative_engine.scenario_generator.generate_scenarios` — for trajectory generation
- A `var_model` dict from `fit_regime_var()` — for VAR coefficients and covariances

The env writes nothing back to canonical state. Episode logs are kept in-memory
for the agent's training loop.

## What's NOT in scaffolding

These are explicit Phase 2B follow-ups (deferred to keep scaffolding scope tight):

1. **DFAST breach severity** — currently a stub returning 0. Wires into
   `regulatory_engine.compute_capital_breach_severity()` in Phase 4.
2. **Diversity bonus** — currently a stub returning 0. Requires building a
   reference bank of canonical 11-event trajectories first.
3. **PPO training loop** — agent training script. Comes next.
4. **Database persistence** — the `rl_episodes` and `rl_scenarios` tables.
   Will be added once we want to persist training-time scenarios for analysis.
5. **Regime-transition action space** — the stretch-goal regime-flipping
   feature is gated on PPO stability, descope candidate.

## Design decisions documented

### Why 1-step episodes?
The canonical scenario generator already runs full 60-day stochastic
trajectories internally. Reinventing that as a step-level MDP would mean
rebuilding the entire stochastic engine inside the env. Keeping the env
thin and treating each episode as a single-action rollout is cleaner,
faster, and makes the RL agent's job *to choose adversarial shock
specifications*, which is the actual research question.

### Why hybrid Dict action space?
PPO supports Dict action spaces in Stable-Baselines3. The alternative
(flatten to a single vector) loses the discrete/continuous structure
and would make the action mask harder to apply.

### Why reject invalid actions instead of masking?
Two options were considered:
- **(a)** clip to nearest valid target — simpler but creates training
  artifacts where the agent can't tell which actions it's "actually"
  picking
- **(b)** reject with negative reward — the agent learns to avoid invalid
  actions as a natural part of its policy

Option (b) was chosen. This will likely need a curriculum where the
plausibility constraint is gradually tightened as training progresses,
but the scaffolding supports it from day one.
