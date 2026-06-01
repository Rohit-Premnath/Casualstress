"""
validate_fixes.py
=================
Targeted unit tests for the three RL reward fixes:

  FIX 1: portfolio_model.compute_portfolio_loss now uses MAX DRAWDOWN
          (min of cumulative path) instead of terminal return.

  FIX 2: rewards.compute_reward in portfolio_adversarial mode excludes
          DFAST from the training signal and has no support gate.

  FIX 3: env still works end-to-end with use_family_templates=True.

Run from ml_pipeline/:
    python -m generative_engine_rl.validate_fixes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.portfolio_model import (
    compute_portfolio_loss,
    adversarial_reward_from_loss,
    get_portfolio_weights,
)
from generative_engine_rl.rewards import compute_reward, RewardBreakdown
from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.env_factory import make_env

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    label = f"  [{status}] {name}"
    if detail:
        label += f"  ({detail})"
    print(label)
    results.append((name, condition))


# ============================================================================
# FIX 1: Max Drawdown
# ============================================================================
print("\n=== FIX 1: portfolio_loss uses max drawdown, not terminal return ===\n")

spec = load_spec()
weights = get_portfolio_weights("balanced")
n_vars = len(spec.core_variables)
var_index = {v: i for i, v in enumerate(spec.core_variables)}
horizon = 60

# Build a trajectory where ^GSPC crashes -30% at mid-horizon then RECOVERS to +10%
# Old code (terminal): portfolio gains → loss = 0   ← WRONG (hides crash)
# New code (drawdown): captures the -30% crash     ← CORRECT
traj_recover = np.zeros((horizon, n_vars), dtype=np.float32)
gspc_col = var_index["^GSPC"]
# First 30 days: cumulative -30% log return (big crash)
traj_recover[:30, gspc_col] = -0.30 / 30          # daily steps down
# Next 30 days: cumulative +40% log return (rally back past initial)
traj_recover[30:, gspc_col] = 0.40 / 30           # daily steps up

# Verify terminal cumsum = (-0.30 + 0.40) = +0.10 (portfolio gained!)
terminal_cum = float(np.sum(traj_recover[:, gspc_col]))
check("Terminal cumulative log return is +10% (old code would give gain)",
      abs(terminal_cum - 0.10) < 1e-4, f"terminal={terminal_cum:.4f}")

# compute_portfolio_loss returns simple_return (negative = loss, positive = gain).
# adversarial_reward_from_loss then converts: max(-simple_return, 0) / threshold.
initial_state = {v: 100.0 for v in spec.core_variables}
raw_return_recover = compute_portfolio_loss(
    initial_state=initial_state,
    trajectory=traj_recover,
    var_index=var_index,
    log_return_vars=spec.log_return_vars,
    weights=weights,
)
rl_reward_recover = adversarial_reward_from_loss(raw_return_recover)
# With balanced portfolio (30% ^GSPC), the worst drawdown is at day 30:
# weighted log return = 0.30 * (-0.30) = -0.09 → simple_return ≈ -8.6%
# adversarial_reward = max(0.086, 0) / 0.30 ≈ 0.287
check("Max drawdown fix: raw simple_return < 0 (loss detected in crash-then-recover)",
      raw_return_recover < 0.0, f"simple_return={raw_return_recover:.4f}")
check("Max drawdown fix: adversarial_reward > 0 when portfolio crashes then recovers",
      rl_reward_recover > 0.0, f"adversarial_reward={rl_reward_recover:.4f}")
check("Max drawdown fix: adversarial_reward in sensible range (~0.1-1.0 for partial drawdown)",
      0.05 < rl_reward_recover < 1.5, f"adversarial_reward={rl_reward_recover:.4f}")

# Contrast: a scenario where portfolio GAINS throughout (no drawdown)
traj_gain = np.zeros((horizon, n_vars), dtype=np.float32)
traj_gain[:, gspc_col] = 0.30 / horizon   # constant daily gains → cumsum always positive
raw_return_gain = compute_portfolio_loss(
    initial_state=initial_state,
    trajectory=traj_gain,
    var_index=var_index,
    log_return_vars=spec.log_return_vars,
    weights=weights,
)
rl_reward_gain = adversarial_reward_from_loss(raw_return_gain)
check("Monotone gain trajectory: adversarial_reward = 0 (no drawdown ever)",
      rl_reward_gain == 0.0, f"raw_return={raw_return_gain:.4f} reward={rl_reward_gain:.4f}")

# Contrast: a scenario where portfolio CRASHES throughout (no recovery)
traj_crash = np.zeros((horizon, n_vars), dtype=np.float32)
traj_crash[:, gspc_col] = -0.50 / horizon  # constant daily losses → cumsum always negative
raw_return_crash = compute_portfolio_loss(
    initial_state=initial_state,
    trajectory=traj_crash,
    var_index=var_index,
    log_return_vars=spec.log_return_vars,
    weights=weights,
)
rl_reward_crash = adversarial_reward_from_loss(raw_return_crash)
check("Monotone crash trajectory: raw_return < 0 (portfolio lost money)",
      raw_return_crash < 0.0, f"simple_return={raw_return_crash:.4f}")
check("Crash adversarial_reward > 0",
      rl_reward_crash > 0.0, f"adversarial_reward={rl_reward_crash:.4f}")
check("Crash reward > recover reward (persistent crash is penalized more than crash+recover)",
      rl_reward_crash > rl_reward_recover,
      f"crash={rl_reward_crash:.4f} > recover={rl_reward_recover:.4f}")


# ============================================================================
# FIX 2: Reward excludes DFAST from portfolio_adversarial, no gate
# ============================================================================
print("\n=== FIX 2: portfolio_adversarial reward excludes DFAST, has no gate ===\n")

# Build a trajectory where portfolio GAINS but DFAST is breached
# This mimics the UNRATE exploit: macro shock breaches DFAST but equities rally
traj_dfast_exploit = np.zeros((horizon, n_vars), dtype=np.float32)
# ^GSPC rallies (portfolio gains — BAD for old reward with DFAST)
traj_dfast_exploit[:, gspc_col] = 0.20 / horizon
# UNRATE spikes (DFAST breach — old reward rewarded this even though portfolio gained)
unrate_col = var_index["UNRATE"]
traj_dfast_exploit[:, unrate_col] = 7.0 / horizon  # +7pp total, breaches DFAST (+1.5pp threshold)

breakdown_exploit = compute_reward(
    spec=spec,
    initial_state={v: 4.0 if v == "UNRATE" else 100.0 for v in spec.core_variables},
    trajectory=traj_dfast_exploit,
    var_index=var_index,
    causal_edges=[],
    trajectory_in_sigma_units=False,
    portfolio_weights=weights,
    reward_mode="portfolio_adversarial",
)
check("DFAST exploit: portfolio_loss = 0 (equities rallied)",
      breakdown_exploit.portfolio_loss == 0.0,
      f"portfolio_loss={breakdown_exploit.portfolio_loss:.4f}")
check("DFAST exploit: dfast_breach > 0 (UNRATE breach detected)",
      breakdown_exploit.dfast_breach > 0.0,
      f"dfast_breach={breakdown_exploit.dfast_breach:.4f}")
check("DFAST exploit: total reward is LOW despite DFAST breach (DFAST excluded from portfolio mode)",
      breakdown_exploit.total < 0.5,
      f"total={breakdown_exploit.total:.4f} (should be ~0 since portfolio didn't lose)")

# Old code would have given: total = 0 + 1.0*(0.5*dfast + 0.3*cf + 0.1*dv)
# New code gives: total = 1.0*pl + 0.3*cf + 0.1*dv (no DFAST term)
# So with pl=0, cf≈0, total ≈ 0 under new code  ← correct

# Now test: with no gate, when portfolio DOES suffer a loss, secondary terms fire
traj_true_crash = np.zeros((horizon, n_vars), dtype=np.float32)
traj_true_crash[:, gspc_col] = -0.40 / horizon    # ^GSPC down -40%
xlf_col = var_index["XLF"]
traj_true_crash[:, xlf_col] = -0.35 / horizon     # XLF down -35%

breakdown_crash = compute_reward(
    spec=spec,
    initial_state=initial_state,
    trajectory=traj_true_crash,
    var_index=var_index,
    causal_edges=[],
    trajectory_in_sigma_units=False,
    portfolio_weights=weights,
    reward_mode="portfolio_adversarial",
)
check("Real crash: portfolio_loss > 0",
      breakdown_crash.portfolio_loss > 0.0,
      f"portfolio_loss={breakdown_crash.portfolio_loss:.4f}")
check("Real crash: total > portfolio_loss (causal_fidelity adds to reward)",
      breakdown_crash.total >= breakdown_crash.portfolio_loss,
      f"total={breakdown_crash.total:.4f} >= pl={breakdown_crash.portfolio_loss:.4f}")
check("Real crash: total is significantly positive (no gate killed the signal)",
      breakdown_crash.total > 0.3,
      f"total={breakdown_crash.total:.4f}")

# Verify DFAST is computed but NOT added to total in portfolio mode
# total should equal pl + 0.3*cf + 0.1*dv (not include df term)
w = spec.reward_weights
expected_total = (
    w["portfolio_loss"] * breakdown_crash.portfolio_loss
    + w["causal_fidelity_bonus"] * breakdown_crash.causal_fidelity
    + w["diversity_bonus"] * breakdown_crash.diversity
)
check("Reward formula: total == pl + cf_weight*cf + dv_weight*dv (no DFAST)",
      abs(breakdown_crash.total - expected_total) < 1e-6,
      f"total={breakdown_crash.total:.6f} expected={expected_total:.6f}")


# ============================================================================
# FIX 3: Env end-to-end with use_family_templates=True (fast mode)
# ============================================================================
print("\n=== FIX 3: Env works end-to-end with family templates enabled ===\n")

env = make_env(
    mode="fast",
    seed=42,
    use_family_templates=True,
    actions_per_episode=1,
    reward_mode="portfolio_adversarial",
)
check("Env construction: n_families > 1 when use_family_templates=True",
      env.env.use_family_templates and env.n_families > 1,
      f"n_families={env.n_families}")

obs, info = env.reset()
check("Env reset: obs shape correct",
      obs.shape == env.observation_space.shape,
      f"shape={obs.shape}")

# Step with action targeting ^GSPC (market_crash family, max magnitude)
action = np.array([
    env.valid_target_indices.index(spec.var_to_idx["^GSPC"])
    if spec.var_to_idx["^GSPC"] in env.valid_target_indices else 0,
    env.env.spec.family_to_idx.get("market_crash", 0) % env.n_families,
    len(env.adverse_levels) - 1,   # max adverse magnitude
], dtype=np.int64)

obs2, reward, terminated, truncated, step_info = env.step(action)
check("Env step: episode terminates",
      bool(terminated),
      f"terminated={terminated}")
check("Env step: action not rejected",
      not step_info.get("action_rejected", False),
      f"action_rejected={step_info.get('action_rejected')}")
check("Env step: reward is finite",
      np.isfinite(reward),
      f"reward={reward:.4f}")
check("Env step: reward_breakdown present",
      "reward_breakdown" in step_info,
      f"keys={list(step_info.keys())[:5]}")
env.close()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"Results: {n_pass}/{len(results)} PASSED  |  {n_fail} FAILED")
print("=" * 60)
if n_fail > 0:
    print("\nFailed tests:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
    sys.exit(1)
else:
    print("\nAll fixes verified. Ready for training.")
    sys.exit(0)
