"""
rollout.py
==========
Random-policy rollout for environment sanity testing.

This script:
    1. Loads the action space spec
    2. Builds a stub VAR model (no DB required)
    3. Builds a stub scenario_fn that returns deterministic trajectories
    4. Constructs CausalStressEnv
    5. Runs N random-policy episodes
    6. Asserts the env contract holds:
       - Observation shape matches observation_space
       - Action sampling stays within bounds
       - Reward is finite
       - Causal plausibility mask is enforced
       - Invalid actions return INVALID_ACTION_PENALTY
       - Action info is populated correctly

Usage:
    python -m generative_engine_rl.rollout
    or
    python ml_pipeline/generative_engine_rl/rollout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.causal_stress_env import (
    CausalStressEnv,
    INVALID_ACTION_PENALTY,
)


# ============================================================================
# STUB VAR MODEL
# ============================================================================

def make_stub_var_model(spec, n_vars=25, lag=1, seed=42):
    """A minimal var_model dict matching what fit_regime_var() returns.
    
    Used for scaffolding tests so the env can run without a DB connection.
    Deterministic given seed — does not consume from any shared RNG.
    """
    rng = np.random.default_rng(seed)
    variables = list(spec.core_variables[:n_vars])
    d = len(variables)

    # Stable VAR(1) coefficients: small off-diagonal, stable diagonal
    B = np.eye(d) * 0.3 + rng.normal(0, 0.02, (d, d))
    # Pad with intercept row
    B_full = np.vstack([np.zeros((1, d)), B])

    # Symmetric PSD covariance
    A = rng.normal(0, 0.1, (d, d))
    cov_normal = A @ A.T + np.eye(d) * 0.01
    cov_crisis = cov_normal * 2.0

    means = np.zeros(d)
    stds = np.ones(d)

    return {
        "coefficients": B_full,
        "covariance_normal": cov_normal,
        "covariance_crisis": cov_crisis,
        "means": means,
        "stds": stds,
        "variables": variables,
        "lag": lag,
        "n_obs": 1000,
    }


def make_stub_causal_adjacency(spec, n_edges=80, seed=42):
    """Build a stub causal adjacency dict over CORE_VARIABLES.

    Uses the canonical "src->tgt" key format that matches what
    canonical_best_model.build_edge_map() produces and what the production
    generate_scenarios() consumes.

    Deterministic given seed — does not consume from any shared RNG.
    """
    rng = np.random.default_rng(seed)
    vars_ = spec.core_variables
    adj: dict = {}
    attempts = 0
    while len(adj) < n_edges and attempts < n_edges * 10:
        attempts += 1
        src = vars_[rng.integers(len(vars_))]
        tgt = vars_[rng.integers(len(vars_))]
        if src == tgt:
            continue
        key = f"{src}->{tgt}"
        if key in adj:
            continue
        adj[key] = {
            "weight": float(rng.normal(0, 0.3)),
            "lag": int(rng.integers(1, 4)),
            "edge_type": "lagged",
            "confidence": 1.0,
            "method": "stub",
        }
    return adj


# ============================================================================
# STUB SCENARIO_FN
# ============================================================================

def make_stub_scenario_fn(seed=0):
    """Returns a callable matching generate_scenarios() signature.

    Builds a deterministic 60-day trajectory whose terminal value reflects
    both the shock magnitude and a small VAR drift. Lets us verify that
    the env passes shock parameters into the generator and uses the
    returned trajectory in the reward.

    Determinism: each call seeds an internal RNG from (seed, target_var,
    rounded_magnitude, family) so identical (target, magnitude, family)
    inputs always produce identical trajectories. Different envs with the
    same constructor seed produce identical trajectories for the same
    inputs.
    """
    base_seed = seed

    def stub_fn(
        var_model, shock_variable, shock_magnitude,
        n_scenarios=1, horizon=60, causal_adjacency=None,
        use_multi_shock=False, event_type="market_crash",
        random_state=None,
    ):
        # Deterministic per-call seeding. If the env passes a random_state
        # we mix it into our sub-seed so episodes vary deterministically
        # given the env's seed.
        rs_mix = 0
        if random_state is not None:
            if isinstance(random_state, np.random.RandomState):
                rs_mix = int(random_state.randint(0, 2**31 - 1))
            else:
                rs_mix = int(random_state) & 0xFFFFFFFF

        sub_seed = (
            base_seed
            ^ (hash(shock_variable) & 0xFFFFFFFF)
            ^ (int(round(shock_magnitude * 100)) & 0xFFFF) << 8
            ^ (hash(event_type) & 0xFFFF)
            ^ rs_mix
        )
        rng = np.random.default_rng(abs(sub_seed))

        variables = var_model["variables"]
        d = len(variables)
        traj = rng.normal(0, 0.01, (horizon, d))
        if shock_variable in variables:
            i = variables.index(shock_variable)
            ramp = np.linspace(shock_magnitude, 0.0, horizon)
            traj[:, i] += ramp
        return [traj]

    return stub_fn


# ============================================================================
# ROLLOUT
# ============================================================================

def run_random_rollout(n_episodes: int = 50, seed: int = 0, verbose: bool = False):
    """Run N random-policy episodes and assert the env contract."""
    spec = load_spec()
    var_model = make_stub_var_model(spec)
    causal_adj = make_stub_causal_adjacency(spec)
    scenario_fn = make_stub_scenario_fn(seed=seed)

    env = CausalStressEnv(
        var_model=var_model,
        causal_adjacency=causal_adj,
        spec=spec,
        regime_at_episode_start="stressed",
        scenario_fn=scenario_fn,
        seed=seed,
        trajectory_in_sigma_units=True,
    )

    rewards = []
    n_invalid = 0
    n_valid = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)

        # Contract check 1: observation shape and dtype
        assert obs.shape == env.observation_space.shape, \
            f"obs shape {obs.shape} != {env.observation_space.shape}"
        assert obs.dtype == np.float32, f"obs dtype {obs.dtype} not float32"

        # Random action
        action = env.action_space.sample()

        # Contract check 2: action bounds
        m = float(np.asarray(action["shock_magnitude"]).reshape(-1)[0])
        lo, hi = spec.rl_action_bounds
        assert lo <= m <= hi, f"action mag {m} out of bounds [{lo}, {hi}]"
        assert 0 <= int(action["target_var"]) < len(spec.core_variables)
        assert 0 <= int(action["event_family"]) < len(spec.family_names)

        # Step
        obs2, reward, terminated, truncated, step_info = env.step(action)

        # Contract check 3: reward is finite
        assert np.isfinite(reward), f"reward {reward} not finite"

        # Contract check 4: 1-step MDP terminates immediately
        assert terminated is True, "expected terminated=True in 1-step MDP"
        assert truncated is False, "expected truncated=False"

        # Contract check 5: invalid actions return INVALID_ACTION_PENALTY
        if step_info.get("action_rejected"):
            assert reward == INVALID_ACTION_PENALTY, \
                f"rejected action gave reward {reward}, expected {INVALID_ACTION_PENALTY}"
            n_invalid += 1
        else:
            n_valid += 1
            # Contract check 6: valid actions populate reward_breakdown
            assert "reward_breakdown" in step_info
            br = step_info["reward_breakdown"]
            assert np.isfinite(br.total)

        rewards.append(reward)
        if verbose:
            print(
                f"  ep {ep:3d}: target={step_info['target_var']:<10} "
                f"family={step_info['family_name']:<20} "
                f"mag={step_info['magnitude']:+.2f} "
                f"reward={reward:+.4f} "
                f"{'REJECT' if step_info.get('action_rejected') else 'OK'}"
            )

    rewards = np.asarray(rewards)
    print(f"\n=== Rollout summary ({n_episodes} episodes) ===")
    print(f"  Valid actions:   {n_valid}/{n_episodes}")
    print(f"  Invalid actions: {n_invalid}/{n_episodes}")
    print(f"  Reward mean:     {rewards.mean():+.4f}")
    print(f"  Reward std:      {rewards.std():.4f}")
    print(f"  Reward range:    [{rewards.min():+.4f}, {rewards.max():+.4f}]")

    return {
        "n_episodes": n_episodes,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "rewards": rewards,
    }


def run_targeted_rollout(seed: int = 0):
    """Run hand-picked actions to verify specific behaviors."""
    spec = load_spec()
    var_model = make_stub_var_model(spec)
    causal_adj = make_stub_causal_adjacency(spec)
    scenario_fn = make_stub_scenario_fn(seed=seed)

    env = CausalStressEnv(
        var_model=var_model,
        causal_adjacency=causal_adj,
        spec=spec,
        regime_at_episode_start="stressed",
        scenario_fn=scenario_fn,
        seed=seed,
        trajectory_in_sigma_units=True,
    )

    print("\n=== Targeted behavior tests ===")

    # Test 1: VIX shock above cap should clamp
    obs, _ = env.reset(seed=1)
    vix_idx = spec.core_variables.index("^VIX")
    family_idx = spec.family_to_idx["volatility_shock"]
    action = {
        "target_var": vix_idx,
        "shock_magnitude": np.array([10.0], dtype=np.float32),  # absurd
        "event_family": family_idx,
    }
    obs2, reward, term, trunc, info = env.step(action)
    print(f"  VIX shock 10.0 -> clamped to {info['magnitude']}")
    assert info["magnitude"] == 3.5, f"VIX clamp failed: got {info['magnitude']}"

    # Test 2: Negative magnitude shock
    obs, _ = env.reset(seed=2)
    gspc_idx = spec.core_variables.index("^GSPC")
    crash_family = spec.family_to_idx["market_crash"]
    action = {
        "target_var": gspc_idx,
        "shock_magnitude": np.array([-3.0], dtype=np.float32),
        "event_family": crash_family,
    }
    obs2, reward, term, trunc, info = env.step(action)
    print(
        f"  ^GSPC -3σ market_crash -> reward={reward:+.4f} "
        f"(loss={info['reward_breakdown'].portfolio_loss:.4f}, "
        f"causal={info['reward_breakdown'].causal_fidelity:.4f})"
    )
    assert not info["action_rejected"], "GSPC should be a valid shock target"

    # Test 3: Episode determinism — same action + seed yields same reward
    env2 = CausalStressEnv(
        var_model=make_stub_var_model(spec),
        causal_adjacency=causal_adj,
        spec=spec,
        regime_at_episode_start="stressed",
        scenario_fn=make_stub_scenario_fn(seed=seed),
        seed=seed,
        trajectory_in_sigma_units=True,
    )
    obs2, _ = env2.reset(seed=2)
    obs2_, reward2, term, trunc, info2 = env2.step(action)
    print(f"  Determinism check: env1 reward={reward:.6f}, env2 reward={reward2:.6f}")
    # Allow tiny float drift but expect close
    assert abs(reward - reward2) < 1e-6, \
        f"non-deterministic: {reward} vs {reward2}"

    # Test 4: Causal plausibility rejection path
    # Construct a synthetic regime where only ^GSPC has outgoing edges.
    # Then shock anything else and verify rejection.
    spec_restrictive = load_spec()
    # Override the stressed-regime outgoing sources to a one-element set
    spec_restrictive.regime_outgoing_sources = {
        **spec_restrictive.regime_outgoing_sources,
        "stressed": {"^GSPC"},
    }
    env3 = CausalStressEnv(
        var_model=make_stub_var_model(spec_restrictive),
        causal_adjacency=causal_adj,
        spec=spec_restrictive,
        regime_at_episode_start="stressed",
        scenario_fn=make_stub_scenario_fn(seed=seed),
        seed=seed,
        trajectory_in_sigma_units=True,
    )
    obs3, info_reset = env3.reset(seed=3)
    # Confirm mask reports only ^GSPC valid
    valid_mask = info_reset["valid_action_mask"]
    n_valid_targets = sum(valid_mask)
    assert n_valid_targets == 1, \
        f"expected 1 valid target under restrictive mask, got {n_valid_targets}"

    # Try shocking an INVALID target (XLK)
    xlk_idx = spec_restrictive.core_variables.index("XLK")
    rate_family = spec_restrictive.family_to_idx["rate_shock"]
    bad_action = {
        "target_var": xlk_idx,
        "shock_magnitude": np.array([2.0], dtype=np.float32),
        "event_family": rate_family,
    }
    obs3_, reward3, term, trunc, info3 = env3.step(bad_action)
    assert info3["action_rejected"] is True, \
        f"expected rejection but action was accepted: {info3}"
    assert reward3 == INVALID_ACTION_PENALTY, \
        f"expected reward {INVALID_ACTION_PENALTY}, got {reward3}"
    print(
        f"  Causal-plausibility rejection: shocking XLK in restrictive regime "
        f"-> reward={reward3} ({info3['reason']})"
    )

    # And a VALID target (^GSPC) should be accepted
    gspc_idx_r = spec_restrictive.core_variables.index("^GSPC")
    obs3, _ = env3.reset(seed=3)
    good_action = {
        "target_var": gspc_idx_r,
        "shock_magnitude": np.array([-2.0], dtype=np.float32),
        "event_family": rate_family,
    }
    obs3_, reward4, term, trunc, info4 = env3.step(good_action)
    assert info4["action_rejected"] is False, \
        "expected ^GSPC to be valid under restrictive mask"
    print(f"  Causal-plausibility accept: ^GSPC accepted with reward={reward4:+.4f}")

    print("  All targeted behavior tests passed")


def main():
    print("=" * 60)
    print("CausalStress RL environment scaffolding rollout")
    print("=" * 60)
    print()

    # Random rollout
    res = run_random_rollout(n_episodes=50, seed=0)

    # Targeted tests
    run_targeted_rollout(seed=0)

    print("\n" + "=" * 60)
    print("Scaffolding rollout complete — env contract holds")
    print("=" * 60)
    return res


if __name__ == "__main__":
    main()