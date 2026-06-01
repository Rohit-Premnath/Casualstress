"""
env_factory.py
==============
Single source of truth for constructing a properly-configured CausalStressEnv
ready for SB3 training. Both training and evaluation call this.

Two modes:

    fast: stub var_model + stub scenario_fn. Sub-second per episode.
          Used for smoke tests and PPO architecture iteration.

    real: real fit_regime_var() + real generate_scenarios(). Multi-second
          per episode. Used for production training runs.

Both modes return the same wrapped env so the agent code is mode-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .action_space_loader import ActionSpaceSpec, load_spec
from .action_wrapper import MultiDiscreteActionWrapper
from .causal_stress_env import CausalStressEnv
from .portfolio_model import DEFAULT_PORTFOLIO_PROFILE

# Make sibling generative_engine importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# FAST MODE — stub model and stub scenario_fn
# ============================================================================

def _make_stub_var_model(spec: ActionSpaceSpec, seed: int = 42) -> Dict[str, Any]:
    """Stub var_model. Mirrors generative_engine_rl.rollout.make_stub_var_model."""
    rng = np.random.default_rng(seed)
    variables = list(spec.core_variables)
    d = len(variables)
    B = np.eye(d) * 0.3 + rng.normal(0, 0.02, (d, d))
    B_full = np.vstack([np.zeros((1, d)), B])
    A = rng.normal(0, 0.1, (d, d))
    cov_normal = A @ A.T + np.eye(d) * 0.01
    return {
        "coefficients": B_full,
        "covariance_normal": cov_normal,
        "covariance_crisis": cov_normal * 2.0,
        "means": np.zeros(d),
        "stds": np.ones(d),
        "variables": variables,
        "lag": 1,
        "n_obs": 1000,
    }


def _make_stub_causal_adjacency(
    spec: ActionSpaceSpec, n_edges: int = 80, seed: int = 42
) -> Dict[str, dict]:
    """Stub adjacency in canonical 'src->tgt' key format."""
    rng = np.random.default_rng(seed)
    vars_ = spec.core_variables
    adj: Dict[str, dict] = {}
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


def _make_stub_scenario_fn(seed: int = 0) -> Callable:
    """Deterministic stub scenario_fn for fast-mode training."""
    base_seed = seed

    def stub_fn(
        var_model, shock_variable, shock_magnitude,
        n_scenarios=1, horizon=60, causal_adjacency=None,
        use_multi_shock=False, event_type="market_crash",
        random_state=None,
        custom_shock_template=None,
    ):
        # Mix random_state into the deterministic sub-seed so episodes vary
        # given the env's seed, while preserving determinism.
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
            # Add downstream shock propagation through causal adjacency
            # so the agent has a learnable pattern between shock and reward.
            if causal_adjacency:
                for key, edata in causal_adjacency.items():
                    if "->" not in key:
                        continue
                    src, tgt = key.split("->", 1)
                    if src == shock_variable and tgt in variables:
                        j = variables.index(tgt)
                        w = float(edata.get("weight", 0.0)) if isinstance(edata, dict) else float(edata)
                        ramp_t = np.linspace(shock_magnitude * w * 0.5, 0.0, horizon)
                        traj[:, j] += ramp_t
        return [traj]

    return stub_fn


# ============================================================================
# REAL MODE — delegates to real_mode_loader
# ============================================================================

def _load_real_var_model_and_graph(spec: ActionSpaceSpec):
    """Load production VAR + canonical causal graph + scenario_fn.

    Returns (var_model, causal_adjacency, scenario_fn). Raises RuntimeError
    with a descriptive message if any production resource is unavailable.

    The actual loading logic lives in real_mode_loader.py to keep the
    production-dependency imports lazy.
    """
    from .real_mode_loader import load_real_var_model_and_graph
    var_model, causal_adj, scenario_fn = load_real_var_model_and_graph()
    return var_model, causal_adj, scenario_fn


# ============================================================================
# FACTORY
# ============================================================================

def make_env(
    mode: str = "fast",
    seed: int = 0,
    regime_at_episode_start: str = "stressed",
    n_magnitude_bins: int = 21,
    spec: Optional[ActionSpaceSpec] = None,
    portfolio_profile: str = DEFAULT_PORTFOLIO_PROFILE,
    actions_per_episode: int = 1,
    use_family_templates: bool = False,
    reward_mode: str = "portfolio_adversarial",
    crisis_seed_prob: float = 0.0,
) -> MultiDiscreteActionWrapper:
    """Build a fully-wrapped CausalStressEnv ready for SB3.

    Args:
        mode: 'fast' for stub-based training, 'real' for production training
        seed: env seed
        regime_at_episode_start: HMM regime label that determines the
            causal-plausibility mask
        n_magnitude_bins: discretization granularity for the continuous
            magnitude action
        spec: optional pre-loaded ActionSpaceSpec (avoids re-parsing YAML
            for each env in a vectorized setup)

    Returns:
        MultiDiscreteActionWrapper around CausalStressEnv. Use this directly
        with SB3 PPO.
    """
    if spec is None:
        spec = load_spec()

    if mode == "fast":
        var_model = _make_stub_var_model(spec, seed=seed)
        causal_adj = _make_stub_causal_adjacency(spec, seed=seed)
        scenario_fn = _make_stub_scenario_fn(seed=seed)
        trajectory_in_sigma_units = True
    elif mode == "real":
        var_model, causal_adj, scenario_fn = _load_real_var_model_and_graph(spec)
        trajectory_in_sigma_units = False
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'fast' or 'real'.")

    env = CausalStressEnv(
        var_model=var_model,
        causal_adjacency=causal_adj,
        spec=spec,
        regime_at_episode_start=regime_at_episode_start,
        scenario_fn=scenario_fn,
        seed=seed,
        trajectory_in_sigma_units=trajectory_in_sigma_units,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
        crisis_seed_prob=crisis_seed_prob,
    )
    wrapped = MultiDiscreteActionWrapper(env, n_magnitude_bins=n_magnitude_bins)
    return wrapped


def env_factory_for_subproc(
    mode: str = "fast",
    base_seed: int = 0,
    rank: int = 0,
    regime_at_episode_start: str = "stressed",
    n_magnitude_bins: int = 21,
    portfolio_profile: str = DEFAULT_PORTFOLIO_PROFILE,
    actions_per_episode: int = 1,
    use_family_templates: bool = False,
    reward_mode: str = "portfolio_adversarial",
    crisis_seed_prob: float = 0.0,
) -> Callable[[], MultiDiscreteActionWrapper]:
    """Returns a closure suitable for SB3's make_vec_env env_fns argument.

    Each parallel env gets a deterministic-but-distinct seed = base_seed + rank.
    """
    def _thunk():
        return make_env(
            mode=mode,
            seed=base_seed + rank,
            regime_at_episode_start=regime_at_episode_start,
            n_magnitude_bins=n_magnitude_bins,
            portfolio_profile=portfolio_profile,
            actions_per_episode=actions_per_episode,
            use_family_templates=use_family_templates,
            reward_mode=reward_mode,
            crisis_seed_prob=crisis_seed_prob,
        )
    return _thunk
