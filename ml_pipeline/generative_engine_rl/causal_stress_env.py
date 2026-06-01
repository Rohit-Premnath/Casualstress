"""
causal_stress_env.py
====================
Gymnasium environment for the adversarial RL scenario generator.

DESIGN
------
This is an EPISODE-LEVEL environment. Each episode is a single shock
specification + 60-day trajectory generation. The agent picks:

    target_var       : discrete, index into 25 CORE_VARIABLES
    shock_magnitude  : continuous in [-5.0, 5.0] sigma units
    event_family     : discrete, index into 8 shock families

The transition runs the existing canonical scenario generator
(generative_engine.scenario_generator.generate_scenarios) ONCE with these
parameters, producing a single 60-day trajectory. Reward is the multi-
component adversarial reward from rewards.py.

This is a 1-step MDP. The reward is rich (multi-component) and the action
space is rich (hybrid discrete+continuous), so PPO can learn meaningful
policies despite the trivial step-count.

CAUSAL PLAUSIBILITY MASK
------------------------
The agent's action is rejected if the chosen target_var has no outgoing
edges in the current regime's causal graph (per action_space.yaml Sec. 5).
A rejected action returns reward = -1.0 (configurable) and a no-op
trajectory. The agent learns to avoid invalid combinations naturally.

ISOLATION
---------
This module imports from canonical generative_engine but writes nothing
back to it. All env state is in-memory only.
"""

from __future__ import annotations

import inspect
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Allow this module to import from sibling generative_engine package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .action_space_loader import (   # noqa: E402
    ActionSpaceSpec,
    causal_plausibility_mask,
    clamp_shock_magnitude,
    load_spec,
)
from .rewards import RewardBreakdown, compute_reward    # noqa: E402
from .portfolio_model import (
    DEFAULT_PORTFOLIO_PROFILE,
    get_portfolio_weights,
    is_adverse_direction,
)


# ============================================================================
# CONFIG
# ============================================================================

INVALID_ACTION_PENALTY = -1.0
NOOP_TRAJECTORY_FILL = 0.0


# ============================================================================
# ENVIRONMENT
# ============================================================================

class CausalStressEnv(gym.Env):
    """One-step MDP wrapper around the canonical scenario generator.

    Args:
        var_model: dict from generative_engine.scenario_generator.fit_regime_var().
            Provides: coefficients, covariance matrices, means, stds, variables.
            For scaffolding, a stub fitter (StubVARModel) can be passed instead
            so the env can run without a database connection.
        causal_adjacency: dict {source: {target: weight}} loaded from
            regime_causal_graphs.json or canonical_best_model.load_canonical_graph().
        spec: ActionSpaceSpec from action_space_loader.load_spec().
        regime_at_episode_start: which HMM regime label the episode starts in.
            Determines which causal-plausibility mask is applied.
        scenario_fn: callable matching generate_scenarios() signature.
            Injected for testability — allows scaffolding without psycopg2.
        seed: optional RNG seed.

    Observation space: Box(25,) — initial state vector (variable values at t=0)
    Action space: Dict({
        target_var:     Discrete(25),
        shock_magnitude: Box(low=-5.0, high=5.0, shape=(1,)),
        event_family:   Discrete(8),
    })
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        var_model: Dict[str, Any],
        causal_adjacency: Dict[str, Any],
        spec: Optional[ActionSpaceSpec] = None,
        regime_at_episode_start: str = "stressed",
        scenario_fn=None,
        seed: Optional[int] = None,
        trajectory_in_sigma_units: bool = False,
        portfolio_profile: str = DEFAULT_PORTFOLIO_PROFILE,
        use_family_templates: bool = False,
        actions_per_episode: int = 1,
        reward_mode: str = "portfolio_adversarial",
        crisis_seed_prob: float = 0.0,
    ):
        super().__init__()

        if spec is None:
            spec = load_spec()
        self.spec = spec

        self.var_model = var_model
        self.causal_adjacency = causal_adjacency
        self.regime_at_episode_start = regime_at_episode_start
        self.scenario_fn = scenario_fn   # may be None until step() is called
        self._rng = np.random.default_rng(seed)
        self.trajectory_in_sigma_units = bool(trajectory_in_sigma_units)
        self.state_support = var_model.get("rl_state_support")
        self._crisis_seeds = list(var_model.get("rl_crisis_seeds", []))
        self._crisis_seed_prob = float(crisis_seed_prob)
        self.historical_bank = var_model.get("rl_historical_bank")
        self.portfolio_profile = str(portfolio_profile)
        self.portfolio_weights = get_portfolio_weights(self.portfolio_profile)
        self.use_family_templates = bool(use_family_templates)
        self.actions_per_episode = max(1, int(actions_per_episode))
        self.reward_mode = str(reward_mode)

        # Cache the variables this var_model knows about, in order
        self.var_model_vars: List[str] = list(var_model["variables"])
        self.var_model_idx: Dict[str, int] = {
            v: i for i, v in enumerate(self.var_model_vars)
        }

        # Observation = sampled real market state when available, else zeros.
        self.base_obs_dim = (
            int(self.state_support["obs_matrix"].shape[1])
            if self.state_support is not None
            else len(spec.core_variables)
        )
        self.root_shock_dim = len(spec.core_variables)
        progress_dim = 1
        obs_dim = self.base_obs_dim + self.root_shock_dim + progress_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Dict({
            "target_var": spaces.Discrete(len(spec.core_variables)),
            "shock_magnitude": spaces.Box(
                low=spec.rl_action_bounds[0],
                high=spec.rl_action_bounds[1],
                shape=(1,),
                dtype=np.float32,
            ),
            "event_family": spaces.Discrete(len(spec.family_names)),
        })

        # Episode state
        self._initial_state: Optional[np.ndarray] = None
        self._last_breakdown: Optional[RewardBreakdown] = None
        self._sampled_state_info: Optional[Dict[str, Any]] = None
        self._episode_step: int = 0
        self._pending_template: Dict[str, float] = {}
        self._pending_actions: List[Dict[str, Any]] = []

        # Cross-episode action history for novelty-based diversity reward.
        # Tracks (target_var_idx, family_idx) tuples over the last 100 episodes.
        # Intentionally NOT reset on reset() — diversity is cross-episode.
        self._action_history: deque = deque(maxlen=100)

    # ------------------------------------------------------------------
    # STEP / RESET / RENDER
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to a fresh episode start state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._episode_step = 0
        self._pending_template = {}
        self._pending_actions = []

        # Probabilistic crisis injection during PPO training rollouts.
        # Only fires when no explicit crisis_seed_idx was requested (teacher
        # data always sets one explicitly), and crisis seeds are available.
        if (
            self._crisis_seeds
            and self._crisis_seed_prob > 0.0
            and (options or {}).get("crisis_seed_idx") is None
            and self._rng.random() < self._crisis_seed_prob
        ):
            options = dict(options) if options else {}
            options["crisis_seed_idx"] = int(self._rng.integers(len(self._crisis_seeds)))

        crisis_idx = (options or {}).get("crisis_seed_idx")
        if (
            crisis_idx is not None
            and self._crisis_seeds
            and int(crisis_idx) < len(self._crisis_seeds)
        ):
            crisis = self._crisis_seeds[int(crisis_idx)]
            base_obs = np.asarray(crisis["obs_vector"], dtype=np.float32)
            self._sampled_state_info = {
                "row_index": -1,
                "date": crisis["date"],
                "regime_name": crisis["regime"],
            }
        elif self.state_support is not None:
            regime_rows = self.state_support["regime_to_rows"].get(
                self.regime_at_episode_start
            )
            if regime_rows is None or len(regime_rows) == 0:
                regime_rows = np.arange(len(self.state_support["obs_matrix"]))
            picked = int(self._rng.choice(regime_rows))
            base_obs = self.state_support["obs_matrix"][picked].astype(
                np.float32
            )
            self._sampled_state_info = {
                "row_index": picked,
                "date": str(self.state_support["dates"][picked])[:10],
                "regime_name": str(self.state_support["regimes"][picked]),
            }
        else:
            base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
            self._sampled_state_info = None

        self._initial_state = self._compose_observation(base_obs)

        info = {
            "regime_at_episode_start": self.regime_at_episode_start,
            "portfolio_profile": self.portfolio_profile,
            "reward_mode": self.reward_mode,
            "actions_per_episode": self.actions_per_episode,
            "valid_action_mask": causal_plausibility_mask(
                self.spec, self.regime_at_episode_start
            ),
        }
        if self._sampled_state_info is not None:
            info["sampled_state"] = dict(self._sampled_state_info)
        return self._initial_state.copy(), info

    def step(self, action: Dict[str, Any]):
        """Run a single shock specification through the canonical generator.

        Returns: (observation, reward, terminated, truncated, info)
        Terminated is always True — this is a 1-step MDP.
        """
        target_idx = int(action["target_var"])
        magnitude = float(np.asarray(action["shock_magnitude"]).reshape(-1)[0])
        family_idx = int(action["event_family"])

        target_var = self.spec.core_variables[target_idx]
        family_name = self.spec.family_names[family_idx]
        magnitude = clamp_shock_magnitude(self.spec, target_var, magnitude)

        # ---- Causal plausibility check ----
        mask = causal_plausibility_mask(self.spec, self.regime_at_episode_start)
        action_valid = mask[target_idx]
        direction_valid = is_adverse_direction(
            target_var,
            magnitude,
            self.spec.log_return_vars,
        )

        if not action_valid or not direction_valid:
            # Reject action
            reward = INVALID_ACTION_PENALTY
            obs = self._initial_state.copy() if self._initial_state is not None else \
                  np.zeros(self.observation_space.shape[0], dtype=np.float32)
            self._last_breakdown = None
            info = {
                "action_rejected": True,
                "reason": (
                    "target_var has no outgoing edges in current regime graph"
                    if not action_valid else
                    "shock direction is not adverse for this variable"
                ),
                "target_var": target_var,
                "family_name": family_name,
                "magnitude": magnitude,
                "regime": self.regime_at_episode_start,
                "portfolio_profile": self.portfolio_profile,
                "reward_mode": self.reward_mode,
            }
            return obs, reward, True, False, info

        self._episode_step += 1
        self._pending_template[target_var] = clamp_shock_magnitude(
            self.spec,
            target_var,
            self._pending_template.get(target_var, 0.0) + magnitude,
        )
        action_record = {
            "step_index": self._episode_step,
            "target_var": target_var,
            "family_name": family_name,
            "magnitude": magnitude,
        }
        self._pending_actions.append(action_record)

        if self._episode_step < self.actions_per_episode:
            obs = self._compose_observation()
            info = {
                "action_rejected": False,
                "intermediate_step": True,
                "step_index": self._episode_step,
                "actions_per_episode": self.actions_per_episode,
                "target_var": target_var,
                "family_name": family_name,
                "generator_event_type": "single_root_sequence",
                "magnitude": magnitude,
                "pending_template": dict(self._pending_template),
                "pending_actions": list(self._pending_actions),
                "regime": self.regime_at_episode_start,
                "portfolio_profile": self.portfolio_profile,
                "reward_mode": self.reward_mode,
            }
            if self._sampled_state_info is not None:
                info["sampled_state"] = dict(self._sampled_state_info)
            return obs, 0.0, False, False, info

        # ---- Run the canonical scenario generator ----
        if self.scenario_fn is None:
            raise RuntimeError(
                "scenario_fn was not injected. The env cannot run trajectory "
                "generation. Pass scenario_fn at construction time, or use "
                "the stub scenario_fn from rollout.py for scaffolding tests."
            )

        # Translate spec target_var to a var_model variable
        # If the var_model doesn't include target_var, fall back to ^GSPC
        # (matches existing scenario_generator.py warning behavior at line 581)
        gen_target = target_var if target_var in self.var_model_idx else "^GSPC"

        # Derive a per-episode seed from the env's RNG. This makes scenario
        # generation REPRODUCIBLE given the env's seed, and prevents the
        # generator from inheriting whatever scipy/numpy state happens to
        # be active in the process. Without this, generate_scenarios()
        # (which uses scipy.stats.student_t.rvs internally) was reading
        # process-global RNG state, producing trajectories that depended
        # on prior PPO/torch/SB3 calls — including outlier trajectories.
        episode_seed = int(self._rng.integers(0, 2**31 - 1))

        scenario_kwargs = dict(
            var_model=self.var_model,
            shock_variable=gen_target,
            shock_magnitude=magnitude,
            n_scenarios=1,        # one trajectory per RL episode
            horizon=self.spec.rl_episode_horizon_days,
            causal_adjacency=self.causal_adjacency,
            use_multi_shock=False,    # deterministic for RL training
            event_type=family_name if self.use_family_templates else "__rl_single_root__",
            custom_shock_template=dict(self._pending_template),
        )
        try:
            if "random_state" in inspect.signature(self.scenario_fn).parameters:
                scenario_kwargs["random_state"] = episode_seed
        except (TypeError, ValueError):
            pass

        scenarios = self.scenario_fn(**scenario_kwargs)

        # scenarios is a list of (n_scenarios, horizon, n_vars) or similar.
        # The canonical scenario_generator returns a list of dataframes/arrays;
        # we handle both shapes for scaffolding flexibility.
        trajectory = self._extract_trajectory(scenarios)

        # Build initial-state dict in real (un-standardized) space
        means = self.var_model["means"]
        initial_state = {
            v: float(means[i])
            for i, v in enumerate(self.var_model_vars)
        }

        # Convert causal_adjacency to list of (source, target, weight) tuples
        causal_edges = self._adjacency_to_edge_list(self.causal_adjacency)

        # Action novelty: 1 - frequency of (target_var, family) in recent history.
        # First episode gets 1.0 (fully novel); 100% concentrated policy gets ~0.01.
        action_key = (target_idx, family_idx)
        history_len = len(self._action_history)
        if history_len == 0:
            action_novelty = 1.0
        else:
            action_novelty = 1.0 - self._action_history.count(action_key) / history_len
        self._action_history.append(action_key)

        breakdown = compute_reward(
            spec=self.spec,
            initial_state=initial_state,
            trajectory=trajectory,
            var_index=self.var_model_idx,
            causal_edges=causal_edges,
            historical_bank=self.historical_bank,
            trajectory_in_sigma_units=self.trajectory_in_sigma_units,
            portfolio_weights=self.portfolio_weights,
            reward_mode=self.reward_mode,
            action_novelty_score=action_novelty,
        )

        self._last_breakdown = breakdown

        info = {
            "action_rejected": False,
            "intermediate_step": False,
            "step_index": self._episode_step,
            "actions_per_episode": self.actions_per_episode,
            "target_var": target_var,
            "family_name": family_name,
            "generator_event_type": scenario_kwargs["event_type"],
            "magnitude": magnitude,
            "regime": self.regime_at_episode_start,
            "portfolio_profile": self.portfolio_profile,
            "reward_mode": self.reward_mode,
            "pending_template": dict(self._pending_template),
            "pending_actions": list(self._pending_actions),
            "action_novelty": action_novelty,
            "reward_breakdown": breakdown,
            # Exposed for trajectory inspection. The trajectory is in
            # daily transform units (log_return for prices, first_diff for
            # rates). var_model_vars gives the column→code mapping.
            "trajectory": trajectory,
            "trajectory_vars": list(self.var_model_vars),
            "initial_state": initial_state,
        }
        if self._sampled_state_info is not None:
            info["sampled_state"] = dict(self._sampled_state_info)
        # Observation post-step is the same initial_state (1-step MDP)
        obs = self._initial_state.copy() if self._initial_state is not None else \
              np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, breakdown.total, True, False, info

    def render(self):
        """No rendering for now."""
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _extract_trajectory(self, scenarios) -> np.ndarray:
        """Coerce scenario generator output to a (horizon, n_vars) ndarray."""
        if isinstance(scenarios, list) and len(scenarios) > 0:
            s = scenarios[0]
        else:
            s = scenarios

        if hasattr(s, "values"):
            arr = np.asarray(s.values)
        else:
            arr = np.asarray(s)

        if arr.ndim == 3:
            # (n_scenarios, horizon, n_vars) -> take first scenario
            arr = arr[0]
        return arr.astype(np.float32)

    def _compose_observation(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        if base_obs is None:
            if self._initial_state is not None:
                base_obs = self._initial_state[: self.base_obs_dim]
            else:
                base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
        root_shocks = np.zeros(self.root_shock_dim, dtype=np.float32)
        for i, var_name in enumerate(self.spec.core_variables):
            if var_name in self._pending_template:
                root_shocks[i] = float(self._pending_template[var_name])
        progress = np.array(
            [float(self._episode_step) / float(self.actions_per_episode)],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                np.asarray(base_obs, dtype=np.float32),
                root_shocks,
                progress,
            ]
        )

    def _adjacency_to_edge_list(
        self, adj: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        """Parse causal adjacency into (source, target, weight) tuples.

        Accepts the canonical "src->tgt" key format used throughout the
        codebase: {"^GSPC->^VIX": {"weight": -0.5, "lag": 1, ...}, ...}.
        Also tolerates the nested {src: {tgt: weight}} legacy format.
        """
        edges = []
        for key, val in adj.items():
            if "->" in key:
                # Canonical format
                src, tgt = key.split("->", 1)
                if isinstance(val, dict):
                    weight = float(val.get("weight", 0.0))
                else:
                    weight = float(val)
                edges.append((src, tgt, weight))
            else:
                # Legacy nested format
                if isinstance(val, dict):
                    for tgt, weight in val.items():
                        edges.append((key, tgt, float(weight)))
        return edges

    @property
    def last_reward_breakdown(self) -> Optional[RewardBreakdown]:
        return self._last_breakdown
