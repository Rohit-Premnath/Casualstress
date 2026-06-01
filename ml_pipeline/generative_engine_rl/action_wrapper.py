"""
action_wrapper.py
=================
Wraps CausalStressEnv to expose a MultiDiscrete action space to the PPO agent.

The underlying env uses a Dict action space:
    target_var:      Discrete(25)
    shock_magnitude: Box(-5.0, 5.0)
    event_family:    Discrete(8)

Stable-Baselines3 PPO works with Dict observation spaces but not Dict ACTION
spaces — actions must be flat. We solve this by discretizing the continuous
shock_magnitude into N bins and exposing the whole action space as
MultiDiscrete([n_targets, n_families, n_magnitude_bins]).

Discretization is intentional: PPO learns discrete actions much faster than
continuous ones on small, low-frequency action spaces. For the eventual
research run, 21 magnitude bins (0.5σ resolution from -5σ to +5σ) is plenty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete

from .causal_stress_env import CausalStressEnv
from .action_space_loader import causal_plausibility_mask
from .portfolio_model import is_adverse_direction


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_MAGNITUDE_BINS = 21       # 0.5σ resolution from -5σ to +5σ


def build_magnitude_grid(
    n_bins: int = DEFAULT_MAGNITUDE_BINS,
    lo: float = -5.0,
    hi: float = 5.0,
) -> np.ndarray:
    """Symmetric magnitude grid centered on 0.

    Example for n_bins=21: [-5.0, -4.5, ..., 0.0, ..., 4.5, 5.0]
    """
    return np.linspace(lo, hi, n_bins, dtype=np.float32)


# ============================================================================
# WRAPPER
# ============================================================================

class MultiDiscreteActionWrapper(gym.Wrapper):
    """Exposes MultiDiscrete([n_targets, n_families, n_mag_bins]) actions
    to the agent, translating to the underlying Dict action.

    The wrapper preserves the env's observation space and reward function;
    only the action interface is changed.
    """

    def __init__(
        self,
        env: CausalStressEnv,
        n_magnitude_bins: int = DEFAULT_MAGNITUDE_BINS,
    ):
        super().__init__(env)
        valid_mask = causal_plausibility_mask(
            env.spec, env.regime_at_episode_start
        )
        self.valid_target_indices = [
            i for i, is_valid in enumerate(valid_mask) if is_valid
        ]
        if not self.valid_target_indices:
            self.valid_target_indices = list(range(len(env.spec.core_variables)))

        self.n_targets = len(self.valid_target_indices)
        self.n_families = len(env.spec.family_names) if env.use_family_templates else 1
        full_grid = build_magnitude_grid(int(n_magnitude_bins))
        self.adverse_levels = np.asarray(
            sorted({abs(float(m)) for m in full_grid if abs(float(m)) > 1e-12}),
            dtype=np.float32,
        )
        self.n_mag_bins = len(self.adverse_levels)

        self.action_space = MultiDiscrete(
            [self.n_targets, self.n_families, self.n_mag_bins]
        )

    def step(self, action: np.ndarray):
        # action is a length-3 array of integer indices
        target_idx, family_idx, mag_idx = int(action[0]), int(action[1]), int(action[2])
        real_target_idx = self.valid_target_indices[target_idx]
        target_var = self.env.spec.core_variables[real_target_idx]
        adverse_level = float(self.adverse_levels[mag_idx])
        probe_negative = -adverse_level
        magnitude = (
            probe_negative
            if is_adverse_direction(
                target_var,
                probe_negative,
                self.env.spec.log_return_vars,
            )
            else adverse_level
        )

        dict_action = {
            "target_var": real_target_idx,
            "shock_magnitude": np.array([magnitude], dtype=np.float32),
            "event_family": family_idx,
        }
        return self.env.step(dict_action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)

    # Convenience helpers for logging/eval
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        target_idx, family_idx, mag_idx = int(action[0]), int(action[1]), int(action[2])
        real_target_idx = self.valid_target_indices[target_idx]
        target_var = self.env.spec.core_variables[real_target_idx]
        adverse_level = float(self.adverse_levels[mag_idx])
        probe_negative = -adverse_level
        magnitude = (
            probe_negative
            if is_adverse_direction(
                target_var,
                probe_negative,
                self.env.spec.log_return_vars,
            )
            else adverse_level
        )
        return {
            "target_var": target_var,
            "family_name": (
                self.env.spec.family_names[family_idx]
                if self.env.use_family_templates else
                "single_root"
            ),
            "magnitude": magnitude,
        }
