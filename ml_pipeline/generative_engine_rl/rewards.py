"""
rewards.py
==========
Multi-component reward function for the adversarial RL scenario generator.

Components (per action_space.yaml Section 7):

    1. portfolio_loss          (weight 1.0)
       Adversarial drawdown signal. Higher loss -> higher reward.

    2. dfast_breach_severity   (weight 0.5)
       How badly the scenario violates DFAST 2026 stress thresholds.
       This is now wired through dfast_breach.py.

    3. causal_fidelity_bonus   (weight 0.3)
       Fraction of cross-variable movements that respect the causal-graph
       direction. Penalizes random shocks dressed up as causal scenarios.
       This is what differentiates causally-constrained adversarial search
       from unconstrained adversarial perturbation.

    4. diversity_bonus         (weight 0.1)
       Distance from the nearest historical crisis trajectory. Encourages
       discovery of NOVEL scenarios rather than rediscovering 2008/2020.

The agent maximizes the weighted sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np

from .action_space_loader import ActionSpaceSpec
from .portfolio_model import (
    compute_portfolio_loss,
    adversarial_reward_from_loss,
)


@dataclass
class RewardBreakdown:
    """Itemized reward components for logging and ablation analysis."""

    portfolio_loss: float
    dfast_breach: float
    causal_fidelity: float
    diversity: float
    total: float


RewardMode = Literal["portfolio_adversarial", "regulatory_adversarial"]


# ============================================================================
# COMPONENT IMPLEMENTATIONS
# ============================================================================

def reward_portfolio_loss(
    spec: ActionSpaceSpec,
    initial_state: Dict[str, float],
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    portfolio_weights: Dict[str, float] | None = None,
) -> float:
    """Component 1: drawdown reward."""
    simple_return = compute_portfolio_loss(
        initial_state=initial_state,
        trajectory=trajectory,
        var_index=var_index,
        log_return_vars=spec.log_return_vars,
        weights=portfolio_weights,
    )
    return adversarial_reward_from_loss(simple_return)


def reward_dfast_breach(
    spec: ActionSpaceSpec,
    initial_state: Dict[str, float],
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    interpret_as_sigma_units: bool = False,
) -> float:
    """Component 2: DFAST 2026 severely-adverse breach severity.

    Calls into generative_engine_rl.dfast_breach to score how badly the
    agent's trajectory exceeds the DFAST 2026 Q1 stress thresholds for
    UNRATE, BBB spread, ^GSPC, and ^VIX.

    Returns:
        - 0.0 if no thresholds are breached, or if trajectory is in sigma
          units (fast-mode stub) where comparison is not meaningful
        - >0 to ~1.0 for breach severities matching the DFAST scenario
        - >1.0 for catastrophic scenarios that exceed DFAST severity
    """
    from .dfast_breach import compute_dfast_breach_severity
    result = compute_dfast_breach_severity(
        trajectory=trajectory,
        var_index=var_index,
        initial_state=initial_state,
        interpret_as_sigma_units=interpret_as_sigma_units,
    )
    return float(result["severity"])


def reward_causal_fidelity(
    spec: ActionSpaceSpec,
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    causal_edges: List[Tuple[str, str, float]],
) -> float:
    """Component 3: fraction of variable comovements that follow causal edges.

    For each (source, target, weight) edge, check whether the sign of the
    target's terminal move is consistent with what the source's terminal
    move would predict (signed by edge weight).

    Returns: float in [0, 1] giving the fraction of edges respected.
    """
    if not causal_edges:
        return 0.0

    horizon = trajectory.shape[0]
    if horizon < 2:
        return 0.0

    respected = 0
    counted = 0
    for source, target, weight in causal_edges:
        si = var_index.get(source)
        ti = var_index.get(target)
        if si is None or ti is None:
            continue

        # Use cumulative move from start to end
        source_move = float(trajectory[-1, si] - trajectory[0, si])
        target_move = float(trajectory[-1, ti] - trajectory[0, ti])

        # Predicted target sign = sign(source_move * weight)
        predicted_sign = np.sign(source_move * weight)
        actual_sign = np.sign(target_move)

        if abs(source_move) < 1e-9 or abs(target_move) < 1e-9:
            # Skip cases where one side didn't move — uninformative
            continue

        counted += 1
        if predicted_sign == actual_sign:
            respected += 1

    if counted == 0:
        return 0.0
    return respected / counted


def reward_diversity(
    spec: ActionSpaceSpec,
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    historical_bank: List[np.ndarray] = None,
) -> float:
    """Component 4: distance to nearest historical crisis trajectory.

    Uses a small fixed reference bank built from the canonical 11 historical
    events in transformed daily-value space. The reward is scaled into [0, 1)
    as `1 - exp(-d_min)`, where d_min is the mean absolute distance to the
    nearest reference path across shared variables and horizon.
    """
    _ = spec
    if not historical_bank:
        return 0.0
    if trajectory.ndim != 2 or trajectory.shape[0] < 1:
        return 0.0

    dists = []
    for ref in historical_bank:
        if ref.ndim != 2:
            continue
        horizon = min(len(trajectory), len(ref))
        width = min(trajectory.shape[1], ref.shape[1])
        if horizon < 5 or width < 3:
            continue
        delta = np.abs(
            trajectory[:horizon, :width].astype(np.float64)
            - ref[:horizon, :width].astype(np.float64)
        )
        dists.append(float(np.mean(delta)))

    if not dists:
        return 0.0

    nearest = min(dists)
    return float(1.0 - np.exp(-nearest))


# ============================================================================
# COMPOSITE REWARD
# ============================================================================

def compute_reward(
    spec: ActionSpaceSpec,
    initial_state: Dict[str, float],
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    causal_edges: List[Tuple[str, str, float]],
    historical_bank: List[np.ndarray] = None,
    trajectory_in_sigma_units: bool = False,
    portfolio_weights: Dict[str, float] | None = None,
    reward_mode: RewardMode = "portfolio_adversarial",
    action_novelty_score: float = 0.0,
) -> RewardBreakdown:
    """Compute full multi-component reward, returning itemized breakdown.

    portfolio_adversarial mode (default):
        total = pl * (1 + lambda_cf * cf + lambda_dv * dv)
        Damage-first multiplicative formula: CF and diversity AMPLIFY damage
        rather than substitute for it. Zero damage → zero total regardless of
        causal fidelity or novelty. This prevents the agent gaming high CF on
        low-damage shocks. DFAST is computed for logging only (same reason as
        before: macro shocks that breach DFAST often cause equity rallies).
        lambda_cf = reward_weights["causal_fidelity_bonus"]  (default 0.3)
        lambda_dv = reward_weights["diversity_bonus"]         (default 0.1)

    regulatory_adversarial mode:
        DFAST is the primary objective; portfolio loss is a supporting term.
        Support gate on DFAST is retained to keep the primary objective active.

    Args:
        trajectory_in_sigma_units: True for fast-mode stub trajectories where
            values are standardized sigmas. The DFAST breach component is
            disabled in this case (returns 0). Other components still apply.
    """
    pl = reward_portfolio_loss(
        spec,
        initial_state,
        trajectory,
        var_index,
        portfolio_weights=portfolio_weights,
    )
    df = reward_dfast_breach(
        spec, initial_state, trajectory, var_index,
        interpret_as_sigma_units=trajectory_in_sigma_units,
    )
    cf = reward_causal_fidelity(spec, trajectory, var_index, causal_edges)
    # Use action novelty (cross-episode frequency) when available; fall back to
    # trajectory-vs-historical-bank distance (returns 0 when no bank is passed).
    dv = action_novelty_score if action_novelty_score > 0.0 else reward_diversity(
        spec, trajectory, var_index, historical_bank
    )

    w = spec.reward_weights
    if reward_mode == "portfolio_adversarial":
        # Damage-first: CF and diversity amplify damage, not substitute for it.
        # When pl=0 the total collapses to 0 regardless of CF/DV, so the agent
        # cannot game structural signals on low-damage shocks.
        lam_cf = w["causal_fidelity_bonus"]
        lam_dv = w["diversity_bonus"]
        total = pl * (1.0 + lam_cf * cf + lam_dv * dv)
    elif reward_mode == "regulatory_adversarial":
        # Regulatory search: DFAST breach is primary; portfolio loss secondary.
        support_gate = float(np.clip(df, 0.0, 1.0))
        total = (
            w["dfast_breach_severity"] * df
            + support_gate * (
                w["portfolio_loss"] * pl
                + w["causal_fidelity_bonus"] * cf
                + w["diversity_bonus"] * dv
            )
        )
    else:
        raise ValueError(f"Unknown reward_mode '{reward_mode}'")

    return RewardBreakdown(
        portfolio_loss=pl,
        dfast_breach=df,
        causal_fidelity=cf,
        diversity=dv,
        total=total,
    )
