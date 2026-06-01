"""
portfolio_model.py
==================
Named portfolio profiles for the adversarial RL/search layer.

The key product question is not "what is the generic worst shock?" but
"what is the worst plausible shock for this portfolio?"  This module
provides a few portfolio profiles with clearly different weak points so
we can benchmark and train against differentiated exposures.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np


# ============================================================================
# PORTFOLIO PROFILES
# ============================================================================

BALANCED_PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "^GSPC": 0.30,
    "XLF": 0.15,
    "XLK": 0.15,
    "XLE": 0.10,
    "XLV": 0.10,
    "XLY": 0.10,
    "XLU": 0.05,
    "TLT": 0.05,
}

TECH_HEAVY_PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "XLK": 0.35,
    "^NDX": 0.25,
    "^GSPC": 0.15,
    "XLY": 0.10,
    "XLF": 0.05,
    "EEM": 0.05,
    "TLT": 0.03,
    "GC=F": 0.02,
}

BOND_HEAVY_PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "TLT": 0.40,
    "LQD": 0.27,
    "HYG": 0.12,
    "XLU": 0.09,
    "GC=F": 0.06,
    "^GSPC": 0.06,
}

CREDIT_HEAVY_PORTFOLIO_WEIGHTS: Dict[str, float] = {
    "HYG": 0.36,
    "LQD": 0.28,
    "XLF": 0.16,
    "^GSPC": 0.10,
    "TLT": 0.06,
    "GC=F": 0.04,
}

PORTFOLIO_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": BALANCED_PORTFOLIO_WEIGHTS,
    "tech_heavy": TECH_HEAVY_PORTFOLIO_WEIGHTS,
    "bond_heavy": BOND_HEAVY_PORTFOLIO_WEIGHTS,
    "credit_heavy": CREDIT_HEAVY_PORTFOLIO_WEIGHTS,
}

DEFAULT_PORTFOLIO_PROFILE = "balanced"


ADVERSE_UP_VARS: Set[str] = {
    "^VIX",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "FEDFUNDS",
    "UNRATE",
    "BAMLH0A0HYM2",
    "BAMLH0A3HYC",
    "BAMLC0A0CM",
}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    return {asset: float(weight) / total for asset, weight in weights.items()}


def get_portfolio_weights(profile: str = DEFAULT_PORTFOLIO_PROFILE) -> Dict[str, float]:
    if profile not in PORTFOLIO_PROFILES:
        raise KeyError(
            f"Unknown portfolio profile '{profile}'. "
            f"Expected one of {sorted(PORTFOLIO_PROFILES)}."
        )
    return normalize_weights(PORTFOLIO_PROFILES[profile])


def is_adverse_direction(
    var_name: str,
    magnitude: float,
    log_return_vars: Set[str],
) -> bool:
    """Return True when the shock direction is economically adverse.

    For a long-only portfolio:
      - investable/log-return assets are adverse when shocked DOWN
      - volatility, rates, unemployment, and credit spreads are adverse when UP
    """
    if abs(float(magnitude)) < 1e-12:
        return False
    if var_name in ADVERSE_UP_VARS:
        return float(magnitude) > 0.0
    if var_name in log_return_vars:
        return float(magnitude) < 0.0
    return float(magnitude) > 0.0


# ============================================================================
# RETURN / LOSS COMPUTATION
# ============================================================================

def compute_portfolio_loss(
    initial_state: Dict[str, float],
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    log_return_vars: Set[str],
    weights: Dict[str, float] | None = None,
) -> float:
    """Compute the portfolio worst-case drawdown over the generated trajectory.

    The canonical generator returns DAILY transformed values:
      - log_return variables: per-day log returns
      - first_diff variables: per-day level changes

    We use the MINIMUM of the cumulative path (max drawdown), not the terminal
    value. This captures stress scenarios where a portfolio crashes mid-horizon
    then recovers — the terminal return would show a gain, hiding the drawdown.
    For adversarial search, what matters is how bad it gets, not where it ends.
    """
    if weights is None:
        weights = get_portfolio_weights(DEFAULT_PORTFOLIO_PROFILE)
    else:
        weights = normalize_weights(weights)

    total_log_return = 0.0
    for asset, weight in weights.items():
        col = var_index.get(asset)
        if col is None:
            continue
        if asset in log_return_vars:
            # Use worst cumulative log-return along the path (max drawdown).
            cumulative_path = np.cumsum(trajectory[:, col])
            cum_log_return = float(np.min(cumulative_path))
        else:
            initial = initial_state.get(asset, 0.0)
            if initial == 0:
                cum_log_return = 0.0
            else:
                # Reconstruct worst level reached: initial + min(cumulative diffs).
                cumulative_diffs = np.cumsum(trajectory[:, col])
                min_level = initial + float(np.min(cumulative_diffs))
                cum_log_return = float(np.log(max(min_level / initial, 1e-9)))
        total_log_return += weight * cum_log_return

    return float(np.expm1(total_log_return))


def adversarial_reward_from_loss(
    simple_return: float,
    severe_loss_threshold: float = 0.30,
) -> float:
    """Map negative portfolio return into a normalized adversarial reward."""
    raw_loss = max(-simple_return, 0.0)
    if severe_loss_threshold <= 0:
        return raw_loss
    return raw_loss / severe_loss_threshold
