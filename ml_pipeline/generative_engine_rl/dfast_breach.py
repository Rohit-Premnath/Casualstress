"""
dfast_breach.py
================
Compute DFAST 2026 severely-adverse breach severity for an arbitrary
60-day RL-generated trajectory.

DESIGN PRINCIPLES
-----------------
1. **No DB calls.** Hits to Postgres in the reward loop would dominate
   training time. The DFAST 2026 thresholds are baked in as constants here,
   sourced from regulatory_engine.DFAST_2026_SEVERELY_ADVERSE.

2. **Cumulative-transform space.** This is the subtle but critical part.
   The agent's trajectory is in TRANSFORMED daily units, not absolute
   levels:
     - log_return variables (^GSPC, ^VIX): each row is a daily log return
     - first_diff variables (UNRATE, BAMLH0A0HYM2): each row is a daily
       first-difference

   So the trajectory's TERMINAL value at row 60 is the daily transform
   on day 60 — NOT the cumulative level after 60 days. To compare against
   DFAST levels we need to compute the cumulative move:
     - log_return: cumulative_log_return = sum(trajectory[:, var])
                   level = initial_level * exp(cumulative_log_return)
     - first_diff: cumulative_diff = sum(trajectory[:, var])
                   level = initial_level + cumulative_diff

   We express thresholds as REQUIRED CUMULATIVE MOVES from baseline:
     - ^GSPC down to 75% of baseline = cumulative log return of log(0.75) = -0.288
     - VIX rising from typical ~20 to 45 = cumulative log return of log(45/20) = +0.811
     - UNRATE rising from typical ~4 to 5.5 = cumulative first_diff of +1.5
     - BBB spread rising from typical ~2 to 4.5 = cumulative first_diff of +2.5

   This avoids needing to reconstruct absolute levels from the env's
   var_model means (which are themselves in standardized space).

3. **Variable code translation.** DFAST defines 5 stress variables. We
   match them to the agent's variable codes used in CORE_VARIABLES:

     DFAST variable    | our_code         | transform_class
     ------------------|------------------|---------------
     GDP_GROWTH        | A191RL1Q225SBEA  | quarterly only — excluded
     UNEMPLOYMENT      | UNRATE           | first_diff
     BBB_SPREAD        | BAMLH0A0HYM2     | first_diff
     EQUITY_PRICES     | ^GSPC            | log_return
     VIX               | ^VIX             | log_return

4. **Stub-aware.** If `interpret_as_sigma_units=True` (fast mode),
   returns 0.0 because the comparison is not meaningful for stubbed
   sigma-scale trajectories.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ============================================================================
# DFAST 2026 SEVERELY ADVERSE — REQUIRED CUMULATIVE MOVES
# ============================================================================
# Thresholds are expressed as cumulative moves from baseline over the
# trajectory horizon. A breach occurs when the cumulative move exceeds
# (or falls below, for "down" direction) the threshold.

DFAST_2026_Q1_CUMULATIVE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "UNRATE": {
        "transform": "first_diff",
        "direction": "up",
        "cumulative_threshold": 1.5,    # 4.0 (baseline) + 1.5 = 5.5 (DFAST Q1)
        "baseline_assumption": 4.0,
        "q1_level": 5.5,
        "description": "Unemployment rate cumulative rise (pp)",
    },
    "BAMLH0A0HYM2": {
        "transform": "first_diff",
        "direction": "up",
        "cumulative_threshold": 2.5,    # 2.0 (baseline) + 2.5 = 4.5 (DFAST Q1)
        "baseline_assumption": 2.0,
        "q1_level": 4.5,
        "description": "BBB/HY spread cumulative widening (pp)",
    },
    "^GSPC": {
        "transform": "log_return",
        "direction": "down",
        "cumulative_threshold": float(np.log(0.75)),    # -0.2877: 25% drop
        "baseline_assumption": 100.0,
        "q1_level": 75.0,
        "description": "Equity cumulative log return (down 25% = breach)",
    },
    "^VIX": {
        "transform": "log_return",
        "direction": "up",
        "cumulative_threshold": float(np.log(45.0 / 20.0)),    # +0.811
        "baseline_assumption": 20.0,
        "q1_level": 45.0,
        "description": "VIX cumulative log return (rise from ~20 to ~45)",
    },
}


# ============================================================================
# BREACH SEVERITY COMPUTATION
# ============================================================================

def compute_dfast_breach_severity(
    trajectory: np.ndarray,
    var_index: Dict[str, int],
    initial_state: Optional[Dict[str, float]] = None,
    interpret_as_sigma_units: bool = False,
) -> Dict[str, Any]:
    """Compute DFAST 2026 severely-adverse breach severity for a trajectory.

    Args:
        trajectory: (horizon, n_vars) — agent's generated trajectory in
            DAILY TRANSFORM units (log_return for prices, first_diff for
            rates/spreads). Values are real (not standardized).
        var_index: variable code -> column index in trajectory.
        initial_state: unused in current implementation (kept for API
            stability). Cumulative comparisons are baseline-independent.
        interpret_as_sigma_units: True for fast-mode stub trajectories
            where values are standardized sigmas. Returns 0.0 in that case.

    Returns:
        dict with keys:
          - 'severity': float, mean per-variable breach severity
          - 'per_variable': dict of cumulative moves and breaches
          - 'tracked_variables': count of DFAST variables found
          - 'reason': explanation when severity is 0
    """
    if interpret_as_sigma_units:
        return {
            "severity": 0.0,
            "per_variable": {},
            "tracked_variables": 0,
            "reason": "trajectory in sigma units, DFAST comparison not meaningful",
        }

    if trajectory.ndim != 2 or trajectory.shape[0] < 1:
        return {
            "severity": 0.0,
            "per_variable": {},
            "tracked_variables": 0,
            "reason": f"invalid trajectory shape {trajectory.shape}",
        }

    per_variable: Dict[str, Dict[str, float]] = {}
    breaches = []

    for code, info in DFAST_2026_Q1_CUMULATIVE_THRESHOLDS.items():
        col = var_index.get(code)
        if col is None:
            continue

        cumulative_move = float(np.sum(trajectory[:, col]))
        threshold = float(info["cumulative_threshold"])
        direction = info["direction"]

        if direction == "down":
            breach = max(0.0, threshold - cumulative_move) / max(abs(threshold), 1e-6)
        else:
            breach = max(0.0, cumulative_move - threshold) / max(abs(threshold), 1e-6)

        baseline = float(info["baseline_assumption"])
        if info["transform"] == "log_return":
            implied_level = baseline * float(np.exp(cumulative_move))
        else:
            implied_level = baseline + cumulative_move

        per_variable[code] = {
            "cumulative_move": cumulative_move,
            "cumulative_threshold": threshold,
            "transform": info["transform"],
            "direction": direction,
            "implied_terminal_level": implied_level,
            "q1_level": float(info["q1_level"]),
            "baseline_assumption": baseline,
            "breach": breach,
            "description": info["description"],
        }
        breaches.append(breach)

    if not breaches:
        return {
            "severity": 0.0,
            "per_variable": {},
            "tracked_variables": 0,
            "reason": "none of the 4 daily DFAST variables present in trajectory",
        }

    severity = float(np.mean(breaches))
    return {
        "severity": severity,
        "per_variable": per_variable,
        "tracked_variables": len(breaches),
        "reason": "" if severity > 0 else "no DFAST thresholds breached",
    }


# ============================================================================
# CONVENIENCE
# ============================================================================

def list_tracked_variables() -> list:
    """Return the list of variable codes the breach scorer tracks."""
    return list(DFAST_2026_Q1_CUMULATIVE_THRESHOLDS.keys())


# Backwards-compat alias for the old constant name
DFAST_2026_Q1_THRESHOLDS = DFAST_2026_Q1_CUMULATIVE_THRESHOLDS