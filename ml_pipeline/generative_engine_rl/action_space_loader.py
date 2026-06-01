"""
action_space_loader.py
======================
Loads the action_space.yaml specification and exposes Python objects the RL
environment can use directly.

This is the bridge between the YAML spec and runtime code. If the spec drifts
from scenario_generator.py, verify_action_space.py will catch it — this loader
trusts the spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent  # .../ml_pipeline/
SPEC_PATH = ROOT / "action_space" / "action_space.yaml"
REGIME_GRAPH_PATH = ROOT / "regime_causal_graphs.json"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ShockFamily:
    name: str
    description: str
    historical_events: List[str]
    shock_template: Dict[str, float]   # variable -> sigma magnitude


@dataclass
class ActionSpaceSpec:
    """Runtime view of action_space.yaml."""

    schema_version: str
    core_variables: List[str]                        # 25 ordered variables
    var_to_idx: Dict[str, int]                       # reverse lookup
    log_return_vars: Set[str]
    first_diff_vars: Set[str]

    vix_template_cap: float
    vix_propagation_scale: float

    shock_families: Dict[str, ShockFamily]           # 8 families incl. 'pandemic' alias
    family_names: List[str]                          # ordered for discrete agent action
    family_to_idx: Dict[str, int]

    guardrails: Dict[str, Dict[str, dict]]           # family -> var -> rule dict

    rl_action_bounds: Tuple[float, float]            # global RL magnitude bounds
    rl_vix_special_caps: Tuple[float, float]
    rl_episode_horizon_days: int

    reward_weights: Dict[str, float]

    # Causal plausibility (loaded separately from regime_causal_graphs.json)
    regime_outgoing_sources: Dict[str, Set[str]] = field(default_factory=dict)


# ============================================================================
# LOADER
# ============================================================================

def load_spec(spec_path: Path = SPEC_PATH,
              regime_path: Path = REGIME_GRAPH_PATH) -> ActionSpaceSpec:
    """Load and parse action_space.yaml + regime_causal_graphs.json into runtime objects."""
    if not spec_path.exists():
        raise FileNotFoundError(
            f"action_space.yaml not found at {spec_path}. Run Phase 1 Task 4."
        )
    if not regime_path.exists():
        raise FileNotFoundError(
            f"regime_causal_graphs.json not found at {regime_path}."
        )

    with open(spec_path) as f:
        raw = yaml.safe_load(f)

    # ---- Variables ----
    core = list(raw["variables"]["core"])
    var_to_idx = {v: i for i, v in enumerate(core)}

    # ---- Shock families ----
    families: Dict[str, ShockFamily] = {}
    for name, fd in raw["shock_families"].items():
        # Resolve 'pandemic' alias -> 'pandemic_exogenous'
        if "shock_template_inherits" in fd:
            parent = fd["shock_template_inherits"]
            template = dict(raw["shock_families"][parent]["shock_template"])
        else:
            template = dict(fd.get("shock_template", {}))

        families[name] = ShockFamily(
            name=name,
            description=fd.get("description", ""),
            historical_events=list(fd.get("historical_events", [])),
            shock_template=template,
        )

    family_names = sorted(families.keys())
    family_to_idx = {n: i for i, n in enumerate(family_names)}

    # ---- RL action bounds ----
    rl_bounds = tuple(raw["rl_action_space"]["shock_magnitude"]["bounds"])
    rl_vix = tuple(raw["rl_action_space"]["shock_magnitude"]["special_caps"]["^VIX"])
    horizon = int(raw["rl_action_space"]["episode"]["horizon_days"])

    # ---- Reward weights ----
    reward_weights = {
        c: float(spec["weight"])
        for c, spec in raw["rl_reward"]["components"].items()
    }

    # ---- Regime causal graphs ----
    with open(regime_path) as f:
        rg = json.load(f)
    regime_outgoing: Dict[str, Set[str]] = {}
    for regime_name, regime_data in rg["regimes"].items():
        regime_outgoing[regime_name] = {e["source"] for e in regime_data["edges"]}

    return ActionSpaceSpec(
        schema_version=raw["schema_version"],
        core_variables=core,
        var_to_idx=var_to_idx,
        log_return_vars=set(raw["variables"]["log_return"]),
        first_diff_vars=set(raw["variables"]["first_diff"]),
        vix_template_cap=float(raw["variables"]["vix_template_cap"]),
        vix_propagation_scale=float(raw["variables"]["vix_propagation_scale"]),
        shock_families=families,
        family_names=family_names,
        family_to_idx=family_to_idx,
        guardrails=dict(raw["guardrails"]),
        rl_action_bounds=rl_bounds,
        rl_vix_special_caps=rl_vix,
        rl_episode_horizon_days=horizon,
        reward_weights=reward_weights,
        regime_outgoing_sources=regime_outgoing,
    )


# ============================================================================
# HELPERS
# ============================================================================

def causal_plausibility_mask(spec: ActionSpaceSpec, regime_name: str) -> List[bool]:
    """Returns a list of length len(core_variables); True if the variable has
    outgoing edges in the given regime's causal graph (i.e. is a valid shock
    target under the causal-plausibility constraint).
    """
    sources = spec.regime_outgoing_sources.get(regime_name)
    if sources is None:
        # Fallback per spec Section 5
        sources = spec.regime_outgoing_sources.get("stressed", set())
    return [v in sources for v in spec.core_variables]


def clamp_shock_magnitude(spec: ActionSpaceSpec,
                          target_var: str,
                          magnitude: float) -> float:
    """Apply the runtime VIX clip and global RL bounds to a proposed shock."""
    lo, hi = spec.rl_action_bounds
    m = max(lo, min(hi, float(magnitude)))
    if target_var == "^VIX":
        vlo, vhi = spec.rl_vix_special_caps
        m = max(vlo, min(vhi, m))
    return m


def family_template_as_dict(spec: ActionSpaceSpec, family_name: str) -> Dict[str, float]:
    """Returns a defensive copy of the shock template for the given family."""
    if family_name not in spec.shock_families:
        raise KeyError(f"Unknown shock family: {family_name}")
    return dict(spec.shock_families[family_name].shock_template)
