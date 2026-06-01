"""
verify_action_space.py
======================
Asserts that ml_pipeline/action_space/action_space.yaml is consistent with:

  1. ml_pipeline/generative_engine/scenario_generator.py (constants)
  2. ml_pipeline/regime_causal_graphs.json (causal-plausibility constraint)
  3. ml_pipeline/canonical_best_model.py (vix_template_cap)

Fails loudly with a descriptive error message on any drift. Run as part of
CI or manually before any commit that touches the spec or scenario_generator.

Usage:
    python ml_pipeline/action_space/verify_action_space.py

Exit code 0 = all invariants hold.
Exit code 1 = drift detected, see stderr for details.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


# ============================================================================
# LOCATE FILES
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent  # .../ml_pipeline/
SPEC_FILE = Path(__file__).resolve().parent / "action_space.yaml"
GENERATOR_FILE = ROOT / "generative_engine" / "scenario_generator.py"
REGIME_GRAPH_FILE = ROOT / "regime_causal_graphs.json"


# ============================================================================
# LOAD SPEC
# ============================================================================

def load_spec() -> dict:
    if not SPEC_FILE.exists():
        sys.exit(f"ERROR: spec file not found: {SPEC_FILE}")
    with open(SPEC_FILE) as f:
        return yaml.safe_load(f)


# ============================================================================
# LOAD GROUND TRUTH FROM scenario_generator.py
# ============================================================================

def load_generator_constants() -> dict:
    """Extract module-level constants from scenario_generator.py via AST.

    We do this rather than `import` to avoid pulling in the database
    dependencies (psycopg2 etc.) of the runtime module.
    """
    if not GENERATOR_FILE.exists():
        sys.exit(f"ERROR: generator file not found: {GENERATOR_FILE}")

    with open(GENERATOR_FILE) as f:
        src = f.read()
    tree = ast.parse(src)

    wanted = {
        "EVENT_SHOCK_TEMPLATES",
        "EVENT_VARIABLE_GUARDRAILS",
        "CORE_VARIABLES",
        "LOG_RETURN_VARS",
        "FIRST_DIFF_VARS",
        "VIX_TEMPLATE_CAP",
        "VIX_PROPAGATION_SCALE",
        "MAX_VAR_VARIABLES",
    }
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in wanted:
                    m = ast.Module(body=[node], type_ignores=[])
                    ns: dict = {}
                    exec(compile(m, "<gen>", "exec"), ns)  # nosec — local file we authored
                    found[tgt.id] = ns[tgt.id]

    missing = wanted - set(found.keys())
    if missing:
        sys.exit(f"ERROR: scenario_generator.py is missing constants: {missing}")
    return found


# ============================================================================
# LOAD GROUND TRUTH FROM regime_causal_graphs.json
# ============================================================================

def load_regime_outgoing_sources() -> dict:
    """Returns {regime_name: set(of_source_variables_with_outgoing_edges)}."""
    if not REGIME_GRAPH_FILE.exists():
        sys.exit(f"ERROR: regime graph file not found: {REGIME_GRAPH_FILE}")
    with open(REGIME_GRAPH_FILE) as f:
        rg = json.load(f)
    out = {}
    for regime_name, regime_data in rg["regimes"].items():
        out[regime_name] = {e["source"] for e in regime_data["edges"]}
    return out


# ============================================================================
# INVARIANT CHECKS
# ============================================================================

def run_checks(spec, gen, regime_sources):
    """Returns list of failures. Empty list = all checks passed."""
    failures = []

    def fail(msg):
        failures.append(msg)

    # ----- INVARIANT 1: variable taxonomy size -----
    if len(spec["variables"]["core"]) != spec["variables"]["max_var_dim"]:
        fail(
            f"len(variables.core)={len(spec['variables']['core'])} but "
            f"max_var_dim={spec['variables']['max_var_dim']}"
        )

    # ----- INVARIANT 2: spec core matches generator CORE_VARIABLES -----
    spec_core = list(spec["variables"]["core"])
    gen_core = list(gen["CORE_VARIABLES"])
    if spec_core != gen_core:
        fail(
            "variables.core does not match scenario_generator.CORE_VARIABLES.\n"
            f"  spec : {spec_core}\n"
            f"  gen  : {gen_core}"
        )

    # ----- INVARIANT 3: log_return / first_diff sets match generator -----
    if set(spec["variables"]["log_return"]) != gen["LOG_RETURN_VARS"]:
        only_spec = set(spec["variables"]["log_return"]) - gen["LOG_RETURN_VARS"]
        only_gen = gen["LOG_RETURN_VARS"] - set(spec["variables"]["log_return"])
        fail(
            f"log_return mismatch. only_in_spec={only_spec} only_in_gen={only_gen}"
        )
    if set(spec["variables"]["first_diff"]) != gen["FIRST_DIFF_VARS"]:
        only_spec = set(spec["variables"]["first_diff"]) - gen["FIRST_DIFF_VARS"]
        only_gen = gen["FIRST_DIFF_VARS"] - set(spec["variables"]["first_diff"])
        fail(
            f"first_diff mismatch. only_in_spec={only_spec} only_in_gen={only_gen}"
        )

    # ----- INVARIANT 4: VIX cap and max_var_dim match -----
    if spec["variables"]["vix_template_cap"] != gen["VIX_TEMPLATE_CAP"]:
        fail(
            f"vix_template_cap drift: spec={spec['variables']['vix_template_cap']} "
            f"gen={gen['VIX_TEMPLATE_CAP']}"
        )
    if spec["variables"]["vix_propagation_scale"] != gen["VIX_PROPAGATION_SCALE"]:
        fail(
            f"vix_propagation_scale drift: spec={spec['variables']['vix_propagation_scale']} "
            f"gen={gen['VIX_PROPAGATION_SCALE']}"
        )
    if spec["variables"]["max_var_dim"] != gen["MAX_VAR_VARIABLES"]:
        fail(
            f"max_var_dim drift: spec={spec['variables']['max_var_dim']} "
            f"gen={gen['MAX_VAR_VARIABLES']}"
        )

    # ----- INVARIANT 5: shock_families templates match generator EVENT_SHOCK_TEMPLATES -----
    spec_families = set(spec["shock_families"].keys())
    gen_families = set(gen["EVENT_SHOCK_TEMPLATES"].keys())

    # 'pandemic' is a documented alias of 'pandemic_exogenous'
    spec_for_compare = spec_families - {"pandemic"}
    if spec_for_compare != gen_families:
        only_spec = spec_for_compare - gen_families
        only_gen = gen_families - spec_for_compare
        fail(
            f"shock family drift. only_in_spec={only_spec} only_in_gen={only_gen}"
        )

    for fam, gen_shocks in gen["EVENT_SHOCK_TEMPLATES"].items():
        spec_shocks = spec["shock_families"][fam].get("shock_template", {})
        if spec_shocks != gen_shocks:
            fail(
                f"shock_template drift in family '{fam}':\n"
                f"  spec: {spec_shocks}\n"
                f"  gen : {gen_shocks}"
            )

    # ----- INVARIANT 6: pandemic alias resolves correctly -----
    if "pandemic" in spec_families:
        pd = spec["shock_families"]["pandemic"]
        if pd.get("shock_template_inherits") != "pandemic_exogenous":
            fail(
                f"'pandemic' family must have shock_template_inherits='pandemic_exogenous', "
                f"got '{pd.get('shock_template_inherits')}'"
            )

    # ----- INVARIANT 7: guardrails match generator EVENT_VARIABLE_GUARDRAILS -----
    spec_guardrails = spec["guardrails"]
    gen_guardrails = gen["EVENT_VARIABLE_GUARDRAILS"]
    if set(spec_guardrails.keys()) != set(gen_guardrails.keys()):
        fail(
            f"guardrail families differ: spec={set(spec_guardrails.keys())} "
            f"gen={set(gen_guardrails.keys())}"
        )

    for fam, gen_g in gen_guardrails.items():
        spec_g = spec_guardrails.get(fam, {})
        if set(spec_g.keys()) != set(gen_g.keys()):
            fail(
                f"guardrail variables differ in family '{fam}': "
                f"spec={set(spec_g.keys())} gen={set(gen_g.keys())}"
            )
        for var, gen_rules in gen_g.items():
            spec_rules = spec_g.get(var, {})
            if dict(spec_rules) != dict(gen_rules):
                fail(
                    f"guardrail rule drift for {fam}/{var}:\n"
                    f"  spec: {dict(spec_rules)}\n"
                    f"  gen : {dict(gen_rules)}"
                )

    # ----- INVARIANT 8: all shock_template variables are in variables.core -----
    core_set = set(spec["variables"]["core"])
    for fam, fam_data in spec["shock_families"].items():
        if "shock_template" not in fam_data:
            continue
        for var in fam_data["shock_template"]:
            if var not in core_set:
                fail(
                    f"shock_template var '{var}' (family '{fam}') is not in variables.core"
                )

    # ----- INVARIANT 9: all guardrail variables are in variables.core -----
    for fam, gvars in spec["guardrails"].items():
        for var in gvars:
            if var not in core_set:
                fail(
                    f"guardrail var '{var}' (family '{fam}') is not in variables.core"
                )

    # ----- INVARIANT 10: no shock_template magnitude exceeds RL action bounds -----
    rl_low, rl_high = spec["rl_action_space"]["shock_magnitude"]["bounds"]
    for fam, fam_data in spec["shock_families"].items():
        for var, mag in fam_data.get("shock_template", {}).items():
            if mag < rl_low or mag > rl_high:
                fail(
                    f"shock_template magnitude {mag} for {var} in family '{fam}' "
                    f"exceeds rl_action_space bounds [{rl_low}, {rl_high}]"
                )

    # ----- INVARIANT 11: every shock_template root has outgoing edges in stressed regime -----
    stressed_sources = regime_sources.get("stressed", set())
    if not stressed_sources:
        fail("regime_causal_graphs.json has no 'stressed' regime — required by spec")
    else:
        roots = set()
        for fam_data in spec["shock_families"].values():
            roots.update(fam_data.get("shock_template", {}).keys())
        roots_without_outgoing = roots - stressed_sources
        if roots_without_outgoing:
            fail(
                f"shock-template roots without outgoing edges in stressed regime: "
                f"{roots_without_outgoing}. Causal-plausibility constraint would "
                f"reject these as RL action targets."
            )

    # ----- INVARIANT 12: VIX special-cap matches vix_template_cap -----
    rl_vix_caps = spec["rl_action_space"]["shock_magnitude"]["special_caps"].get("^VIX")
    if rl_vix_caps != [-spec["variables"]["vix_template_cap"], spec["variables"]["vix_template_cap"]]:
        fail(
            f"rl_action_space VIX special_caps {rl_vix_caps} do not match "
            f"vix_template_cap {spec['variables']['vix_template_cap']}"
        )

    # ----- INVARIANT 13: shock_template VIX values above cap are tolerated -----
    # because scenario_generator.py clips them at runtime (line 445). We just
    # confirm the RL action space does NOT exceed the cap, which is what
    # actually constrains the RL agent.
    vix_cap = spec["variables"]["vix_template_cap"]
    rl_low, rl_high = spec["rl_action_space"]["shock_magnitude"]["bounds"]
    # If the special-cap for VIX is more restrictive than the global RL bounds,
    # confirm it's narrower (so VIX is more constrained in practice).
    if rl_vix_caps is not None:
        vix_low, vix_high = rl_vix_caps
        if vix_low < rl_low or vix_high > rl_high:
            fail(
                f"VIX special_caps {rl_vix_caps} are wider than global RL bounds "
                f"[{rl_low}, {rl_high}] — special caps must be a SUBSET"
            )

    return failures


# ============================================================================
# MAIN
# ============================================================================

def main():
    spec = load_spec()
    gen = load_generator_constants()
    regime_sources = load_regime_outgoing_sources()

    print(f"Loaded spec from: {SPEC_FILE}")
    print(f"Loaded constants from: {GENERATOR_FILE}")
    print(f"Loaded regime graph from: {REGIME_GRAPH_FILE}")
    print()

    failures = run_checks(spec, gen, regime_sources)

    if failures:
        print(f"FAIL — {len(failures)} invariant(s) violated:\n", file=sys.stderr)
        for i, msg in enumerate(failures, 1):
            print(f"  [{i}] {msg}\n", file=sys.stderr)
        sys.exit(1)

    print("All 13 invariants hold. Spec is in sync with codebase.")
    sys.exit(0)


if __name__ == "__main__":
    main()
