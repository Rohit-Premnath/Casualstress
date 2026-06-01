# Action Space Specification

This folder contains the canonical specification of the CausalStress
scenario-generation action space — the complete set of valid shocks,
guardrails, and causal-plausibility constraints that any scenario generator
(canonical or adversarial) must respect.

## Files

| File | Purpose |
|---|---|
| `action_space.yaml` | Single source of truth for the action space spec |
| `verify_action_space.py` | Asserts spec ↔ codebase consistency. Run before any commit that touches this folder or `scenario_generator.py` |
| `README.md` | This file |

## Why this exists

Before this folder, the action space lived as Python constants scattered
across `scenario_generator.py`:

- `EVENT_SHOCK_TEMPLATES` (8 event families × shock magnitudes)
- `EVENT_VARIABLE_GUARDRAILS` (4 families × per-variable caps)
- `CORE_VARIABLES`, `LOG_RETURN_VARS`, `FIRST_DIFF_VARS`
- `VIX_TEMPLATE_CAP`, `VIX_PROPAGATION_SCALE`, `MAX_VAR_VARIABLES`

These constants describe the action space implicitly. To build the Phase 2B
RL adversarial scenario generator, the action space needs to be specified
**explicitly and formally** so the agent's action mask can be derived from
the same source as the canonical generator's templates.

This file makes the implicit explicit, while the verifier ensures the two
representations cannot drift apart.

## Two consumers

1. **The existing canonical multi-root generator**
   (`generative_engine/scenario_generator.py`) — uses the Python constants
   directly today. Eventually will be migrated to load from this YAML, but
   that migration is NOT part of Phase 1. Until then, the YAML is
   *descriptive* of what's already in code.

2. **The Phase 2B RL adversarial generator**
   (`generative_engine_rl/` — to be created in Phase 2B) — will load this
   YAML at agent init time and use it to:
   - Define the discrete `target_var` action dimension (Section 6)
   - Define the continuous `shock_magnitude` bounds (Section 6)
   - Apply the causal-plausibility action mask (Section 5)
   - Compose the multi-component reward (Section 7)
   - Initialize episodes from event-family templates (Section 3)

## Schema sections

The spec has eight sections:

1. Variable taxonomy — which variables are shockable, transform classes
2. Action units — sigma units, coordinate system
3. Shock-type taxonomy — 8 event families with deterministic templates
4. Per-variable per-family guardrails — post-hoc safety caps
5. Causal plausibility constraint — the RL action mask rule
6. RL agent action space — Phase 2B forward-looking definition
7. Reward signal — Phase 2B reward function components
8. Schema invariants — what the verifier asserts

## Running the verifier

```bash
cd ml_pipeline
python3 action_space/verify_action_space.py
```

Exit code 0 = all 12 invariants hold. Exit code 1 = drift detected, see
stderr for the specific invariant(s) violated.

The verifier checks the spec against three external sources:

- `generative_engine/scenario_generator.py` — runtime constants
- `regime_causal_graphs.json` — causal graph structure
- (implicit) `canonical_best_model.py` — VIX cap consistency

## When to run the verifier

- Before any commit that modifies `action_space.yaml`
- Before any commit that modifies `scenario_generator.py` constants
- Before any commit that modifies `regime_causal_graphs.json`
- As part of CI on every PR to `ml_pipeline/`

## Maintenance discipline

If `scenario_generator.py` is modified to add a new event family, change a
shock magnitude, or alter a guardrail:

1. Make the change in `scenario_generator.py`
2. Make the corresponding change in `action_space.yaml`
3. Run `python3 action_space/verify_action_space.py`
4. If green, commit both files together

If you commit one without the other, the verifier will catch it on the next
run. Do not ignore verifier failures — they indicate the spec and runtime
are no longer aligned, which means the Phase 2B RL agent will be operating
on a stale view of the action space.
