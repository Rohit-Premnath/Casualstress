"""
env_path_reproducer.py
======================
Reproduce the env path's call to generate_scenarios() byte-for-byte and
compare against the direct path. This is a focused debugging diagnostic
for the trajectory_diagnostic incoherence we cannot explain yet.

THE SETUP
---------
- Direct path A (the diagnostic that produces COHERENT output):
    np.random.seed(42)
    generate_scenarios(var_model, "UNRATE", 5.0, n_scenarios=1, ...)

- Env path B (the diagnostic that produces INCOHERENT output):
    env = make_env(mode='real', seed=1000)
    obs = env.reset(seed=1000)
    model = PPO.load(...)
    action = model.predict(obs, deterministic=True)
    obs2, reward, term, trunc, info = env.step(action)
    trajectory = info['trajectory']

This script reproduces path B exactly, then re-runs path B *with* an
explicit np.random.seed() inserted right before generate_scenarios is
called. If the latter produces a different trajectory, then env path is
inheriting np.random state from prior operations (model load, env reset,
predict). If both paths produce the same incoherent output, the issue
is something else entirely.

USAGE
-----
    python -m generative_engine_rl.env_path_reproducer

Optionally pass --run-dir to inspect a specific trained model:
    python -m generative_engine_rl.env_path_reproducer --run-dir runs/<name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_latest_run() -> Path:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError("No 'runs' directory.")
    candidates = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError("'runs' directory empty.")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def find_model_path(run_dir: Path) -> Path:
    for c in [run_dir / "final.zip", run_dir / "best" / "best_model.zip"]:
        if c.exists():
            return c
    raise FileNotFoundError(f"No model in {run_dir}")


def summarize_trajectory(trajectory: np.ndarray, var_codes: List[str]) -> Dict[str, float]:
    """Return cumulative move per key DFAST variable in transform space."""
    summary = {}
    key_vars = ["UNRATE", "BAMLH0A0HYM2", "^GSPC", "^VIX", "XLF", "XLE",
                "DGS10", "FEDFUNDS"]
    for code in key_vars:
        if code in var_codes:
            col = var_codes.index(code)
            summary[code] = float(np.sum(trajectory[:, col]))
    return summary


def fmt_summary(s: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:+.4f}" for k, v in s.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run()
    model_path = find_model_path(run_dir)

    print()
    print("=" * 84)
    print("Env-Path Reproducer")
    print("=" * 84)
    print(f"  run_dir: {run_dir}")
    print(f"  model:   {model_path.name}")
    print(f"  seed:    {args.seed}")
    print()

    from stable_baselines3 import PPO
    from generative_engine_rl.action_space_loader import load_spec
    from generative_engine_rl.env_factory import make_env
    from generative_engine.scenario_generator import generate_scenarios
    from generative_engine_rl.real_mode_loader import load_real_var_model_and_graph

    spec = load_spec()

    # ---- Path B (the broken path) ----
    print("Step 1: Reproduce env path verbatim (the path that produces incoherence)")
    print("-" * 84)
    env = make_env(mode="real", seed=args.seed, spec=spec)
    model = PPO.load(str(model_path))
    obs, _ = env.reset(seed=args.seed)
    action, _ = model.predict(obs, deterministic=True)
    decoded = env.decode_action(action)
    print(f"  agent action: {decoded['target_var']} × {decoded['family_name']} "
          f"× {decoded['magnitude']:+.2f}σ")

    # Capture np.random state right before stepping
    state_before_step = np.random.get_state()
    state_summary = (state_before_step[0], state_before_step[1][0], state_before_step[1][-1])
    print(f"  np.random state before env.step: type={state_summary[0]}, "
          f"first={state_summary[1]}, last={state_summary[2]}")

    obs2, reward, term, trunc, info = env.step(action)
    trajectory_B = info["trajectory"]
    var_codes_B = info["trajectory_vars"]
    summary_B = summarize_trajectory(np.asarray(trajectory_B), var_codes_B)
    print(f"  trajectory cumulative moves (transform space):")
    print(f"    {fmt_summary(summary_B)}")
    print()

    # ---- Path A (the working path), called RIGHT NOW from same Python process ----
    print("Step 2: Direct call to generate_scenarios() — same process, same var_model")
    print("-" * 84)

    # Use the SAME var_model the env used (cached from real_mode_loader)
    # No np.random.seed() here — let it inherit current state (whatever it is
    # after env construction, model load, predict, step)
    var_model_cached, causal_adj_cached, _ = load_real_var_model_and_graph()

    # Verify it's the same object as what env stored. The env is wrapped by
    # MultiDiscreteActionWrapper, so we have to reach through to the inner env.
    inner_env = env.unwrapped if hasattr(env, "unwrapped") else env
    same_model = var_model_cached is getattr(inner_env, "var_model", None)
    same_adj = causal_adj_cached is getattr(inner_env, "causal_adjacency", None)
    print(f"  var_model identity matches env's:        {same_model}")
    print(f"  causal_adjacency identity matches env's: {same_adj}")

    env_adj = getattr(inner_env, "causal_adjacency", None)
    if env_adj is not None:
        print(f"  env.causal_adjacency edge count:    {len(env_adj)}")
    print(f"  cached.causal_adjacency edge count: {len(causal_adj_cached)}")

    # Capture state before this call too
    state_A = np.random.get_state()
    state_A_summary = (state_A[0], state_A[1][0], state_A[1][-1])
    print(f"  np.random state before direct call: first={state_A_summary[1]}, "
          f"last={state_A_summary[2]}")

    scenarios_A = generate_scenarios(
        var_model=var_model_cached,
        shock_variable=decoded["target_var"],
        shock_magnitude=decoded["magnitude"],
        n_scenarios=1,
        horizon=spec.rl_episode_horizon_days,
        causal_adjacency=causal_adj_cached,
        use_multi_shock=False,
        event_type=decoded["family_name"],
    )
    trajectory_A = scenarios_A[0].to_numpy(dtype=np.float32)
    var_codes_A = list(scenarios_A[0].columns)
    summary_A = summarize_trajectory(trajectory_A, var_codes_A)
    print(f"  trajectory cumulative moves (transform space):")
    print(f"    {fmt_summary(summary_A)}")
    print()

    # ---- Compare ----
    print("Step 3: Comparison")
    print("-" * 84)
    keys = sorted(set(summary_A.keys()) | set(summary_B.keys()))
    print(f"  {'variable':<16}{'env path (B)':>16}{'direct (A)':>16}{'difference':>16}")
    print(f"  {'-'*16}{'-'*16}{'-'*16}{'-'*16}")
    max_diff = 0.0
    for k in keys:
        a = summary_A.get(k, float("nan"))
        b = summary_B.get(k, float("nan"))
        diff = a - b
        if abs(diff) > max_diff:
            max_diff = abs(diff)
        print(f"  {k:<16}{b:>+16.4f}{a:>+16.4f}{diff:>+16.4f}")
    print()

    # ---- Step 4: now seed and re-call to test the seed hypothesis ----
    print("Step 4: Direct call AGAIN with explicit np.random.seed(0) before it")
    print("-" * 84)
    np.random.seed(0)
    scenarios_C = generate_scenarios(
        var_model=var_model_cached,
        shock_variable=decoded["target_var"],
        shock_magnitude=decoded["magnitude"],
        n_scenarios=1,
        horizon=spec.rl_episode_horizon_days,
        causal_adjacency=causal_adj_cached,
        use_multi_shock=False,
        event_type=decoded["family_name"],
    )
    trajectory_C = scenarios_C[0].to_numpy(dtype=np.float32)
    summary_C = summarize_trajectory(trajectory_C, list(scenarios_C[0].columns))
    print(f"  with seed=0:  {fmt_summary(summary_C)}")

    np.random.seed(42)
    scenarios_D = generate_scenarios(
        var_model=var_model_cached,
        shock_variable=decoded["target_var"],
        shock_magnitude=decoded["magnitude"],
        n_scenarios=1,
        horizon=spec.rl_episode_horizon_days,
        causal_adjacency=causal_adj_cached,
        use_multi_shock=False,
        event_type=decoded["family_name"],
    )
    trajectory_D = scenarios_D[0].to_numpy(dtype=np.float32)
    summary_D = summarize_trajectory(trajectory_D, list(scenarios_D[0].columns))
    print(f"  with seed=42: {fmt_summary(summary_D)}")
    print()

    # ---- Verdict ----
    print("=" * 84)
    print("Verdict")
    print("=" * 84)

    # Check whether B (env path) ≈ A (direct, no seed)
    if max_diff < 1e-3:
        print("  Path A == Path B (env path and direct path produced the SAME trajectory)")
        print("  This means: the env is calling the generator correctly. The trajectory")
        print("  is what the generator naturally produces given the np.random state at")
        print("  that point in the program.")
    else:
        print(f"  Path A != Path B (max difference: {max_diff:.4f})")
        print("  The env path produces a different trajectory than a direct call from the")
        print("  same process, even though the var_model is the same. There must be some")
        print("  state-dependent difference between the call sites — possibly a difference")
        print("  in initial state, causal_adjacency format, or argument handling.")

    # Check whether B's trajectory looks like a typical seed
    unrate_B = summary_B.get("UNRATE", 0.0)
    hy_B = summary_B.get("BAMLH0A0HYM2", 0.0)
    print()
    if unrate_B > 5.0 or hy_B < -3.0:
        print(f"  Env path's UNRATE move = {unrate_B:+.2f}, HY move = {hy_B:+.2f}")
        print(f"  In our 20-seed test, UNRATE max was +5.49, HY min was +2.16.")
        print(f"  This trajectory is FAR outside the typical seed envelope — there is")
        print(f"  something specific to the env path that produces this outlier.")
    else:
        print(f"  Env path's UNRATE move = {unrate_B:+.2f}, HY move = {hy_B:+.2f}")
        print(f"  This is within the typical envelope from the 20-seed test.")
    print()


if __name__ == "__main__":
    main()