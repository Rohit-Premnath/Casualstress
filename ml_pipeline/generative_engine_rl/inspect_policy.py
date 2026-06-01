"""
inspect_policy.py
=================
Load a trained PPO policy and analyze what it learned.

Answers four questions about a trained agent:

    1. Action concentration: how spread out are its choices?
    2. Per-dimension preferences: which target_var, family, magnitude does it
       prefer over many independent eval episodes?
    3. Per-action reward: what reward does each top action actually generate?
    4. Stochastic vs deterministic behavior: how much does sampling differ
       from argmax?

USAGE
-----
By default, finds the most recent run in runs/ and inspects its final.zip:
    python -m generative_engine_rl.inspect_policy

To inspect a specific run:
    python -m generative_engine_rl.inspect_policy --run-dir runs/<name>

To inspect against the real env (slow, but tells you about real reward):
    python -m generative_engine_rl.inspect_policy --mode real

To run more sample episodes (default 100):
    python -m generative_engine_rl.inspect_policy --n-episodes 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # optional; only needed when running inspect_policy.py as a standalone script

from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.action_wrapper import build_magnitude_grid
from generative_engine_rl.env_factory import make_env


# ============================================================================
# RUN DISCOVERY
# ============================================================================

def find_latest_run(runs_dir: Path = Path("runs")) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"No '{runs_dir}' directory found. Run training first: "
            f"python -m generative_engine_rl.train_ppo"
        )
    candidates = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"'{runs_dir}' exists but contains no runs."
        )
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest


def find_model_path(run_dir: Path) -> Path:
    """Prefer 'final.zip', fall back to 'best/best_model.zip', then any ckpt."""
    candidates = [
        run_dir / "final.zip",
        run_dir / "best" / "best_model.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    ckpts = sorted(run_dir.glob("ckpt_*.zip"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No model file found in {run_dir}")


# ============================================================================
# SAMPLING
# ============================================================================

def sample_actions(
    model,
    env,
    n_episodes: int,
    deterministic: bool,
    seed_base: int = 1000,
) -> List[Dict[str, Any]]:
    """Run n_episodes deterministic (or stochastic) policy episodes and
    record decoded actions, rewards, and reward breakdowns.
    """
    samples = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + i)
        action, _ = model.predict(obs, deterministic=deterministic)
        decoded = env.decode_action(action)
        obs2, reward, term, trunc, step_info = env.step(action)
        br = step_info.get("reward_breakdown")
        samples.append({
            "target_var": decoded["target_var"],
            "family_name": decoded["family_name"],
            "magnitude": decoded["magnitude"],
            "reward": float(reward),
            "rejected": step_info.get("action_rejected", False),
            "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
            "dfast_breach": float(br.dfast_breach) if br else 0.0,
            "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
        })
    return samples


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_action_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-dimension Counters and concentration metrics."""
    targets = Counter(s["target_var"] for s in samples)
    families = Counter(s["family_name"] for s in samples)
    magnitudes = Counter(round(s["magnitude"], 2) for s in samples)

    # Joint action: (target, family, magnitude)
    joint = Counter(
        (s["target_var"], s["family_name"], round(s["magnitude"], 2))
        for s in samples
    )

    n = len(samples)
    return {
        "n_samples": n,
        "unique_targets": len(targets),
        "unique_families": len(families),
        "unique_magnitudes": len(magnitudes),
        "unique_joint_actions": len(joint),
        "top_target_share": targets.most_common(1)[0][1] / n if n else 0.0,
        "top_family_share": families.most_common(1)[0][1] / n if n else 0.0,
        "top_magnitude_share": magnitudes.most_common(1)[0][1] / n if n else 0.0,
        "top_joint_share": joint.most_common(1)[0][1] / n if n else 0.0,
        "targets": targets,
        "families": families,
        "magnitudes": magnitudes,
        "joint": joint,
    }


def per_action_reward_table(
    samples: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """For the top-k joint actions by frequency, compute reward statistics."""
    joint_groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for s in samples:
        key = (s["target_var"], s["family_name"], round(s["magnitude"], 2))
        joint_groups.setdefault(key, []).append(s)

    rows = []
    for key, group in joint_groups.items():
        rewards = np.asarray([g["reward"] for g in group], dtype=np.float64)
        pls = np.asarray([g["portfolio_loss"] for g in group], dtype=np.float64)
        dfs = np.asarray([g["dfast_breach"] for g in group], dtype=np.float64)
        cfs = np.asarray([g["causal_fidelity"] for g in group], dtype=np.float64)
        rows.append({
            "target_var": key[0],
            "family_name": key[1],
            "magnitude": key[2],
            "n_picks": len(group),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "portfolio_loss_mean": float(pls.mean()),
            "dfast_breach_mean": float(dfs.mean()),
            "causal_fidelity_mean": float(cfs.mean()),
        })
    rows.sort(key=lambda r: r["n_picks"], reverse=True)
    return rows[:top_k]


def concentration_verdict(action_dist: Dict[str, Any]) -> str:
    """Categorize how concentrated the policy is."""
    top_joint = action_dist["top_joint_share"]
    n_unique = action_dist["unique_joint_actions"]
    n_samples = action_dist["n_samples"]

    if top_joint >= 0.95:
        return f"FULLY CONVERGED — {top_joint:.0%} on a single action"
    elif top_joint >= 0.6:
        return f"STRONGLY CONCENTRATED — top action {top_joint:.0%}, {n_unique} actions used"
    elif top_joint >= 0.3:
        return f"MODE-FINDING — top action {top_joint:.0%}, {n_unique} actions used"
    elif n_unique < n_samples * 0.5:
        return f"PARTIALLY CONCENTRATED — {n_unique} unique actions across {n_samples} samples"
    else:
        return f"NEAR-RANDOM — {n_unique} unique actions across {n_samples} samples"


# ============================================================================
# DISPLAY
# ============================================================================

def print_inspection(
    run_dir: Path,
    model_path: Path,
    mode: str,
    n_episodes: int,
    det_samples: List[Dict[str, Any]],
    stoch_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    det_dist = analyze_action_distribution(det_samples)
    stoch_dist = analyze_action_distribution(stoch_samples)
    det_top_actions = per_action_reward_table(det_samples, top_k=5)
    stoch_top_actions = per_action_reward_table(stoch_samples, top_k=5)

    print()
    print("=" * 72)
    print("Trained Policy Inspection")
    print("=" * 72)
    print(f"  run_dir:    {run_dir}")
    print(f"  model:      {model_path.relative_to(run_dir)}")
    print(f"  mode:       {mode}")
    print(f"  n_episodes: {n_episodes} (each in deterministic + stochastic)")
    print()

    # --- Deterministic policy ---
    print("-" * 72)
    print("Deterministic policy (model.predict with deterministic=True)")
    print("-" * 72)
    print(f"  Verdict:        {concentration_verdict(det_dist)}")
    print(f"  Unique actions: {det_dist['unique_joint_actions']}")
    print()
    print(f"  Target preferences (top 5):")
    for var, n in det_dist["targets"].most_common(5):
        print(f"    {var:<20} {n:>4} ({n/det_dist['n_samples']:.0%})")
    print()
    print(f"  Family preferences (top 5):")
    for fam, n in det_dist["families"].most_common(5):
        print(f"    {fam:<24} {n:>4} ({n/det_dist['n_samples']:.0%})")
    print()
    print(f"  Magnitude preferences (top 5):")
    for mag, n in det_dist["magnitudes"].most_common(5):
        print(f"    {mag:>+6.2f}σ              {n:>4} ({n/det_dist['n_samples']:.0%})")
    print()

    print(f"  Top joint actions (target × family × magnitude):")
    print(f"    {'target':<20}{'family':<22}{'mag':>7}  {'count':>6}  {'reward μ':>9}  "
          f"{'P/L μ':>8}  {'DF μ':>6}  {'CF μ':>6}")
    print(f"    {'-'*20}{'-'*22}{'-'*7}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*6}")
    for r in det_top_actions:
        print(f"    {r['target_var']:<20}{r['family_name']:<22}"
              f"{r['magnitude']:>+6.2f}σ  "
              f"{r['n_picks']:>6}  "
              f"{r['reward_mean']:>+9.4f}  "
              f"{r['portfolio_loss_mean']:>+8.4f}  "
              f"{r['dfast_breach_mean']:>6.3f}  "
              f"{r['causal_fidelity_mean']:>6.3f}")
    print()

    # --- Stochastic policy ---
    print("-" * 72)
    print("Stochastic policy (model.predict with deterministic=False)")
    print("-" * 72)
    print(f"  Verdict:        {concentration_verdict(stoch_dist)}")
    print(f"  Unique actions: {stoch_dist['unique_joint_actions']}")
    print()
    print(f"  Top joint actions:")
    print(f"    {'target':<20}{'family':<22}{'mag':>7}  {'count':>6}  {'reward μ':>9}")
    print(f"    {'-'*20}{'-'*22}{'-'*7}  {'-'*6}  {'-'*9}")
    for r in stoch_top_actions:
        print(f"    {r['target_var']:<20}{r['family_name']:<22}"
              f"{r['magnitude']:>+6.2f}σ  "
              f"{r['n_picks']:>6}  "
              f"{r['reward_mean']:>+9.4f}")
    print()

    # --- Comparison summary ---
    rejected_det = sum(1 for s in det_samples if s["rejected"])
    rejected_stoch = sum(1 for s in stoch_samples if s["rejected"])
    det_rewards = np.asarray([s["reward"] for s in det_samples])
    stoch_rewards = np.asarray([s["reward"] for s in stoch_samples])

    print("-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"  Deterministic mean reward: {det_rewards.mean():+.4f} ± {det_rewards.std():.4f}")
    print(f"  Stochastic mean reward:    {stoch_rewards.mean():+.4f} ± {stoch_rewards.std():.4f}")
    print(f"  Rejected actions: {rejected_det} det / {rejected_stoch} stoch  (out of {n_episodes} each)")
    print()

    # --- Sanity flags ---
    flags = []
    if det_dist["unique_joint_actions"] == 1 and len(det_samples) >= 50:
        # Single deterministic action — fine, but flag what it picked
        top = det_top_actions[0]
        if top["reward_mean"] < 0.05:
            flags.append(
                f"Deterministic policy converged on a LOW-reward action "
                f"({top['reward_mean']:.4f}). Investigate reward shaping."
            )
        if top["causal_fidelity_mean"] < 0.2:
            flags.append(
                f"Deterministic action has LOW causal fidelity "
                f"({top['causal_fidelity_mean']:.3f}). The agent may have "
                f"found a non-causal exploit."
            )
        if abs(top["magnitude"]) < 0.5 and top["family_name"] != "rate_shock":
            flags.append(
                f"Deterministic action picked a SMALL magnitude ({top['magnitude']:+.2f}σ). "
                f"For an adversarial agent, small shocks usually indicate the reward "
                f"signal is weak or saturated."
            )

    if stoch_dist["unique_joint_actions"] == n_episodes:
        flags.append(
            f"Stochastic policy is essentially uniform ({stoch_dist['unique_joint_actions']} "
            f"unique out of {n_episodes}). Policy distribution hasn't sharpened. "
            f"Train longer or increase entropy coefficient pressure."
        )

    if rejected_det / max(n_episodes, 1) > 0.1:
        flags.append(
            f"More than 10% of deterministic actions REJECTED ({rejected_det}/{n_episodes}). "
            f"Agent is repeatedly picking causally-invalid targets — investigate."
        )

    if flags:
        print("-" * 72)
        print("Sanity flags")
        print("-" * 72)
        for f in flags:
            print(f"  ⚠  {f}")
        print()
    else:
        print("-" * 72)
        print("Sanity flags: none — policy looks reasonable")
        print("-" * 72)
        print()

    return {
        "deterministic": {
            "distribution": {
                "n_samples": det_dist["n_samples"],
                "unique_joint_actions": det_dist["unique_joint_actions"],
                "top_joint_share": det_dist["top_joint_share"],
                "verdict": concentration_verdict(det_dist),
                "targets": dict(det_dist["targets"]),
                "families": dict(det_dist["families"]),
                "magnitudes": {str(k): v for k, v in det_dist["magnitudes"].items()},
            },
            "top_actions": det_top_actions,
            "reward_mean": float(det_rewards.mean()),
            "reward_std": float(det_rewards.std()),
            "rejections": rejected_det,
        },
        "stochastic": {
            "distribution": {
                "n_samples": stoch_dist["n_samples"],
                "unique_joint_actions": stoch_dist["unique_joint_actions"],
                "top_joint_share": stoch_dist["top_joint_share"],
                "verdict": concentration_verdict(stoch_dist),
            },
            "top_actions": stoch_top_actions,
            "reward_mean": float(stoch_rewards.mean()),
            "reward_std": float(stoch_rewards.std()),
            "rejections": rejected_stoch,
        },
        "sanity_flags": flags,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None,
                        help="Specific run directory to inspect. Default: latest run.")
    parser.add_argument("--mode", choices=["fast", "real"], default=None,
                        help="Env mode for evaluation. Default: read from run config.")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--save-json", action="store_true",
                        help="Save the full inspection report to "
                             "<run_dir>/inspection.json")
    args = parser.parse_args()

    # Locate run
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
    model_path = find_model_path(run_dir)

    # Determine mode
    if args.mode is None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            mode = cfg.get("mode", "fast")
        else:
            mode = "fast"
    else:
        mode = args.mode

    # Load spec and env
    print(f"Loading spec...")
    spec = load_spec()
    print(f"Building inspection env (mode={mode})...")
    t0 = time.time()
    env = make_env(mode=mode, seed=args.seed_base, spec=spec)
    print(f"  env ready in {time.time() - t0:.1f}s")

    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(str(model_path))
    print(f"  loaded ({model.policy.__class__.__name__})")

    # Sample
    print(f"Running {args.n_episodes} deterministic episodes...")
    t0 = time.time()
    det_samples = sample_actions(
        model, env, args.n_episodes,
        deterministic=True, seed_base=args.seed_base,
    )
    print(f"  {time.time() - t0:.1f}s")

    print(f"Running {args.n_episodes} stochastic episodes...")
    t0 = time.time()
    stoch_samples = sample_actions(
        model, env, args.n_episodes,
        deterministic=False, seed_base=args.seed_base + 50_000,
    )
    print(f"  {time.time() - t0:.1f}s")

    report = print_inspection(
        run_dir=run_dir,
        model_path=model_path,
        mode=mode,
        n_episodes=args.n_episodes,
        det_samples=det_samples,
        stoch_samples=stoch_samples,
    )

    if args.save_json:
        out = run_dir / "inspection.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"  Saved full inspection to {out}")


if __name__ == "__main__":
    main()