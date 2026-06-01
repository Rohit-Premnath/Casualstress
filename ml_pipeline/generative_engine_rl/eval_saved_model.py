"""
eval_saved_model.py
===================
Evaluate a saved PPO model against a random baseline and print a full
breakdown of reward components. Writes eval_results.json into the same
run directory as the model.

Usage (from ml_pipeline/):
    python -m generative_engine_rl.eval_saved_model \\
        --model runs/ppo_fixed_v1_20260510_021405/final.zip \\
        --n-episodes 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.action_space_loader import load_spec


# ============================================================================
# HELPERS
# ============================================================================

def evaluate_policy_episodes(
    model, env, n_episodes: int, deterministic: bool = True
) -> Dict[str, Any]:
    rewards: List[float] = []
    breakdowns: List[Dict[str, float]] = []
    action_log: List[Dict[str, Any]] = []
    rejections = 0
    obs = env.reset()
    for ep in range(n_episodes):
        done_flag = False
        episode_reward = 0.0
        final_info: Dict[str, Any] = {}
        while not done_flag:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info_list = env.step(action)
            info = info_list[0] if isinstance(info_list, (list, tuple)) else info_list
            episode_reward += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            done_flag = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            final_info = info if isinstance(info, dict) else {}
            if final_info.get("action_rejected"):
                rejections += 1
        rewards.append(float(episode_reward))
        if "reward_breakdown" in final_info:
            br = final_info["reward_breakdown"]
            breakdowns.append({
                "portfolio_loss": float(br.portfolio_loss),
                "dfast_breach": float(br.dfast_breach),
                "causal_fidelity": float(br.causal_fidelity),
                "diversity": float(br.diversity),
                "total": float(br.total),
            })
        if "last_action" in final_info and ep < 20:
            action_log.append({
                "ep": ep,
                "target": final_info.get("last_action", {}).get("shock_variable", "?"),
                "family": final_info.get("last_action", {}).get("event_type", "?"),
                "magnitude": final_info.get("last_action", {}).get("shock_magnitude", 0.0),
                "portfolio_loss": float(breakdowns[-1]["portfolio_loss"]) if breakdowns else 0.0,
                "dfast_breach": float(breakdowns[-1]["dfast_breach"]) if breakdowns else 0.0,
                "total": float(rewards[-1]),
            })
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return {
        "n_episodes": n_episodes,
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std()),
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "rejections": rejections,
        "breakdowns": breakdowns,
        "action_log": action_log,
    }


def run_random_baseline(eval_env, n_episodes: int, seed: int = 9999) -> Dict[str, Any]:
    spec = load_spec()
    rng = np.random.default_rng(seed)
    nvec = eval_env.action_space.nvec
    n_targets, n_families, n_mag = int(nvec[0]), int(nvec[1]), int(nvec[2])
    rewards: List[float] = []
    breakdowns: List[Dict[str, float]] = []
    obs = eval_env.reset()
    for _ in range(n_episodes):
        done_flag = False
        episode_reward = 0.0
        final_info: Dict[str, Any] = {}
        while not done_flag:
            a = np.array([[rng.integers(n_targets), rng.integers(n_families), rng.integers(n_mag)]])
            obs, reward, done, info_list = eval_env.step(a)
            info = info_list[0] if isinstance(info_list, (list, tuple)) else info_list
            episode_reward += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            done_flag = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            final_info = info if isinstance(info, dict) else {}
        rewards.append(float(episode_reward))
        if "reward_breakdown" in final_info:
            br = final_info["reward_breakdown"]
            breakdowns.append({
                "portfolio_loss": float(br.portfolio_loss),
                "dfast_breach": float(br.dfast_breach),
                "causal_fidelity": float(br.causal_fidelity),
                "diversity": float(br.diversity),
                "total": float(br.total),
            })
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return {
        "n_episodes": n_episodes,
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std()),
        "min": float(rewards_arr.min()),
        "max": float(rewards_arr.max()),
        "breakdowns": breakdowns,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to final.zip")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--mode", default="real", choices=["fast", "real"])
    parser.add_argument("--portfolio-profile", default="balanced")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--no-family-templates", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    run_dir = model_path.parent
    use_family_templates = not args.no_family_templates

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"  mode:      {args.mode}")
    print(f"  episodes:  {args.n_episodes}")
    print(f"  portfolio: {args.portfolio_profile}")
    print(f"  templates: {use_family_templates}")
    print(f"{'='*60}\n")

    # Build eval env (use a different seed than training to avoid overfitting)
    eval_env_inner = make_env(
        mode=args.mode,
        seed=args.seed,
        portfolio_profile=args.portfolio_profile,
        actions_per_episode=1,
        use_family_templates=use_family_templates,
        reward_mode="portfolio_adversarial",
    )
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_inner)])

    # Load model
    print("Loading model...")
    model = PPO.load(str(model_path), env=eval_env)
    print(f"  action space: {model.action_space}")

    # Random baseline
    print("\nRunning random baseline...")
    random_results = run_random_baseline(eval_env, n_episodes=args.n_episodes, seed=args.seed + 9999)
    print(f"  mean: {random_results['mean']:+.4f} ± {random_results['std']:.4f}")
    print(f"  range: [{random_results['min']:+.4f}, {random_results['max']:+.4f}]")

    if random_results["breakdowns"]:
        pl_vals = [b["portfolio_loss"] for b in random_results["breakdowns"]]
        df_vals = [b["dfast_breach"] for b in random_results["breakdowns"]]
        cf_vals = [b["causal_fidelity"] for b in random_results["breakdowns"]]
        print(f"  portfolio_loss:  mean={np.mean(pl_vals):+.4f}  nonzero={np.mean(np.array(pl_vals) > 0)*100:.1f}%")
        print(f"  dfast_breach:    mean={np.mean(df_vals):+.4f}")
        print(f"  causal_fidelity: mean={np.mean(cf_vals):+.4f}")

    # Trained policy (deterministic)
    print("\nRunning trained policy (deterministic)...")
    trained_results = evaluate_policy_episodes(model, eval_env, n_episodes=args.n_episodes, deterministic=True)
    print(f"  mean: {trained_results['mean']:+.4f} ± {trained_results['std']:.4f}")
    print(f"  range: [{trained_results['min']:+.4f}, {trained_results['max']:+.4f}]")
    print(f"  rejected actions: {trained_results['rejections']}/{trained_results['n_episodes']}")

    if trained_results["breakdowns"]:
        pl_vals = [b["portfolio_loss"] for b in trained_results["breakdowns"]]
        df_vals = [b["dfast_breach"] for b in trained_results["breakdowns"]]
        cf_vals = [b["causal_fidelity"] for b in trained_results["breakdowns"]]
        dv_vals = [b["diversity"] for b in trained_results["breakdowns"]]
        print(f"\n  Reward component breakdown (trained policy):")
        print(f"    portfolio_loss:  mean={np.mean(pl_vals):+.4f}  nonzero={np.mean(np.array(pl_vals) > 0)*100:.1f}%  max={np.max(pl_vals):+.4f}")
        print(f"    dfast_breach:    mean={np.mean(df_vals):+.4f}  (logged only, not in reward)")
        print(f"    causal_fidelity: mean={np.mean(cf_vals):+.4f}")
        print(f"    diversity:       mean={np.mean(dv_vals):+.4f}")

    # Verdict
    improvement = trained_results["mean"] - random_results["mean"]
    random_floor = random_results["mean"] + 0.5 * max(random_results["std"], 1e-6)
    beat_floor = trained_results["mean"] > random_floor
    converged = trained_results["std"] < random_results["std"] * 0.1

    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    print(f"  improvement over random:  {improvement:+.4f}")
    print(f"  beats 0.5s floor:         {beat_floor}")
    print(f"  policy converged (std):   {converged}  ({trained_results['std']:.4f} vs {random_results['std']:.4f})")

    if beat_floor or converged:
        reason = []
        if beat_floor:
            reason.append(f"eval mean beats random by >0.5σ (+{improvement:.4f})")
        if converged:
            reason.append(f"policy std collapsed ({trained_results['std']:.4f} << {random_results['std']:.4f})")
        print(f"  SMOKE TEST: PASS  —  {' AND '.join(reason)}")
    else:
        print(f"  SMOKE TEST: FAIL  —  improvement={improvement:+.4f} did not clear 0.5σ floor ({random_floor:.4f})")

    # Save results
    results = {
        "model_path": str(model_path),
        "random_baseline": {k: v for k, v in random_results.items() if k != "breakdowns"},
        "trained_policy": {k: v for k, v in trained_results.items() if k not in ("breakdowns", "action_log")},
        "improvement_over_random": improvement,
        "beat_random_floor": beat_floor,
        "policy_converged": converged,
        "smoke_test_passed": beat_floor or converged,
        "random_breakdowns_sample": random_results["breakdowns"][:10],
        "trained_breakdowns_sample": trained_results["breakdowns"][:10],
    }
    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
