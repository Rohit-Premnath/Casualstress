"""
baseline_compare.py
===================
Compare a trained RL policy against simple one-shot baselines on the same
real-mode environment seeds.

Baselines:
    1. RL deterministic policy
    2. Random single action
    3. Hand-picked heuristic shocks
    4. Brute-force best single action over the full one-step action grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.inspect_policy import find_latest_run, find_model_path
from generative_engine_rl.portfolio_model import DEFAULT_PORTFOLIO_PROFILE


def _base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def find_action(env, target_var: str, magnitude: float, family_idx: int = 0):
    """Find the MultiDiscrete action encoding a (var, magnitude) pair.

    Matches on target_var and magnitude sign/abs; uses family_idx=0 by default.
    Returns None if the variable is not in the env's valid target set.
    """
    n_targets, n_families, n_mag_bins = (int(x) for x in env.action_space.nvec)
    for ti in range(n_targets):
        for mi in range(n_mag_bins):
            action = np.array([ti, min(family_idx, n_families - 1), mi], dtype=np.int64)
            dec = env.decode_action(action)
            if (
                dec["target_var"] == target_var
                and abs(float(dec["magnitude"]) - float(magnitude)) < 0.26
            ):
                return action
    return None


def step_with_seed(env, action: np.ndarray, seed: int) -> Dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    obs2, reward, term, trunc, info = env.step(action)
    decoded = env.decode_action(action)
    if not getattr(_base_env(env), "use_family_templates", True):
        decoded = dict(decoded)
        decoded["family_name"] = "single_root"
    br = info.get("reward_breakdown")
    return {
        "seed": seed,
        "decoded": decoded,
        "reward": float(reward),
        "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
        "dfast_breach": float(br.dfast_breach) if br else 0.0,
        "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
        "diversity": float(br.diversity) if br else 0.0,
        "sampled_state": reset_info.get("sampled_state"),
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    rewards = np.asarray([r["reward"] for r in records], dtype=np.float64)
    losses = np.asarray([r["portfolio_loss"] for r in records], dtype=np.float64)
    dfast = np.asarray([r["dfast_breach"] for r in records], dtype=np.float64)
    causal = np.asarray([r["causal_fidelity"] for r in records], dtype=np.float64)
    diversity = np.asarray([r["diversity"] for r in records], dtype=np.float64)
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_portfolio_loss": float(losses.mean()),
        "mean_dfast_breach": float(dfast.mean()),
        "mean_causal_fidelity": float(causal.mean()),
        "mean_diversity": float(diversity.mean()),
    }


def random_actions(env, rng: np.random.Generator, n: int) -> List[np.ndarray]:
    """Sample n random actions from the env's actual action space."""
    n_targets, n_families, n_mag_bins = (int(x) for x in env.action_space.nvec)
    actions = []
    for _ in range(n):
        actions.append(np.array([
            int(rng.integers(n_targets)),
            int(rng.integers(n_families)),
            int(rng.integers(n_mag_bins)),
        ], dtype=np.int64))
    return actions


def heuristic_actions(env) -> List[Tuple[str, np.ndarray]]:
    """Return hand-picked adversarial actions found by searching the env's action space."""
    candidates = [
        ("gspc_crash",      "^GSPC",          -5.0),
        ("vix_spike",       "^VIX",            5.0),
        ("rates_panic",     "DGS10",           5.0),
        ("credit_blowout",  "BAMLH0A0HYM2",    5.0),
        ("pandemic_unrate", "UNRATE",           5.0),
    ]
    result = []
    for name, var, mag in candidates:
        action = find_action(env, var, mag)
        if action is not None:
            result.append((name, action))
    return result


def brute_force_best_action(env, seed: int) -> Dict[str, Any]:
    """Exhaustively evaluate all valid actions; return the best."""
    n_targets, n_families, n_mag_bins = (int(x) for x in env.action_space.nvec)
    best = None
    for ti in range(n_targets):
        for fi in range(n_families):
            for mi in range(n_mag_bins):
                action = np.array([ti, fi, mi], dtype=np.int64)
                result = step_with_seed(env, action, seed)
                if best is None or result["reward"] > best["reward"]:
                    best = result
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="real", choices=["fast", "real"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--portfolio-profile", default=DEFAULT_PORTFOLIO_PROFILE)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    import json as _json

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    model_path = find_model_path(run_dir)

    # Read run config to match env settings used during training
    cfg_path = run_dir / "config.json"
    run_cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    use_family_templates = bool(run_cfg.get("use_family_templates", False))
    reward_mode = run_cfg.get("reward_mode", "portfolio_adversarial")

    env = make_env(
        mode=args.mode,
        seed=0,
        portfolio_profile=args.portfolio_profile,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )
    model = PPO.load(str(model_path))

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    rng = np.random.default_rng(12345)

    rl_records: List[Dict[str, Any]] = []
    random_records: List[Dict[str, Any]] = []
    heuristic_records: List[Dict[str, Any]] = []
    brute_records: List[Dict[str, Any]] = []

    heuristics = heuristic_actions(env)
    random_pool = random_actions(env, rng, len(seeds))

    for i, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        action, _ = model.predict(obs, deterministic=True)
        rl_result = step_with_seed(env, np.asarray(action, dtype=np.int64), seed)
        rl_records.append(rl_result)

        random_records.append(step_with_seed(env, random_pool[i], seed))

        best_heur = None
        for name, action in heuristics:
            result = step_with_seed(env, action, seed)
            result["heuristic_name"] = name
            if best_heur is None or result["reward"] > best_heur["reward"]:
                best_heur = result
        heuristic_records.append(best_heur)

        brute_records.append(brute_force_best_action(env, seed))

    print()
    print("=" * 72)
    print("One-step adversarial baseline comparison")
    print("=" * 72)
    print(f"  run_dir: {run_dir}")
    print(f"  model:   {model_path.name}")
    print(f"  mode:    {args.mode}")
    print(f"  portfolio: {args.portfolio_profile}")
    print(f"  seeds:   {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print()

    for label, records in [
        ("RL deterministic", rl_records),
        ("Random action", random_records),
        ("Best heuristic", heuristic_records),
        ("Brute-force best single action", brute_records),
    ]:
        s = summarize(records)
        print(
            f"{label:<30} "
            f"reward {s['mean_reward']:+.4f} ± {s['std_reward']:.4f} | "
            f"P/L {s['mean_portfolio_loss']:+.4f} | "
            f"DFAST {s['mean_dfast_breach']:.3f} | "
            f"CF {s['mean_causal_fidelity']:.3f} | "
            f"DIV {s['mean_diversity']:.3f}"
        )

    top_rl = max(rl_records, key=lambda r: r["reward"])
    top_brute = max(brute_records, key=lambda r: r["reward"])
    print()
    print("Top RL action sample:")
    print(
        f"  {top_rl['decoded']['target_var']} / {top_rl['decoded']['family_name']} / "
        f"{top_rl['decoded']['magnitude']:+.2f}s -> reward {top_rl['reward']:+.4f}"
    )
    print("Top brute-force action sample:")
    print(
        f"  {top_brute['decoded']['target_var']} / {top_brute['decoded']['family_name']} / "
        f"{top_brute['decoded']['magnitude']:+.2f}s -> reward {top_brute['reward']:+.4f}"
    )

    result = {
        "run_dir": str(run_dir),
        "model": model_path.name,
        "mode": args.mode,
        "portfolio_profile": args.portfolio_profile,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "rl": summarize(rl_records),
        "random": summarize(random_records),
        "heuristic": summarize(heuristic_records),
        "brute_force": summarize(brute_records),
        "top_rl_action": {
            "target_var": top_rl["decoded"]["target_var"],
            "family_name": top_rl["decoded"]["family_name"],
            "magnitude": top_rl["decoded"]["magnitude"],
            "reward": top_rl["reward"],
        },
        "top_brute_action": {
            "target_var": top_brute["decoded"]["target_var"],
            "family_name": top_brute["decoded"]["family_name"],
            "magnitude": top_brute["decoded"]["magnitude"],
            "reward": top_brute["reward"],
        },
    }
    out_json = (
        Path(args.out_json)
        if args.out_json
        else run_dir / "generalization" / f"baseline_{args.seed_start}_{args.n_seeds}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(_json.dumps(result, indent=2, default=str))
    print(f"\n  saved: {out_json}")


if __name__ == "__main__":
    main()
