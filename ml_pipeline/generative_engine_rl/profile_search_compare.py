"""
profile_search_compare.py
=========================
Compare the worst plausible one-step shock across named portfolio profiles.

This is the product-facing benchmark:
    "Different portfolios have different weak points; what shock is worst
     for each one under the same causal scenario engine?"
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

from generative_engine_rl.baseline_compare import brute_force_best_action
from generative_engine_rl.env_factory import make_env
from generative_engine_rl.portfolio_model import PORTFOLIO_PROFILES


def summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    rewards = np.asarray([r["reward"] for r in records], dtype=np.float64)
    losses = np.asarray([r["portfolio_loss"] for r in records], dtype=np.float64)
    dfast = np.asarray([r["dfast_breach"] for r in records], dtype=np.float64)
    causal = np.asarray([r["causal_fidelity"] for r in records], dtype=np.float64)
    diversity = np.asarray([r["diversity"] for r in records], dtype=np.float64)
    return {
        "mean_reward": float(rewards.mean()),
        "mean_portfolio_loss": float(losses.mean()),
        "mean_dfast_breach": float(dfast.mean()),
        "mean_causal_fidelity": float(causal.mean()),
        "mean_diversity": float(diversity.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="real", choices=["fast", "real"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=5000)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print()
    print("=" * 84)
    print("Portfolio-specific worst plausible one-step shock comparison")
    print("=" * 84)
    print(f"  mode:  {args.mode}")
    print(f"  seeds: {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print()

    for profile_name in sorted(PORTFOLIO_PROFILES):
        env = make_env(mode=args.mode, seed=0, portfolio_profile=profile_name)
        best_records = [brute_force_best_action(env, seed) for seed in seeds]
        summary = summarize(best_records)
        top = max(best_records, key=lambda r: r["reward"])
        print(
            f"{profile_name:<14} "
            f"reward {summary['mean_reward']:+.4f} | "
            f"P/L {summary['mean_portfolio_loss']:+.4f} | "
            f"DFAST {summary['mean_dfast_breach']:.3f} | "
            f"CF {summary['mean_causal_fidelity']:.3f} | "
            f"DIV {summary['mean_diversity']:.3f}"
        )
        print(
            f"  worst shock: {top['decoded']['target_var']} / "
            f"{top['decoded']['family_name']} / "
            f"{top['decoded']['magnitude']:+.2f}σ"
        )
        print()


if __name__ == "__main__":
    main()
