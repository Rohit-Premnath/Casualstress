"""
eval_bandit.py
==============
Standalone evaluation for a saved BanditRewardNet.

Loads bandit.pt from the specified run directory and runs the held-out
benchmark in 1-step mode (to match PPO v5's evaluation setup exactly),
saving a new heldout_results_1step.json alongside the original.

Usage:
    python -m generative_engine_rl.eval_bandit --run-dir runs/bandit_v1_balanced
    python -m generative_engine_rl.eval_bandit --all-profiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.neural_bandit import BanditRewardNet
from generative_engine_rl.train_bandit import run_heldout_eval, print_summary


PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]


def eval_one(run_dir: Path, eval_actions_per_episode: int, out_suffix: str) -> dict:
    model_path = run_dir / "bandit.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"bandit.pt not found in {run_dir}")

    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    profile = cfg.get("portfolio_profile", run_dir.name.split("bandit_v1_")[-1])

    print(f"\n{'='*68}")
    print(f"  Bandit eval — {profile}  (actions_per_ep={eval_actions_per_episode})")
    print(f"{'='*68}")

    net = BanditRewardNet.load(str(model_path))

    summary = run_heldout_eval(
        net=net,
        mode=cfg.get("mode", "real"),
        portfolio_profile=profile,
        n_magnitude_bins=cfg.get("n_magnitude_bins", 21),
        reward_mode=cfg.get("reward_mode", "portfolio_adversarial"),
        seed_start=cfg.get("eval_seed_start", 20000),
        n_seeds=cfg.get("n_eval_seeds", 16),
        beam_width=cfg.get("beam_width", 6),
        eval_actions_per_episode=eval_actions_per_episode,
        ucb_beta=cfg.get("ucb_beta", 0.5),
        verbose=True,
    )
    summary["eval_actions_per_episode"] = eval_actions_per_episode

    out_json = run_dir / f"heldout_results{out_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print_summary(summary)
    print(f"  saved: {out_json}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--all-profiles", action="store_true")
    p.add_argument("--eval-actions-per-episode", type=int, default=1,
                   help="1=match PPO v5 (default), 2=full 2-step eval")
    args = p.parse_args()

    suffix = f"_{args.eval_actions_per_episode}step"
    runs_dir = ROOT / "runs"

    if args.all_profiles:
        for profile in PROFILES:
            run_dir = runs_dir / f"bandit_v1_{profile}"
            if not run_dir.exists():
                print(f"  [SKIP] {run_dir} not found")
                continue
            try:
                eval_one(run_dir, args.eval_actions_per_episode, suffix)
            except Exception as e:
                print(f"  [FAIL] {profile}: {e}")
    elif args.run_dir:
        eval_one(Path(args.run_dir), args.eval_actions_per_episode, suffix)
    else:
        print("Provide --run-dir <path> or --all-profiles")


if __name__ == "__main__":
    main()
