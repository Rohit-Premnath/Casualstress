"""
train_v5_all_profiles.py
========================
Launch v5 training for all 4 portfolio profiles in sequence.

Phase 1c improvements vs v4:
    Crisis seeds in training rollouts (crisis_seed_prob=0.5):
        50% of PPO training episodes start from one of the 12 historical
        crisis states (Lehman, COVID crash, UK gilt, etc.) rather than
        a random stressed-regime draw. This gives the policy denser
        gradient signal from genuinely high-damage starting states,
        targeting the beam gap that v4 left open.

    Extended timesteps (100k vs 50k):
        Doubling training steps gives the policy more opportunity to
        learn the fine-grained action discrimination needed in a
        2000-action space.

All other hyperparameters are held identical to v4 for clean comparison.

Usage (from ml_pipeline/):
    python -m generative_engine_rl.train_v5_all_profiles
    python -m generative_engine_rl.train_v5_all_profiles --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]

BASE_ARGS = [
    "--mode", "real",
    "--n-magnitude-bins", "21",
    "--actions-per-episode", "1",
    "--reward-mode", "portfolio_adversarial",
    "--warm-start-beam",
    "--warm-start-beam-seeds", "12",
    "--warm-start-seed-start", "20000",
    "--warm-start-beam-width", "6",
    "--warm-start-beam-top-k", "3",
    "--warm-start-min-portfolio-loss", "0.1",
    "--warm-start-epochs", "8",
    "--warm-start-batch-size", "32",
    "--warm-start-lr", "0.001",
    "--train-crisis-seed-prob", "0.5",
    "--total-timesteps", "100000",
    "--n-steps", "256",
    "--batch-size", "64",
    "--n-epochs", "4",
    "--ent-coef", "0.05",
    "--eval-freq", "5000",
    "--n-eval-episodes", "30",
    "--checkpoint-freq", "10000",
    "--smoke-test-episodes", "50",
    "--output-dir", "runs",
]


def run_profile(profile: str, dry_run: bool = False) -> None:
    cmd = [
        PYTHON, "-m", "generative_engine_rl.train_ppo",
        "--run-name", f"ppo_v5_{profile}",
        "--portfolio-profile", profile,
        "--seed", "0",
    ] + BASE_ARGS

    print(f"\n{'='*72}")
    print(f"  Training v5 — {profile}")
    print(f"{'='*72}")
    if dry_run:
        print("  [dry-run] would run:", " ".join(cmd))
        return

    env_overrides = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=False, text=True, env=env_overrides,
                          cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"  [WARNING] {profile} training exited with code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--profiles", nargs="+", default=PROFILES,
                        choices=PROFILES, metavar="PROFILE",
                        help="Which profiles to train (default: all 4)")
    args = parser.parse_args()

    print(f"\nPhase 1c RL training — v5")
    print(f"  crisis seeds in training rollouts: prob=0.5")
    print(f"  extended timesteps: 100k")
    print(f"  profiles: {args.profiles}")

    for profile in args.profiles:
        run_profile(profile, dry_run=args.dry_run)

    print(f"\n{'='*72}")
    print("  All v5 training runs complete.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
