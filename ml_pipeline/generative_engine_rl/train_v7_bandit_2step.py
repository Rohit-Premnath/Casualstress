"""
train_v7_bandit_2step.py
========================
Launch BanditRewardNet v2 training (2-step) for all 4 portfolio profiles.

Key differences from v1 (train_v6_bandit_all_profiles.py):
  - --n-steps 2: collects both step-1 and step-2 (obs_1, action_2, r_2) data
  - --n-step2-branches 6: expands top-6 step-1 actions at each training seed
  - --eval-actions-per-episode 2: held-out eval runs the bandit in 2-step mode
  - out_dir: runs/bandit_v2_{profile}

Dataset size per profile:
  Step-1: 50 seeds × 250 actions = 12,500 triplets
  Step-2: 50 seeds × 6 branches × 250 actions = 75,000 triplets
  Total: 87,500 triplets (vs 12,500 in v1)

Expected training time per profile: ~90–120 minutes
  (~15 min data collection + ~90 min training at 2.3s/epoch × 500 epochs)

Usage:
    python -m generative_engine_rl.train_v7_bandit_2step
    python -m generative_engine_rl.train_v7_bandit_2step --profiles balanced tech_heavy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Use backend venv if available (has all ML deps); fall back to sys.executable
_venv_py = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_py) if _venv_py.exists() else sys.executable

PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]

COMMON_ARGS = [
    "--mode", "real",
    "--n-magnitude-bins", "21",
    "--reward-mode", "portfolio_adversarial",
    "--n-steps", "2",
    "--n-step2-branches", "6",
    "--train-seed-start", "1000",
    "--n-train-seeds", "50",
    "--n-epochs", "500",
    "--batch-size", "2048",
    "--lr", "1e-3",
    "--alpha", "0.6",
    "--patience", "60",
    "--eval-seed-start", "20000",
    "--n-eval-seeds", "16",
    "--beam-width", "6",
    "--eval-actions-per-episode", "2",
    "--ucb-beta", "0.5",
]


PROFILE_EXTRA_ARGS = {
    "credit_heavy": ["--crisis-seed-prob", "0.5"],
    "bond_heavy": ["--crisis-seed-prob", "0.3"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="+", default=PROFILES,
                   choices=PROFILES, metavar="PROFILE")
    args = p.parse_args()

    for profile in args.profiles:
        out_dir = ROOT / "runs" / f"bandit_v2_{profile}"
        print(f"\n{'='*68}")
        print(f"  Training bandit_v2  [{profile}]")
        print(f"  out_dir: {out_dir}")
        print(f"{'='*68}\n")

        extra = PROFILE_EXTRA_ARGS.get(profile, [])
        cmd = [
            PYTHON, "-m", "generative_engine_rl.train_bandit",
            "--portfolio-profile", profile,
            "--out-dir", str(out_dir),
            "--run-name", f"bandit_v2_{profile}",
        ] + COMMON_ARGS + extra

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n[FAIL] {profile} exited with code {result.returncode}")
            sys.exit(result.returncode)
        print(f"\n[DONE] {profile}")


if __name__ == "__main__":
    main()
