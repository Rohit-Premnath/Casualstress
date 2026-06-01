"""
train_v8_bandit_v3.py
=====================
Bandit v3 training for the 3 profiles that didn't clear the 85% gate in v2.
Bond_heavy already cleared (86.4%) and is skipped.

Changes from v2 (train_v7_bandit_2step.py):
  - --n-step2-branches 12 (was 6): doubles step-2 data, better conditional landscape coverage
  - --train-seed-start 2000: fresh seed range, no overlap with v2 (1000-1049)
  - Per-profile seed counts (override COMMON_ARGS default of 50):
      balanced:     150 seeds  (obs space is diffuse; coverage is the bottleneck)
      credit_heavy: 100 seeds  (still on steep scaling curve; v1->v2 gained +23pp)
      tech_heavy:    50 seeds  (gap is pathway diversity, not volume)
  - tech_heavy gets crisis_seed_prob=0.3 (missing macro-rate starting states in v2)

Dataset sizes:
  balanced:     150x250 + 150x12x250 = 487,500 triplets
  credit_heavy: 100x250 + 100x12x250 = 325,000 triplets
  tech_heavy:    50x250 +  50x12x250 = 162,500 triplets

Estimated training time per profile:
  tech_heavy:   ~2-3 hours
  credit_heavy: ~4-6 hours
  balanced:     ~6-8 hours

Usage:
    python -m generative_engine_rl.train_v8_bandit_v3
    python -m generative_engine_rl.train_v8_bandit_v3 --profiles tech_heavy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_venv_py = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_py) if _venv_py.exists() else sys.executable

# Bond_heavy cleared 85% gate at 86.4% in v2 — not included
PROFILES = ["tech_heavy", "credit_heavy", "balanced"]

COMMON_ARGS = [
    "--mode", "real",
    "--n-magnitude-bins", "21",
    "--reward-mode", "portfolio_adversarial",
    "--n-steps", "2",
    "--n-step2-branches", "12",        # doubled from v2's 6
    "--train-seed-start", "2000",       # fresh range; v2 used 1000-1049
    "--n-train-seeds", "50",            # default; overridden per-profile below
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

# Extra args appended after COMMON_ARGS — argparse last-value-wins resolves duplicates
PROFILE_EXTRA_ARGS = {
    "balanced": [
        "--n-train-seeds", "150",       # diffuse obs space needs broad seed coverage
    ],
    "tech_heavy": [
        "--crisis-seed-prob", "0.3",    # bandit was missing macro-rate pathways (FEDFUNDS->UNRATE)
    ],
    "credit_heavy": [
        "--n-train-seeds", "100",       # v1->v2 gained +23pp; still on steep scaling curve
        "--crisis-seed-prob", "0.5",    # keep same as v2
    ],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="+", default=PROFILES,
                   choices=PROFILES, metavar="PROFILE")
    args = p.parse_args()

    for profile in args.profiles:
        out_dir = ROOT / "runs" / f"bandit_v3_{profile}"
        print(f"\n{'='*68}")
        print(f"  Training bandit_v3  [{profile}]")
        print(f"  out_dir: {out_dir}")
        print(f"{'='*68}\n")

        extra = PROFILE_EXTRA_ARGS.get(profile, [])
        cmd = [
            PYTHON, "-m", "generative_engine_rl.train_bandit",
            "--portfolio-profile", profile,
            "--out-dir", str(out_dir),
            "--run-name", f"bandit_v3_{profile}",
        ] + COMMON_ARGS + extra

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n[FAIL] {profile} exited with code {result.returncode}")
            sys.exit(result.returncode)
        print(f"\n[DONE] {profile}")


if __name__ == "__main__":
    main()
