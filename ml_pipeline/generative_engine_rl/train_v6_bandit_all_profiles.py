"""
train_v6_bandit_all_profiles.py
================================
Phase 2: Neural Contextual Bandit across all 4 portfolio profiles.

For each profile:
    1. Collect complete reward landscape (all 4200 actions × 10 seeds).
    2. Train BanditRewardNet (MSE + BPR ranking loss, 300 epochs).
    3. Evaluate on held-out seeds 20000-20015, compare vs beam/heuristic/random.
    4. Save results JSON.

Expected runtime: ~15 min data collection + ~2 min training per profile.
Total: ~70 min for all 4 profiles (sequential).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(
    Path(__file__).resolve().parent.parent.parent
    / "backend" / ".venv" / "Scripts" / "python.exe"
)

BASE_ARGS = [
    "--mode", "real",
    "--n-magnitude-bins", "21",
    "--reward-mode", "portfolio_adversarial",
    "--train-seed-start", "1000",
    "--n-train-seeds", "50",     # 50 seeds × 250 actions = 12,500 points (~2 min)
    "--hidden", "128",
    "--dropout", "0.15",
    "--n-epochs", "500",
    "--batch-size", "512",       # smaller batch for 12.5k dataset
    "--lr", "1e-3",
    "--weight-decay", "1e-4",
    "--alpha", "0.6",
    "--patience", "60",
    "--eval-seed-start", "20000",
    "--n-eval-seeds", "16",
    "--beam-width", "6",
    "--ucb-beta", "0.5",
    "--out-dir", "runs",
]

PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]


def main() -> None:
    print()
    print("Phase 2 — Neural Contextual Bandit training")
    print(f"  profiles: {PROFILES}")
    print(f"  train seeds: 1000-1049  (50 seeds × 250 actions = 12.5k training points)")
    print(f"  eval seeds: 20000-20015")
    print()

    results = {}
    for profile in PROFILES:
        print(f"\n{'='*72}")
        print(f"  Training bandit — {profile}")
        print(f"{'='*72}")

        cmd = [
            PYTHON, "-m", "generative_engine_rl.train_bandit",
            "--portfolio-profile", profile,
            "--run-name", f"bandit_v1_{profile}",
            *BASE_ARGS,
        ]

        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        elapsed = time.time() - t0

        status = "PASS" if result.returncode == 0 else "FAIL"
        results[profile] = {"status": status, "elapsed_s": elapsed}
        print(f"\n  [{status}] {profile} in {elapsed:.0f}s")

    print()
    print(f"{'='*72}")
    print("  All bandit training runs complete.")
    print(f"{'='*72}")
    for profile, r in results.items():
        print(f"  {profile}: {r['status']}  ({r['elapsed_s']:.0f}s)")
    print()


if __name__ == "__main__":
    main()
