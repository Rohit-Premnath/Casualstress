"""
run_all_benchmarks.py
=====================
Run held-out benchmarks for all 4 trained portfolio profiles and produce a
cross-profile comparison summary JSON.

Runs for each profile:
    1. baseline_compare.py  — RL vs random vs heuristic vs brute-force (one-step)
    2. heldout_generalization.py — RL vs random vs heuristic vs beam (16 seeds)

Then runs once:
    3. profile_search_compare.py — brute-force worst shock per profile (no model)

Usage (from ml_pipeline/):
    python -m generative_engine_rl.run_all_benchmarks
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs"

PROFILES = {
    "balanced":     "ppo_v3_diverse_20260510_034843",
    "tech_heavy":   "ppo_v3_tech_heavy_20260510_155516",
    "bond_heavy":   "ppo_v3_bond_heavy_20260510_160555",
    "credit_heavy": "ppo_v3_credit_heavy_20260510_161540",
}


def run_cmd(cmd: list[str], label: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    env_overrides = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=False, text=True, env=env_overrides)
    if proc.returncode != 0:
        print(f"  [WARNING] exited with code {proc.returncode}")


def main() -> None:
    python = sys.executable

    # ---- Per-profile benchmarks ----
    for profile, run_name in PROFILES.items():
        run_dir = RUNS / run_name
        if not (run_dir / "final.zip").exists():
            print(f"  [SKIP] {profile}: no final.zip in {run_dir}")
            continue

        # 1. baseline_compare: one-step RL vs random vs heuristic vs brute-force
        run_cmd(
            [
                python, "-m", "generative_engine_rl.baseline_compare",
                "--run-dir", str(run_dir),
                "--portfolio-profile", profile,
                "--n-seeds", "10",
                "--seed-start", "4000",
                "--mode", "real",
            ],
            f"baseline_compare — {profile}",
        )

        # 2. heldout_generalization: RL vs random vs heuristic vs beam (16 seeds)
        run_cmd(
            [
                python, "-m", "generative_engine_rl.heldout_generalization",
                "--run-dir", str(run_dir),
                "--portfolio-profile", profile,
                "--n-seeds", "16",
                "--seed-start", "20000",
                "--mode", "real",
                "--beam-width", "6",
            ],
            f"heldout_generalization — {profile}",
        )

    # ---- Cross-profile brute-force shock comparison ----
    run_cmd(
        [
            python, "-m", "generative_engine_rl.profile_search_compare",
            "--n-seeds", "5",
            "--seed-start", "5000",
            "--mode", "real",
        ],
        "profile_search_compare — all profiles",
    )

    # ---- Aggregate all heldout JSON results ----
    print(f"\n{'='*72}")
    print("  CROSS-PROFILE SUMMARY (heldout_generalization)")
    print(f"{'='*72}")
    print(f"  {'Profile':<14}  {'RL mean':>9}  {'Random':>9}  {'Heuristic':>10}  {'Beam':>9}  {'RL lift':>8}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*8}")

    summary: Dict[str, Any] = {}
    for profile, run_name in PROFILES.items():
        run_dir = RUNS / run_name
        gen_dir = run_dir / "generalization"
        json_files = sorted(gen_dir.glob("heldout_20000_16.json")) if gen_dir.exists() else []
        if not json_files:
            print(f"  {profile:<14}  (no results)")
            continue
        data = json.loads(json_files[0].read_text())
        rl_r = data["rl"]["mean_reward"]
        rnd_r = data["random"]["mean_reward"]
        heur_r = data["heuristic"]["mean_reward"]
        beam_r = data["beam"]["mean_reward"]
        lift = (rl_r / max(rnd_r, 1e-6) - 1) * 100
        print(f"  {profile:<14}  {rl_r:>+9.4f}  {rnd_r:>+9.4f}  {heur_r:>+10.4f}  {beam_r:>+9.4f}  {lift:>+7.1f}%")
        summary[profile] = {
            "heldout": data,
            "baseline": None,
        }
        # Try to load baseline JSON too
        base_files = sorted(gen_dir.glob("baseline_4000_10.json")) if gen_dir.exists() else []
        if base_files:
            summary[profile]["baseline"] = json.loads(base_files[0].read_text())

    out = ROOT / "runs" / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Cross-profile summary saved: {out}")


if __name__ == "__main__":
    main()
