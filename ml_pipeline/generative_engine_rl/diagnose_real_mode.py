"""
diagnose_real_mode.py
======================
Pre-flight check for real-mode RL training.

Verifies that everything real-mode needs is in place before launching an
expensive training run. Cheaper to fail in 1 second here than to fail
30 seconds into a multi-hour training run.

Usage:
    python -m generative_engine_rl.diagnose_real_mode

Exit code 0 = real mode is ready.
Exit code 1 = at least one prerequisite is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.real_mode_loader import (
    diagnose_real_mode_readiness,
    print_diagnosis,
)


def main():
    print("Running real-mode pre-flight check...")
    report = diagnose_real_mode_readiness()
    print_diagnosis(report)

    all_ready = (
        report["production_imports"]
        and report["db_connection"]
        and report["processed_data_present"]
        and report["regimes_present"]
        and report["canonical_graph_present"]
    )

    if all_ready:
        print("All prerequisites present. Real mode should work.")
        print()
        print("Next steps:")
        print("  Quick smoke test of real mode (1-3 minutes, ~500 timesteps):")
        print("    python -m generative_engine_rl.train_ppo \\")
        print("        --mode real --total-timesteps 512 --n-envs 1 \\")
        print("        --n-steps 64 --eval-freq 0 --checkpoint-freq 0")
        print()
        print("  Full real-mode training (multi-hour):")
        print("    python -m generative_engine_rl.train_ppo \\")
        print("        --mode real --total-timesteps 200000 --n-envs 1")
        sys.exit(0)
    else:
        print("Real mode NOT ready. Address the FAIL items above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
