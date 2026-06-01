"""
heldout_generalization.py
=========================
Run a reproducible held-out sequence benchmark for a trained RL policy and
save the results as a JSON artifact under the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.inspect_policy import find_latest_run, find_model_path
from generative_engine_rl.portfolio_model import DEFAULT_PORTFOLIO_PROFILE
from generative_engine_rl.sequence_compare import (
    beam_best_sequence,
    heuristic_sequences,
    load_run_config,
    random_sequence,
    rl_sequence,
    run_sequence,
    sequence_to_str,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="real", choices=["fast", "real"])
    parser.add_argument("--portfolio-profile", default=None)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    model_path = find_model_path(run_dir)
    model = PPO.load(str(model_path))
    run_cfg = load_run_config(run_dir)
    portfolio_profile = (
        args.portfolio_profile
        if args.portfolio_profile is not None
        else run_cfg.get("portfolio_profile", DEFAULT_PORTFOLIO_PROFILE)
    )
    reward_mode = run_cfg.get("reward_mode", "portfolio_adversarial")
    actions_per_episode = int(run_cfg.get("actions_per_episode", 2))
    use_family_templates = bool(run_cfg.get("use_family_templates", False))

    seq_env = make_env(
        mode=args.mode,
        seed=0,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )
    one_step_env = make_env(
        mode=args.mode,
        seed=0,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    heuristics = heuristic_sequences(seq_env, portfolio_profile)

    rl_records: List[Dict[str, Any]] = []
    random_records: List[Dict[str, Any]] = []
    heuristic_records: List[Dict[str, Any]] = []
    beam_records: List[Dict[str, Any]] = []

    import numpy as np
    rng = np.random.default_rng(24680)

    for seed in seeds:
        rl_records.append(rl_sequence(model, seq_env, seed))
        random_records.append(random_sequence(seq_env, rng, seed))

        best_heur = None
        for name, actions in heuristics:
            result = run_sequence(seq_env, actions, seed)
            result["heuristic_name"] = name
            if best_heur is None or result["reward"] > best_heur["reward"]:
                best_heur = result
        heuristic_records.append(best_heur)
        beam_records.append(
            beam_best_sequence(
                seq_env=seq_env,
                one_step_env=one_step_env,
                seed=seed,
                beam_width=args.beam_width,
            )
        )

    summary = {
        "run_dir": str(run_dir),
        "model": model_path.name,
        "mode": args.mode,
        "portfolio_profile": portfolio_profile,
        "reward_mode": reward_mode,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "beam_width": args.beam_width,
        "rl": summarize(rl_records),
        "random": summarize(random_records),
        "heuristic": summarize(heuristic_records),
        "beam": summarize(beam_records),
        "top_rl_sequence": sequence_to_str(max(rl_records, key=lambda r: r["reward"])),
        "top_beam_sequence": sequence_to_str(max(beam_records, key=lambda r: r["reward"])),
    }

    out_json = (
        Path(args.out_json)
        if args.out_json
        else run_dir / "generalization" / f"heldout_{args.seed_start}_{args.n_seeds}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 76)
    print("Held-out generalization benchmark")
    print("=" * 76)
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio: {portfolio_profile}")
    print(f"  reward_mode: {reward_mode}")
    print(f"  seeds: {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print(f"  RL reward:       {summary['rl']['mean_reward']:+.4f} ± {summary['rl']['std_reward']:.4f}")
    print(f"  Random reward:   {summary['random']['mean_reward']:+.4f} ± {summary['random']['std_reward']:.4f}")
    print(f"  Heuristic reward:{summary['heuristic']['mean_reward']:+.4f} ± {summary['heuristic']['std_reward']:.4f}")
    print(f"  Beam reward:     {summary['beam']['mean_reward']:+.4f} ± {summary['beam']['std_reward']:.4f}")
    print(f"  top RL sequence:   {summary['top_rl_sequence']}")
    print(f"  top beam sequence: {summary['top_beam_sequence']}")
    print(f"  saved: {out_json}")


if __name__ == "__main__":
    main()
