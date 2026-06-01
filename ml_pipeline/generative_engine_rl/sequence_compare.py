"""
sequence_compare.py
===================
Compare a trained 2-step RL policy against 2-step baselines on identical seeds.

Baselines:
    1. RL deterministic 2-step policy
    2. Random 2-step action sequence
    3. Hand-picked 2-step heuristic sequence
    4. Beam-search 2-step sequence using top one-step actions as candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # optional; only needed when running sequence_compare.py as a standalone script

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.inspect_policy import find_latest_run, find_model_path
from generative_engine_rl.portfolio_model import DEFAULT_PORTFOLIO_PROFILE


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def action_catalog(env) -> List[np.ndarray]:
    nvec = env.action_space.nvec
    catalog: List[np.ndarray] = []
    for target_idx in range(int(nvec[0])):
        for family_idx in range(int(nvec[1])):
            for mag_idx in range(int(nvec[2])):
                catalog.append(
                    np.array([target_idx, family_idx, mag_idx], dtype=np.int64)
                )
    return catalog


def action_from_decoded(env, target_var: str, magnitude: float) -> np.ndarray:
    for action in action_catalog(env):
        decoded = env.decode_action(action)
        if (
            decoded["target_var"] == target_var
            and abs(float(decoded["magnitude"]) - float(magnitude)) < 1e-6
        ):
            return action
    raise KeyError(f"No wrapped action found for {target_var} / {magnitude:+.2f}σ")


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


def run_sequence(env, actions: Sequence[np.ndarray], seed: int) -> Dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    decoded_actions: List[Dict[str, Any]] = []
    total_reward = 0.0
    final_info: Dict[str, Any] = {}

    for action in actions:
        decoded_actions.append(dict(env.decode_action(action)))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break

    br = final_info.get("reward_breakdown")
    return {
        "seed": seed,
        "sequence": decoded_actions,
        "reward": float(total_reward),
        "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
        "dfast_breach": float(br.dfast_breach) if br else 0.0,
        "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
        "diversity": float(br.diversity) if br else 0.0,
        "sampled_state": reset_info.get("sampled_state"),
    }


def rl_sequence(model, env, seed: int) -> Dict[str, Any]:
    obs, reset_info = env.reset(seed=seed)
    decoded_actions: List[Dict[str, Any]] = []
    total_reward = 0.0
    final_info: Dict[str, Any] = {}

    done_flag = False
    while not done_flag:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.int64)
        decoded_actions.append(dict(env.decode_action(action)))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done_flag = bool(terminated or truncated)
        final_info = info

    br = final_info.get("reward_breakdown")
    return {
        "seed": seed,
        "sequence": decoded_actions,
        "reward": float(total_reward),
        "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
        "dfast_breach": float(br.dfast_breach) if br else 0.0,
        "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
        "diversity": float(br.diversity) if br else 0.0,
        "sampled_state": reset_info.get("sampled_state"),
    }


def random_sequence(env, rng: np.random.Generator, seed: int) -> Dict[str, Any]:
    catalog = action_catalog(env)
    actions = [
        catalog[int(rng.integers(len(catalog)))],
        catalog[int(rng.integers(len(catalog)))],
    ]
    return run_sequence(env, actions, seed)


def heuristic_sequences(env, portfolio_profile: str) -> List[Tuple[str, List[np.ndarray]]]:
    choices: Dict[str, List[Tuple[str, float]]] = {
        "tech_heavy": [
            ("ndx_xlk_crash", [("^NDX", -5.0), ("XLK", -5.0)]),
            ("ndx_rates", [("^NDX", -5.0), ("DGS2", +4.0)]),
            ("tech_vix", [("XLK", -5.0), ("^VIX", +4.0)]),
        ],
        "bond_heavy": [
            ("bond_credit", [("HYG", -5.0), ("LQD", -5.0)]),
            ("rates_credit", [("DGS10", +5.0), ("HYG", -5.0)]),
            ("rates_rates", [("DGS2", +4.0), ("DGS10", +4.0)]),
        ],
        "credit_heavy": [
            ("credit_spread", [("HYG", -5.0), ("BAMLH0A0HYM2", +5.0)]),
            ("credit_financials", [("HYG", -5.0), ("XLF", -5.0)]),
            ("spread_rates", [("BAMLH0A0HYM2", +5.0), ("DGS2", +4.0)]),
        ],
        "balanced": [
            ("broad_equity", [("^GSPC", -5.0), ("XLF", -5.0)]),
            ("equity_rates", [("^GSPC", -5.0), ("DGS10", +4.0)]),
            ("credit_energy", [("XLF", -5.0), ("XLE", -5.0)]),
        ],
    }
    spec = choices.get(portfolio_profile, choices[DEFAULT_PORTFOLIO_PROFILE])
    out: List[Tuple[str, List[np.ndarray]]] = []
    for name, seq in spec:
        out.append(
            (
                name,
                [action_from_decoded(env, target_var, magnitude) for target_var, magnitude in seq],
            )
        )
    return out


def beam_best_sequence(
    seq_env,
    one_step_env,
    seed: int,
    beam_width: int = 8,
) -> Dict[str, Any]:
    scored_actions: List[Tuple[float, np.ndarray]] = []
    for action in action_catalog(one_step_env):
        result = run_sequence(one_step_env, [action], seed)
        scored_actions.append((result["reward"], action))
    scored_actions.sort(key=lambda x: x[0], reverse=True)
    candidate_actions = [action for _, action in scored_actions[:beam_width]]

    best = None
    for first in candidate_actions:
        for second in candidate_actions:
            result = run_sequence(seq_env, [first, second], seed)
            if best is None or result["reward"] > best["reward"]:
                best = result
    return best


def sequence_to_str(record: Dict[str, Any]) -> str:
    return " -> ".join(
        f"{step['target_var']} {step['magnitude']:+.2f}σ"
        for step in record["sequence"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="real", choices=["fast", "real"])
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument("--portfolio-profile", default=None)
    parser.add_argument("--beam-width", type=int, default=8)
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

    seq_env = make_env(
        mode=args.mode,
        seed=0,
        portfolio_profile=portfolio_profile,
        actions_per_episode=2,
        use_family_templates=False,
        reward_mode=reward_mode,
    )
    one_step_env = make_env(
        mode=args.mode,
        seed=0,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        use_family_templates=False,
        reward_mode=reward_mode,
    )

    rng = np.random.default_rng(24680)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    rl_records: List[Dict[str, Any]] = []
    random_records: List[Dict[str, Any]] = []
    heuristic_records: List[Dict[str, Any]] = []
    beam_records: List[Dict[str, Any]] = []

    heuristics = heuristic_sequences(seq_env, portfolio_profile)

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

    print()
    print("=" * 76)
    print("Two-step adversarial sequence comparison")
    print("=" * 76)
    print(f"  run_dir: {run_dir}")
    print(f"  model:   {model_path.name}")
    print(f"  mode:    {args.mode}")
    print(f"  portfolio: {portfolio_profile}")
    print(f"  reward_mode: {reward_mode}")
    print(f"  seeds:   {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print(f"  beam width: {args.beam_width}")
    print()

    for label, records in [
        ("RL deterministic", rl_records),
        ("Random 2-step", random_records),
        ("Best heuristic 2-step", heuristic_records),
        ("Beam best 2-step", beam_records),
    ]:
        s = summarize(records)
        print(
            f"{label:<24} "
            f"reward {s['mean_reward']:+.4f} ± {s['std_reward']:.4f} | "
            f"P/L {s['mean_portfolio_loss']:+.4f} | "
            f"DFAST {s['mean_dfast_breach']:.3f} | "
            f"CF {s['mean_causal_fidelity']:.3f} | "
            f"DIV {s['mean_diversity']:.3f}"
        )

    top_rl = max(rl_records, key=lambda r: r["reward"])
    top_beam = max(beam_records, key=lambda r: r["reward"])
    print()
    print("Top RL sequence sample:")
    print(f"  {sequence_to_str(top_rl)} -> reward {top_rl['reward']:+.4f}")
    print("Top beam sequence sample:")
    print(f"  {sequence_to_str(top_beam)} -> reward {top_beam['reward']:+.4f}")


if __name__ == "__main__":
    main()
