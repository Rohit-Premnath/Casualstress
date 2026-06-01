"""
teacher_data.py
===============
Shared utilities for building, saving, and loading beam-teacher datasets for
RL warm start, plus reproducible held-out seed panels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env


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


def run_sequence(
    env,
    actions: Sequence[np.ndarray],
    seed: int,
    reset_options: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    obs, _ = env.reset(seed=seed, options=reset_options)
    total_reward = 0.0
    final_info: Dict[str, Any] = {}
    decoded_actions: List[Dict[str, Any]] = []
    for action in actions:
        decoded_actions.append(dict(env.decode_action(action)))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break
    return total_reward, final_info, decoded_actions


def top_beam_sequences(
    seq_env,
    one_step_env,
    seed: int,
    beam_width: int,
    top_k: int,
    reset_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    scored_actions: List[Tuple[float, np.ndarray]] = []
    for action in action_catalog(one_step_env):
        reward, _, _ = run_sequence(one_step_env, [action], seed, reset_options=reset_options)
        scored_actions.append((reward, action))
    scored_actions.sort(key=lambda x: x[0], reverse=True)
    candidate_actions = [action for _, action in scored_actions[:beam_width]]

    candidates: List[Dict[str, Any]] = []
    for first in candidate_actions:
        for second in candidate_actions:
            reward, final_info, decoded_actions = run_sequence(seq_env, [first, second], seed, reset_options=reset_options)
            br = final_info.get("reward_breakdown")
            candidates.append(
                {
                    "actions": [first.copy(), second.copy()],
                    "decoded_actions": decoded_actions,
                    "reward": float(reward),
                    "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
                    "dfast_breach": float(br.dfast_breach) if br else 0.0,
                    "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
                    "diversity": float(br.diversity) if br else 0.0,
                }
            )
    candidates.sort(key=lambda r: r["reward"], reverse=True)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in candidates:
        key = tuple(tuple(int(x) for x in action.tolist()) for action in row["actions"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


def build_beam_teacher_dataset(
    *,
    mode: str,
    seed: int,
    n_magnitude_bins: int,
    portfolio_profile: str,
    actions_per_episode: int,
    use_family_templates: bool,
    reward_mode: str,
    seed_start: int,
    n_seeds: int,
    beam_width: int,
    top_k: int,
    min_portfolio_loss: float,
    use_crisis_seeds: bool = False,
) -> Dict[str, Any]:
    seq_env = make_env(
        mode=mode,
        seed=seed + 200_000,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )
    one_step_env = make_env(
        mode=mode,
        seed=seed + 300_000,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )

    # Resolve crisis seed list from the inner env (empty in fast mode)
    env_crisis_seeds: List[Dict[str, Any]] = []
    if use_crisis_seeds:
        env_inner = seq_env.unwrapped
        env_crisis_seeds = list(getattr(env_inner, "_crisis_seeds", []))
        if env_crisis_seeds:
            print(f"  [teacher] using {len(env_crisis_seeds)} historical crisis starting states")
        else:
            print("  [teacher] use_crisis_seeds=True but no crisis seeds in env (fast mode?)")

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    sample_weights: List[float] = []
    sequence_records: List[Dict[str, Any]] = []
    kept_sequences = 0
    filtered_out_sequences = 0

    for i, current_seed in enumerate(range(seed_start, seed_start + n_seeds)):
        reset_options: Optional[Dict[str, Any]] = None
        if use_crisis_seeds and env_crisis_seeds:
            reset_options = {"crisis_seed_idx": i % len(env_crisis_seeds)}

        top_sequences = top_beam_sequences(
            seq_env=seq_env,
            one_step_env=one_step_env,
            seed=current_seed,
            beam_width=beam_width,
            top_k=top_k,
            reset_options=reset_options,
        )
        filtered_sequences = [
            row
            for row in top_sequences
            if row["portfolio_loss"] >= min_portfolio_loss
        ]
        if not filtered_sequences:
            filtered_sequences = top_sequences[:1]
            filtered_out_sequences += max(0, len(top_sequences) - 1)
        else:
            filtered_out_sequences += max(0, len(top_sequences) - len(filtered_sequences))

        for row in filtered_sequences:
            obs, _ = seq_env.reset(seed=current_seed, options=reset_options)
            weight = max(float(row["reward"]), 1e-6)
            for action in row["actions"]:
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.int64).copy())
                sample_weights.append(weight)
                obs, reward, terminated, truncated, info = seq_env.step(action)
                if terminated or truncated:
                    break
            seq_record: Dict[str, Any] = {
                "seed": current_seed,
                "reward": float(row["reward"]),
                "portfolio_loss": float(row["portfolio_loss"]),
                "dfast_breach": float(row["dfast_breach"]),
                "causal_fidelity": float(row["causal_fidelity"]),
                "diversity": float(row["diversity"]),
                "decoded_actions": row["decoded_actions"],
            }
            if reset_options and "crisis_seed_idx" in reset_options:
                cidx = reset_options["crisis_seed_idx"]
                if cidx < len(env_crisis_seeds):
                    seq_record["crisis_event"] = env_crisis_seeds[cidx].get("event_name", "")
                    seq_record["crisis_date"] = env_crisis_seeds[cidx].get("date", "")
            sequence_records.append(seq_record)
            kept_sequences += 1

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "meta": {
            "mode": mode,
            "portfolio_profile": portfolio_profile,
            "actions_per_episode": actions_per_episode,
            "use_family_templates": use_family_templates,
            "reward_mode": reward_mode,
            "seed_start": seed_start,
            "n_seeds": n_seeds,
            "beam_width": beam_width,
            "top_k": top_k,
            "min_portfolio_loss": min_portfolio_loss,
            "use_crisis_seeds": use_crisis_seeds,
            "n_crisis_seeds_used": len(env_crisis_seeds),
            "kept_sequences": kept_sequences,
            "filtered_out_sequences": filtered_out_sequences,
            "sequence_records": sequence_records,
        },
    }


def build_stagewise_teacher_dataset(
    *,
    mode: str,
    seed: int,
    n_magnitude_bins: int,
    portfolio_profile: str,
    actions_per_episode: int,
    use_family_templates: bool,
    reward_mode: str,
    seed_start: int,
    n_seeds: int,
    beam_width: int,
    top_k: int,
    min_portfolio_loss: float,
    use_crisis_seeds: bool = False,
) -> Dict[str, Any]:
    """Build a conflict-free teacher dataset for sequential decisions.

    The main issue with the earlier warm-start path was that we kept multiple
    top-k sequences for the same initial state. That creates contradictory
    first-step labels for behavior cloning. Here we keep only the best beam
    sequence per seed and store the first and second decisions separately.
    """
    seq_env = make_env(
        mode=mode,
        seed=seed + 200_000,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )
    one_step_env = make_env(
        mode=mode,
        seed=seed + 300_000,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )

    # Resolve crisis seed list from the inner env (empty in fast mode)
    env_crisis_seeds_s: List[Dict[str, Any]] = []
    if use_crisis_seeds:
        env_inner_s = seq_env.unwrapped
        env_crisis_seeds_s = list(getattr(env_inner_s, "_crisis_seeds", []))
        if env_crisis_seeds_s:
            print(f"  [teacher] using {len(env_crisis_seeds_s)} historical crisis starting states (stagewise)")
        else:
            print("  [teacher] use_crisis_seeds=True but no crisis seeds in env (fast mode?)")

    first_observations: List[np.ndarray] = []
    first_actions: List[np.ndarray] = []
    first_weights: List[float] = []
    second_observations: List[np.ndarray] = []
    second_actions: List[np.ndarray] = []
    second_weights: List[float] = []
    sequence_records: List[Dict[str, Any]] = []
    filtered_out_sequences = 0

    for i, current_seed in enumerate(range(seed_start, seed_start + n_seeds)):
        reset_options_s: Optional[Dict[str, Any]] = None
        if use_crisis_seeds and env_crisis_seeds_s:
            reset_options_s = {"crisis_seed_idx": i % len(env_crisis_seeds_s)}

        top_sequences = top_beam_sequences(
            seq_env=seq_env,
            one_step_env=one_step_env,
            seed=current_seed,
            beam_width=beam_width,
            top_k=top_k,
            reset_options=reset_options_s,
        )
        viable = [
            row for row in top_sequences
            if row["portfolio_loss"] >= min_portfolio_loss
        ]
        chosen = viable[0] if viable else top_sequences[0]
        filtered_out_sequences += max(0, len(top_sequences) - (1 if viable else 0))

        obs, _ = seq_env.reset(seed=current_seed, options=reset_options_s)
        seq_weight = max(float(chosen["reward"]), 1e-6)

        first_observations.append(np.asarray(obs, dtype=np.float32).copy())
        first_actions.append(np.asarray(chosen["actions"][0], dtype=np.int64).copy())
        first_weights.append(seq_weight)

        obs_after_first, _, terminated, truncated, _ = seq_env.step(chosen["actions"][0])
        if not (terminated or truncated):
            second_observations.append(
                np.asarray(obs_after_first, dtype=np.float32).copy()
            )
            second_actions.append(
                np.asarray(chosen["actions"][1], dtype=np.int64).copy()
            )
            second_weights.append(seq_weight)

        seq_rec_s: Dict[str, Any] = {
            "seed": current_seed,
            "reward": float(chosen["reward"]),
            "portfolio_loss": float(chosen["portfolio_loss"]),
            "dfast_breach": float(chosen["dfast_breach"]),
            "causal_fidelity": float(chosen["causal_fidelity"]),
            "diversity": float(chosen["diversity"]),
            "decoded_actions": chosen["decoded_actions"],
            "candidate_count": len(top_sequences),
            "alternative_sequences": [
                {
                    "reward": float(row["reward"]),
                    "portfolio_loss": float(row["portfolio_loss"]),
                    "decoded_actions": row["decoded_actions"],
                }
                for row in top_sequences
            ],
        }
        if reset_options_s and "crisis_seed_idx" in reset_options_s:
            cidx_s = reset_options_s["crisis_seed_idx"]
            if cidx_s < len(env_crisis_seeds_s):
                seq_rec_s["crisis_event"] = env_crisis_seeds_s[cidx_s].get("event_name", "")
                seq_rec_s["crisis_date"] = env_crisis_seeds_s[cidx_s].get("date", "")
        sequence_records.append(seq_rec_s)

    observations = []
    actions = []
    sample_weights = []
    stage_ids = []
    if first_observations:
        observations.append(np.asarray(first_observations, dtype=np.float32))
        actions.append(np.asarray(first_actions, dtype=np.int64))
        sample_weights.append(np.asarray(first_weights, dtype=np.float32))
        stage_ids.append(
            np.full(len(first_observations), 1, dtype=np.int64)
        )
    if second_observations:
        observations.append(np.asarray(second_observations, dtype=np.float32))
        actions.append(np.asarray(second_actions, dtype=np.int64))
        sample_weights.append(np.asarray(second_weights, dtype=np.float32))
        stage_ids.append(
            np.full(len(second_observations), 2, dtype=np.int64)
        )

    return {
        "observations": np.concatenate(observations, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "sample_weights": np.concatenate(sample_weights, axis=0),
        "stage_ids": np.concatenate(stage_ids, axis=0),
        "first_observations": np.asarray(first_observations, dtype=np.float32),
        "first_actions": np.asarray(first_actions, dtype=np.int64),
        "first_weights": np.asarray(first_weights, dtype=np.float32),
        "second_observations": np.asarray(second_observations, dtype=np.float32),
        "second_actions": np.asarray(second_actions, dtype=np.int64),
        "second_weights": np.asarray(second_weights, dtype=np.float32),
        "meta": {
            "mode": mode,
            "portfolio_profile": portfolio_profile,
            "actions_per_episode": actions_per_episode,
            "use_family_templates": use_family_templates,
            "reward_mode": reward_mode,
            "seed_start": seed_start,
            "n_seeds": n_seeds,
            "beam_width": beam_width,
            "top_k": top_k,
            "min_portfolio_loss": min_portfolio_loss,
            "kept_sequences": len(sequence_records),
            "filtered_out_sequences": filtered_out_sequences,
            "teacher_style": "stagewise_best_per_seed",
            "use_crisis_seeds": use_crisis_seeds,
            "n_crisis_seeds_used": len(env_crisis_seeds_s),
            "sequence_records": sequence_records,
        },
    }


def save_teacher_dataset(dataset: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in dataset.items()
        if key != "meta" and isinstance(value, np.ndarray)
    }
    np.savez_compressed(out_path, **arrays)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(dataset["meta"], indent=2, default=str))


def load_teacher_dataset(path: Path) -> Dict[str, Any]:
    npz = np.load(path)
    meta_path = path.with_suffix(".json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    out = {key: npz[key] for key in npz.files}
    out["meta"] = meta
    return out
