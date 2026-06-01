"""
train_ppo.py
============
PPO training scaffolding for the adversarial scenario generator.

This is *infrastructure*, not a converged training run. It provides:
    - Vectorized env construction via env_factory
    - PPO with MultiDiscrete action space and MlpPolicy
    - Periodic checkpointing
    - Periodic evaluation against a held-out env
    - Tensorboard logging
    - Smoke-test sanity check (random vs trained policy comparison)

USAGE
-----
Smoke test (default — fast mode, ~1 min):
    python -m generative_engine_rl.train_ppo

Production run (once you're past scaffolding):
    python -m generative_engine_rl.train_ppo --mode real --total-timesteps 200000

Resume from checkpoint:
    python -m generative_engine_rl.train_ppo --resume-from runs/<timestamp>/ckpt_50000.zip

OUTPUT
------
runs/
  <run_name>/
    ckpt_<step>.zip         # periodic checkpoints
    final.zip               # final policy
    config.json             # all hyperparameters and CLI flags
    eval_results.json       # random vs trained policy comparison
    tb/                     # tensorboard event files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env, env_factory_for_subproc
from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.portfolio_model import DEFAULT_PORTFOLIO_PROFILE
from generative_engine_rl.teacher_data import (
    build_beam_teacher_dataset,
    build_stagewise_teacher_dataset,
    load_teacher_dataset,
    save_teacher_dataset,
)


# ============================================================================
# LEARNING-SIGNALS CALLBACK
# ============================================================================

class LearningSignalsCallback(BaseCallback):
    """Capture per-rollout training signals for the smoke-test verdict.

    The smoke test cannot rely on a single threshold (e.g. trained_mean >
    random_mean + 0.5σ) for real-mode runs because real trajectories are
    stochastic — even a converged adversarial policy still has reward
    variance from the env's stochastic VAR sampler.

    Instead we capture the *trajectory* of three orthogonal learning
    signals across the training run:

        1. ep_rew_mean trend     — is reward improving over time?
        2. entropy_loss trend    — is the policy committing to specific actions?
        3. max clip_fraction     — is the policy making nontrivial updates?

    Any TWO of these going in the right direction is sufficient evidence
    that PPO is learning. This is robust to the noise floor of stochastic
    real-mode envs.
    """

    def __init__(self):
        super().__init__()
        self.history: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # SB3 puts metrics in self.logger.name_to_value
        nv = self.logger.name_to_value
        snapshot = {
            "timestep": int(self.num_timesteps),
            "ep_rew_mean": float(nv.get("rollout/ep_rew_mean", float("nan"))),
            "entropy_loss": float(nv.get("train/entropy_loss", float("nan"))),
            "clip_fraction": float(nv.get("train/clip_fraction", float("nan"))),
            "approx_kl": float(nv.get("train/approx_kl", float("nan"))),
        }
        self.history.append(snapshot)

    def diagnose_learning(self) -> Dict[str, Any]:
        """Return a learning-signals report with PASS/FAIL per signal."""
        # Filter to snapshots where PPO has actually run at least one update
        # (entropy_loss is NaN until the first PPO update completes)
        clean = [
            r for r in self.history
            if r["entropy_loss"] == r["entropy_loss"]
        ]
        if len(clean) < 4:
            return {
                "n_rollouts": len(self.history),
                "n_clean_rollouts": len(clean),
                "verdict": "INCONCLUSIVE",
                "reason": "fewer than 4 PPO update rollouts captured — too short to assess",
            }

        # Use first quartile vs last quartile to assess trends
        q = max(len(clean) // 4, 1)
        first_q = clean[:q]
        last_q = clean[-q:]

        def safe_mean(rows, key):
            vals = [r[key] for r in rows if r[key] == r[key]]    # filter NaN
            return sum(vals) / len(vals) if vals else float("nan")

        # Reward trend (may be NaN-only in some configurations)
        first_rew = safe_mean(first_q, "ep_rew_mean")
        last_rew = safe_mean(last_q, "ep_rew_mean")
        rew_available = (first_rew == first_rew) and (last_rew == last_rew)
        if rew_available:
            rew_trend = last_rew - first_rew
            rew_improved = rew_trend > 0
        else:
            rew_trend = float("nan")
            rew_improved = None    # signal unavailable

        # Entropy trend
        first_ent = safe_mean(first_q, "entropy_loss")
        last_ent = safe_mean(last_q, "entropy_loss")
        # Entropy loss is negative; "less negative" toward 0 is more deterministic.
        # Decrease in magnitude (positive trend) means policy is concentrating.
        ent_trend = last_ent - first_ent
        ent_decreased = ent_trend > 0.5

        # Max clip_fraction across run
        max_clip = max(
            (r["clip_fraction"] for r in clean if r["clip_fraction"] == r["clip_fraction"]),
            default=0.0,
        )
        clipping_active = max_clip > 0.01

        # Count signals that can fire (skip rew_improved if it's None)
        active_signals = [s for s in [rew_improved, ent_decreased, clipping_active] if s is not None]
        signals_passed = sum(active_signals)
        signals_total = len(active_signals)

        verdict = "PASS" if signals_passed >= 2 else "FAIL"

        return {
            "n_rollouts": len(self.history),
            "n_clean_rollouts": len(clean),
            "first_quartile_reward": first_rew,
            "last_quartile_reward": last_rew,
            "reward_trend": rew_trend,
            "reward_available": rew_available,
            "reward_improved": rew_improved,
            "first_quartile_entropy": first_ent,
            "last_quartile_entropy": last_ent,
            "entropy_trend": ent_trend,
            "entropy_decreased": ent_decreased,
            "max_clip_fraction": max_clip,
            "clipping_active": clipping_active,
            "signals_passed": signals_passed,
            "signals_total": signals_total,
            "verdict": verdict,
        }


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    # Run
    run_name: str = "ppo_adversarial"
    output_dir: str = "runs"
    seed: int = 0

    # Env
    mode: str = "real"                              # fast | real
    n_envs: int = 1
    n_magnitude_bins: int = 21
    regime_at_episode_start: str = "stressed"
    portfolio_profile: str = DEFAULT_PORTFOLIO_PROFILE
    actions_per_episode: int = 1
    use_family_templates: bool = True
    reward_mode: str = "portfolio_adversarial"
    warm_start_beam: bool = False
    warm_start_dataset_path: Optional[str] = None
    warm_start_seed_start: int = 10_000
    warm_start_beam_seeds: int = 12
    warm_start_beam_width: int = 6
    warm_start_beam_top_k: int = 3
    warm_start_min_portfolio_loss: float = 0.10
    warm_start_use_crisis_seeds: bool = True
    train_crisis_seed_prob: float = 0.5
    warm_start_epochs: int = 8
    warm_start_batch_size: int = 32
    warm_start_lr: float = 1e-3
    use_subproc: bool = False                       # SubprocVecEnv vs DummyVecEnv

    # PPO hyperparameters
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 256                              # rollout buffer per env (real mode: 256 episodes/rollout)
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Logging
    tensorboard: bool = True
    eval_freq: int = 5000
    n_eval_episodes: int = 30
    checkpoint_freq: int = 10_000

    # Smoke test
    run_smoke_test: bool = True
    smoke_test_episodes: int = 50

    # Resume
    resume_from: Optional[str] = None


# ============================================================================
# HELPERS
# ============================================================================

def make_vec_env_local(cfg: TrainConfig):
    """Build a vectorized env. Each env gets a distinct seed."""
    if cfg.use_subproc and cfg.n_envs > 1:
        if cfg.mode == "real":
            print(
                "  WARNING: --use-subproc with --mode real opens one DB "
                "connection per env. Make sure your Postgres max_connections "
                f"can handle {cfg.n_envs}+ concurrent connections, and the "
                "VAR fit will run {cfg.n_envs} times redundantly. Consider "
                "DummyVecEnv (default) for real-mode runs."
            )
        env_fns = [
            env_factory_for_subproc(
                mode=cfg.mode,
                base_seed=cfg.seed,
                rank=i,
                regime_at_episode_start=cfg.regime_at_episode_start,
                n_magnitude_bins=cfg.n_magnitude_bins,
                portfolio_profile=cfg.portfolio_profile,
                actions_per_episode=cfg.actions_per_episode,
                use_family_templates=cfg.use_family_templates,
                reward_mode=cfg.reward_mode,
                crisis_seed_prob=cfg.train_crisis_seed_prob,
            )
            for i in range(cfg.n_envs)
        ]
        return SubprocVecEnv(env_fns)
    else:
        env_fns = [
            (lambda i=i: Monitor(
                make_env(
                    mode=cfg.mode,
                    seed=cfg.seed + i,
                    regime_at_episode_start=cfg.regime_at_episode_start,
                    n_magnitude_bins=cfg.n_magnitude_bins,
                    portfolio_profile=cfg.portfolio_profile,
                    actions_per_episode=cfg.actions_per_episode,
                    use_family_templates=cfg.use_family_templates,
                    reward_mode=cfg.reward_mode,
                    crisis_seed_prob=cfg.train_crisis_seed_prob,
                )
            ))
            for i in range(cfg.n_envs)
        ]
        return DummyVecEnv(env_fns)


def evaluate_policy_episodes(model, env, n_episodes: int, deterministic: bool = True):
    """Run n_episodes of a policy on the env, return rewards and per-episode info."""
    rewards: List[float] = []
    breakdowns: List[Dict[str, float]] = []
    rejections = 0
    obs = env.reset()
    for _ in range(n_episodes):
        done_flag = False
        episode_reward = 0.0
        final_info: Dict[str, Any] = {}
        while not done_flag:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info_list = env.step(action)
            info = info_list[0] if isinstance(info_list, (list, tuple)) else info_list
            episode_reward += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            done_flag = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            final_info = info if isinstance(info, dict) else {}
            if final_info.get("action_rejected"):
                rejections += 1
        rewards.append(float(episode_reward))
        if "reward_breakdown" in final_info:
            br = final_info["reward_breakdown"]
            breakdowns.append({
                "portfolio_loss": float(br.portfolio_loss),
                "dfast_breach": float(br.dfast_breach),
                "causal_fidelity": float(br.causal_fidelity),
                "diversity": float(br.diversity),
                "total": float(br.total),
            })
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "n_episodes": n_episodes,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "rejections": rejections,
        "breakdowns": breakdowns[:10],   # sample first 10 for logging
    }


def run_random_baseline(cfg: TrainConfig, eval_env, n_episodes: int):
    """Random-policy baseline as a sanity-check denominator."""
    rng = np.random.default_rng(cfg.seed + 9999)
    rewards: List[float] = []
    obs = eval_env.reset()
    spec = load_spec()
    n_targets = int(getattr(eval_env.action_space, "nvec", np.array([len(spec.core_variables), 1, cfg.n_magnitude_bins]))[0])
    action_nvec = getattr(eval_env.action_space, "nvec", np.array([n_targets, 1, cfg.n_magnitude_bins]))
    n_families = int(action_nvec[1])
    n_mag = int(action_nvec[2])
    for _ in range(n_episodes):
        done_flag = False
        episode_reward = 0.0
        while not done_flag:
            a = np.array([
                [rng.integers(n_targets), rng.integers(n_families), rng.integers(n_mag)]
            ])
            obs, reward, done, info_list = eval_env.step(a)
            episode_reward += float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            done_flag = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        rewards.append(float(episode_reward))
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "n_episodes": n_episodes,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
    }


def _action_catalog(env) -> List[np.ndarray]:
    nvec = env.action_space.nvec
    catalog: List[np.ndarray] = []
    for target_idx in range(int(nvec[0])):
        for family_idx in range(int(nvec[1])):
            for mag_idx in range(int(nvec[2])):
                catalog.append(
                    np.array([target_idx, family_idx, mag_idx], dtype=np.int64)
                )
    return catalog


def _run_sequence(env, actions: List[np.ndarray], seed: int):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    final_info: Dict[str, Any] = {}
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break
    return total_reward, final_info


def _top_beam_sequences(
    seq_env,
    one_step_env,
    seed: int,
    beam_width: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    scored_actions: List[tuple[float, np.ndarray]] = []
    for action in _action_catalog(one_step_env):
        reward, _ = _run_sequence(one_step_env, [action], seed)
        scored_actions.append((reward, action))
    scored_actions.sort(key=lambda x: x[0], reverse=True)
    candidate_actions = [action for _, action in scored_actions[:beam_width]]

    candidates: List[Dict[str, Any]] = []
    for first in candidate_actions:
        for second in candidate_actions:
            reward, final_info = _run_sequence(seq_env, [first, second], seed)
            br = final_info.get("reward_breakdown")
            candidates.append(
                {
                    "actions": [first.copy(), second.copy()],
                    "reward": float(reward),
                    "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
                    "dfast_breach": float(br.dfast_breach) if br else 0.0,
                }
            )
    candidates.sort(key=lambda r: r["reward"], reverse=True)
    if not candidates:
        raise RuntimeError("Beam search did not produce any candidate sequence.")
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


def _collect_beam_teacher_dataset(cfg: TrainConfig) -> Dict[str, Any]:
    seq_env = make_env(
        mode=cfg.mode,
        seed=cfg.seed + 200_000,
        n_magnitude_bins=cfg.n_magnitude_bins,
        portfolio_profile=cfg.portfolio_profile,
        actions_per_episode=cfg.actions_per_episode,
        use_family_templates=cfg.use_family_templates,
        reward_mode=cfg.reward_mode,
    )
    one_step_env = make_env(
        mode=cfg.mode,
        seed=cfg.seed + 300_000,
        n_magnitude_bins=cfg.n_magnitude_bins,
        portfolio_profile=cfg.portfolio_profile,
        actions_per_episode=1,
        use_family_templates=cfg.use_family_templates,
        reward_mode=cfg.reward_mode,
    )

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    sample_weights: List[float] = []
    seq_rewards: List[float] = []
    seq_portfolio_losses: List[float] = []
    seed_start = cfg.seed + 10_000
    used_seeds: List[int] = []
    kept_sequences = 0
    filtered_out_sequences = 0

    for seed in range(seed_start, seed_start + cfg.warm_start_beam_seeds):
        top_sequences = _top_beam_sequences(
            seq_env=seq_env,
            one_step_env=one_step_env,
            seed=seed,
            beam_width=cfg.warm_start_beam_width,
            top_k=cfg.warm_start_beam_top_k,
        )
        used_seeds.append(seed)
        filtered_sequences = [
            row
            for row in top_sequences
            if row["portfolio_loss"] >= cfg.warm_start_min_portfolio_loss
        ]
        if not filtered_sequences:
            filtered_sequences = top_sequences[:1]
            filtered_out_sequences += max(0, len(top_sequences) - len(filtered_sequences))
        else:
            filtered_out_sequences += max(0, len(top_sequences) - len(filtered_sequences))

        for row in filtered_sequences:
            obs, _ = seq_env.reset(seed=seed)
            weight = max(float(row["reward"]), 1e-6)
            total_reward = 0.0
            total_portfolio_loss = 0.0
            for action in row["actions"]:
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.int64).copy())
                sample_weights.append(weight)
                obs, reward, terminated, truncated, info = seq_env.step(action)
                total_reward += float(reward)
                br = info.get("reward_breakdown")
                if br is not None:
                    total_portfolio_loss = float(br.portfolio_loss)
                if terminated or truncated:
                    break
            seq_rewards.append(total_reward)
            seq_portfolio_losses.append(total_portfolio_loss)
            kept_sequences += 1

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "sequence_rewards": seq_rewards,
        "sequence_portfolio_losses": seq_portfolio_losses,
        "seeds": used_seeds,
        "kept_sequences": kept_sequences,
        "filtered_out_sequences": filtered_out_sequences,
    }


def _behavior_clone_from_beam(model: PPO, dataset: Dict[str, Any], cfg: TrainConfig) -> Dict[str, Any]:
    def _clone_arrays(
        obs_np: np.ndarray,
        acts_np: np.ndarray,
        weights_np: np.ndarray,
        label: str,
    ) -> Dict[str, Any]:
        if len(obs_np) == 0:
            return {"ran": False, "reason": f"empty {label} dataset", "label": label}

        device = model.policy.device
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        acts_t = torch.as_tensor(acts_np, dtype=torch.long, device=device)
        weights_t = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
        nvec = list(model.action_space.nvec)

        opt = torch.optim.Adam(model.policy.parameters(), lr=cfg.warm_start_lr)
        batch_size = max(1, min(cfg.warm_start_batch_size, len(obs_np)))
        idx = np.arange(len(obs_np))
        losses: List[float] = []
        accs: List[float] = []

        for _ in range(cfg.warm_start_epochs):
            np.random.shuffle(idx)
            batch_losses: List[float] = []
            batch_accs: List[float] = []
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                b_obs = obs_t[batch_idx]
                b_act = acts_t[batch_idx]
                b_w = weights_t[batch_idx]

                features = model.policy.extract_features(b_obs)
                latent_pi, _ = model.policy.mlp_extractor(features)
                logits = model.policy.action_net(latent_pi)
                splits = torch.split(logits, nvec, dim=1)

                losses_i = []
                correct = 0
                counted = 0
                for dim, n_classes in enumerate(nvec):
                    if n_classes <= 1:
                        continue
                    raw_loss = F.cross_entropy(
                        splits[dim],
                        b_act[:, dim],
                        reduction="none",
                    )
                    loss_i = (raw_loss * b_w).mean()
                    losses_i.append(loss_i)
                    preds = torch.argmax(splits[dim], dim=1)
                    correct += int((preds == b_act[:, dim]).sum().item())
                    counted += int(b_act[:, dim].shape[0])

                if not losses_i:
                    continue

                loss = torch.stack(losses_i).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

                batch_losses.append(float(loss.item()))
                batch_accs.append(float(correct / max(counted, 1)))

            if batch_losses:
                losses.append(float(np.mean(batch_losses)))
                accs.append(float(np.mean(batch_accs)))

        return {
            "ran": True,
            "label": label,
            "n_examples": int(len(obs_np)),
            "loss_start": float(losses[0]) if losses else None,
            "loss_end": float(losses[-1]) if losses else None,
            "acc_start": float(accs[0]) if accs else None,
            "acc_end": float(accs[-1]) if accs else None,
        }

    seq_records = dataset["meta"].get("sequence_records", [])
    summary: Dict[str, Any] = {
        "ran": False,
        "n_sequences": int(len(seq_records)),
        "mean_teacher_reward": float(
            np.mean([r["reward"] for r in seq_records])
        ) if seq_records else 0.0,
        "mean_teacher_portfolio_loss": float(
            np.mean([r["portfolio_loss"] for r in seq_records])
        ) if seq_records else 0.0,
        "kept_sequences": int(dataset["meta"].get("kept_sequences", len(seq_records))),
        "filtered_out_sequences": int(dataset["meta"].get("filtered_out_sequences", 0)),
        "teacher_style": dataset["meta"].get("teacher_style", "flat"),
    }

    if (
        "first_observations" in dataset
        and "second_observations" in dataset
        and len(dataset["first_observations"]) > 0
    ):
        first_summary = _clone_arrays(
            dataset["first_observations"],
            dataset["first_actions"],
            dataset["first_weights"],
            label="first_action",
        )
        second_summary = _clone_arrays(
            dataset["second_observations"],
            dataset["second_actions"],
            dataset["second_weights"],
            label="second_action",
        )
        summary.update(
            {
                "ran": bool(first_summary.get("ran") or second_summary.get("ran")),
                "n_examples": int(
                    len(dataset["first_observations"]) + len(dataset["second_observations"])
                ),
                "first_stage": first_summary,
                "second_stage": second_summary,
                "loss_start": first_summary.get("loss_start"),
                "loss_end": second_summary.get("loss_end", first_summary.get("loss_end")),
                "acc_start": first_summary.get("acc_start"),
                "acc_end": second_summary.get("acc_end", first_summary.get("acc_end")),
            }
        )
        return summary

    obs_np = dataset["observations"]
    acts_np = dataset["actions"]
    weights_np = dataset["sample_weights"]
    flat_summary = _clone_arrays(obs_np, acts_np, weights_np, label="flat")
    summary.update(flat_summary)
    return summary


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(cfg: TrainConfig) -> Dict[str, Any]:
    # ---- Real-mode pre-flight check ----
    if cfg.mode == "real":
        from generative_engine_rl.real_mode_loader import (
            diagnose_real_mode_readiness,
            print_diagnosis,
        )
        report = diagnose_real_mode_readiness()
        all_ready = (
            report["production_imports"]
            and report["db_connection"]
            and report["processed_data_present"]
            and report["regimes_present"]
            and report["canonical_graph_present"]
        )
        if not all_ready:
            print_diagnosis(report)
            raise RuntimeError(
                "Real-mode prerequisites not satisfied. See diagnosis above. "
                "Run `python -m generative_engine_rl.diagnose_real_mode` for "
                "more detail, or use --mode fast for stub-based training."
            )

    # ---- Setup ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.run_name}_{timestamp}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PPO training run: {run_name}")
    print(f"  output:  {run_dir}")
    print(f"  mode:    {cfg.mode}")
    print(f"  portfolio: {cfg.portfolio_profile}")
    print(f"  actions/episode: {cfg.actions_per_episode}")
    print(f"  n_envs:  {cfg.n_envs}")
    print(f"  steps:   {cfg.total_timesteps}")
    print(f"  seed:    {cfg.seed}")
    print(f"{'='*60}\n")

    set_random_seed(cfg.seed)

    # ---- Save config snapshot ----
    cfg_dict = asdict(cfg)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    # ---- Build envs ----
    print("Building training env...")
    train_env = make_vec_env_local(cfg)
    print(f"  {cfg.n_envs} parallel envs, action space: {train_env.action_space}")

    print("Building evaluation env...")
    eval_env_inner = make_env(
        mode=cfg.mode,
        seed=cfg.seed + 10_000,
        n_magnitude_bins=cfg.n_magnitude_bins,
        portfolio_profile=cfg.portfolio_profile,
        actions_per_episode=cfg.actions_per_episode,
        use_family_templates=cfg.use_family_templates,
        reward_mode=cfg.reward_mode,
    )
    eval_env = DummyVecEnv([lambda: Monitor(eval_env_inner)])

    # ---- Build PPO ----
    tb_dir = run_dir / "tb" if cfg.tensorboard else None

    # Auto-disable tensorboard if package is not installed
    if tb_dir is not None:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            print("  tensorboard not installed; disabling Tensorboard logging")
            tb_dir = None

    if cfg.resume_from:
        print(f"Resuming from {cfg.resume_from}")
        model = PPO.load(cfg.resume_from, env=train_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            tensorboard_log=str(tb_dir) if tb_dir else None,
            seed=cfg.seed,
            verbose=1,
        )

    warm_start_summary = None
    if cfg.warm_start_beam:
        print("\nCollecting beam-teacher sequences for warm start...")
        dataset_path = (
            Path(cfg.warm_start_dataset_path)
            if cfg.warm_start_dataset_path
            else run_dir / "teacher_dataset.npz"
        )
        if cfg.warm_start_dataset_path and dataset_path.exists():
            teacher_dataset = load_teacher_dataset(dataset_path)
            print(f"  loaded teacher dataset: {dataset_path}")
        else:
            if cfg.actions_per_episode > 1:
                teacher_dataset = build_stagewise_teacher_dataset(
                    mode=cfg.mode,
                    seed=cfg.seed,
                    n_magnitude_bins=cfg.n_magnitude_bins,
                    portfolio_profile=cfg.portfolio_profile,
                    actions_per_episode=cfg.actions_per_episode,
                    use_family_templates=cfg.use_family_templates,
                    reward_mode=cfg.reward_mode,
                    seed_start=cfg.warm_start_seed_start,
                    n_seeds=cfg.warm_start_beam_seeds,
                    beam_width=cfg.warm_start_beam_width,
                    top_k=cfg.warm_start_beam_top_k,
                    min_portfolio_loss=cfg.warm_start_min_portfolio_loss,
                    use_crisis_seeds=cfg.warm_start_use_crisis_seeds,
                )
            else:
                teacher_dataset = build_beam_teacher_dataset(
                    mode=cfg.mode,
                    seed=cfg.seed,
                    n_magnitude_bins=cfg.n_magnitude_bins,
                    portfolio_profile=cfg.portfolio_profile,
                    actions_per_episode=cfg.actions_per_episode,
                    use_family_templates=cfg.use_family_templates,
                    reward_mode=cfg.reward_mode,
                    seed_start=cfg.warm_start_seed_start,
                    n_seeds=cfg.warm_start_beam_seeds,
                    beam_width=cfg.warm_start_beam_width,
                    top_k=cfg.warm_start_beam_top_k,
                    min_portfolio_loss=cfg.warm_start_min_portfolio_loss,
                    use_crisis_seeds=cfg.warm_start_use_crisis_seeds,
                )
            save_teacher_dataset(teacher_dataset, dataset_path)
            print(f"  saved teacher dataset: {dataset_path}")
        seq_records = teacher_dataset["meta"].get("sequence_records", [])
        print(
            f"  teacher examples: {len(teacher_dataset['observations'])} "
            f"from {len(seq_records)} kept sequences "
            f"(filtered {teacher_dataset['meta'].get('filtered_out_sequences', 0)}) | "
            f"mean teacher reward "
            f"{np.mean([r['reward'] for r in seq_records]):+.4f} | "
            f"mean teacher P/L "
            f"{np.mean([r['portfolio_loss'] for r in seq_records]):+.4f}"
        )
        if teacher_dataset["meta"].get("teacher_style"):
            print(f"  teacher style: {teacher_dataset['meta']['teacher_style']}")
        print("Running behavior-cloning warm start...")
        warm_start_summary = _behavior_clone_from_beam(model, teacher_dataset, cfg)
        if warm_start_summary.get("ran"):
            if warm_start_summary.get("first_stage") and warm_start_summary.get("second_stage"):
                fs = warm_start_summary["first_stage"]
                ss = warm_start_summary["second_stage"]
                print(
                    f"  first-stage BC loss {fs['loss_start']:.4f} -> {fs['loss_end']:.4f} | "
                    f"acc {fs['acc_start']:.3f} -> {fs['acc_end']:.3f}"
                )
                print(
                    f"  second-stage BC loss {ss['loss_start']:.4f} -> {ss['loss_end']:.4f} | "
                    f"acc {ss['acc_start']:.3f} -> {ss['acc_end']:.3f}"
                )
            else:
                print(
                    f"  BC loss {warm_start_summary['loss_start']:.4f} -> "
                    f"{warm_start_summary['loss_end']:.4f} | "
                    f"acc {warm_start_summary['acc_start']:.3f} -> "
                    f"{warm_start_summary['acc_end']:.3f}"
                )

    # ---- Random baseline (BEFORE training) ----
    print("\nRandom-policy baseline...")
    random_eval = run_random_baseline(cfg, eval_env, cfg.smoke_test_episodes)
    print(f"  random mean reward: {random_eval['mean']:+.4f} ± {random_eval['std']:.4f}")

    # ---- Callbacks ----
    callbacks: List[BaseCallback] = []
    learning_signals = LearningSignalsCallback()
    callbacks.append(learning_signals)

    if cfg.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
                save_path=str(run_dir),
                name_prefix="ckpt",
            )
        )
    if cfg.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir),
                eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
                verbose=0,
            )
        )

    # ---- Train ----
    print(f"\nTraining for {cfg.total_timesteps} timesteps...")
    t0 = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo",
        reset_num_timesteps=cfg.resume_from is None,
    )
    train_secs = time.time() - t0
    print(f"  trained in {train_secs:.1f}s ({cfg.total_timesteps / max(train_secs, 0.001):.0f} steps/s)")

    # ---- Save final ----
    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    print(f"  saved final model: {final_path}")

    # ---- Trained-policy evaluation ----
    print("\nEvaluating trained policy...")
    trained_eval = evaluate_policy_episodes(
        model, eval_env, cfg.smoke_test_episodes, deterministic=True
    )
    print(f"  trained mean reward: {trained_eval['mean']:+.4f} ± {trained_eval['std']:.4f}")
    print(f"  reward range:        [{trained_eval['min']:+.4f}, {trained_eval['max']:+.4f}]")
    print(f"  rejected actions:    {trained_eval['rejections']}/{trained_eval['n_episodes']}")

    # ---- Smoke-test verdict ----
    # Two signal sources: (1) trajectory of training metrics, (2) trained vs
    # random eval comparison. Pass if EITHER source clearly indicates learning.
    learning_diag = learning_signals.diagnose_learning()
    improvement = trained_eval["mean"] - random_eval["mean"]
    random_floor = random_eval["mean"] + 0.5 * max(random_eval["std"], 1e-6)
    beat_floor = trained_eval["mean"] > random_floor

    print()
    print("=" * 60)
    print("Learning diagnosis")
    print("=" * 60)
    print(f"  rollouts captured:    {learning_diag.get('n_rollouts')}  "
          f"(clean: {learning_diag.get('n_clean_rollouts', '?')})")

    if "reward_available" in learning_diag and learning_diag["reward_available"]:
        rt = learning_diag["reward_trend"]
        sign = "+" if rt >= 0 else ""
        print(f"  reward trend:         {sign}{rt:.4f}  "
              f"({learning_diag['first_quartile_reward']:.4f} -> "
              f"{learning_diag['last_quartile_reward']:.4f})  "
              f"{'IMPROVED' if learning_diag['reward_improved'] else 'flat/down'}")
    elif "reward_available" in learning_diag:
        print(f"  reward trend:         (unavailable - Monitor wrapper not "
              f"populating ep_rew_mean buffer)")

    if "entropy_trend" in learning_diag:
        et = learning_diag["entropy_trend"]
        sign = "+" if et >= 0 else ""
        print(f"  entropy trend:        {sign}{et:.4f}  "
              f"({learning_diag['first_quartile_entropy']:.4f} -> "
              f"{learning_diag['last_quartile_entropy']:.4f})  "
              f"{'POLICY CONCENTRATING' if learning_diag['entropy_decreased'] else 'no change'}")
        print(f"  max clip_fraction:    {learning_diag['max_clip_fraction']:.4f}  "
              f"{'UPDATES NONTRIVIAL' if learning_diag['clipping_active'] else 'no updates'}")
        print(f"  signals passed:       {learning_diag['signals_passed']}/{learning_diag['signals_total']}")

    print(f"  random eval mean:     {random_eval['mean']:+.4f} ± {random_eval['std']:.4f}")
    print(f"  trained eval mean:    {trained_eval['mean']:+.4f} ± {trained_eval['std']:.4f}")
    print(f"  eval improvement:     {improvement:+.4f}  "
          f"{'> 0.5σ floor' if beat_floor else '<= 0.5σ floor'}")
    print()

    # Multi-signal verdict
    training_signals_pass = learning_diag.get("verdict") == "PASS"
    converged = trained_eval["std"] < random_eval["std"] * 0.1
    eval_pass = beat_floor or converged
    smoke_passed = training_signals_pass or eval_pass

    # Build a precise reason string
    pass_reasons = []
    if training_signals_pass:
        pass_reasons.append("training signals confirm learning")
    if beat_floor:
        pass_reasons.append("eval mean beats random by >0.5σ")
    elif converged:
        pass_reasons.append(f"trained policy converged (std {trained_eval['std']:.4f} << random std {random_eval['std']:.4f})")

    if smoke_passed:
        reason = " AND ".join(pass_reasons) if pass_reasons else "passed (no specific signal)"
    else:
        reason = "no learning signal in training trajectory or eval comparison"

    verdict = "PASS" if smoke_passed else "FAIL"
    print(f"Smoke test: {reason}  [{verdict}]")

    # ---- Persist eval results ----
    eval_summary = {
        "run_name": run_name,
        "config": cfg_dict,
        "training_seconds": train_secs,
        "random_baseline": random_eval,
        "trained_policy": trained_eval,
        "improvement_over_random": improvement,
        "smoke_test_passed": smoke_passed,
        "learning_diagnosis": learning_diag,
        "training_signal_history": learning_signals.history,
        "warm_start_summary": warm_start_summary,
    }
    (run_dir / "eval_results.json").write_text(json.dumps(eval_summary, indent=2, default=str))

    return eval_summary


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="ppo_adversarial")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["fast", "real"], default="real")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-magnitude-bins", type=int, default=21)
    p.add_argument("--portfolio-profile", default=DEFAULT_PORTFOLIO_PROFILE)
    p.add_argument("--actions-per-episode", type=int, default=1)
    p.add_argument("--no-family-templates", action="store_true",
                   help="Disable family templates (default: templates ON)")
    p.add_argument(
        "--reward-mode",
        choices=["portfolio_adversarial", "regulatory_adversarial"],
        default="portfolio_adversarial",
    )
    p.add_argument("--warm-start-beam", action="store_true")
    p.add_argument("--warm-start-dataset-path", default=None)
    p.add_argument("--warm-start-seed-start", type=int, default=10_000)
    p.add_argument("--warm-start-beam-seeds", type=int, default=12)
    p.add_argument("--warm-start-beam-width", type=int, default=6)
    p.add_argument("--warm-start-beam-top-k", type=int, default=3)
    p.add_argument("--warm-start-min-portfolio-loss", type=float, default=0.10)
    p.add_argument("--no-crisis-seeds", action="store_true",
                   help="Disable historical crisis seed injection for warm start (default: ON)")
    p.add_argument("--train-crisis-seed-prob", type=float, default=0.5,
                   help="Fraction of PPO training episodes that start from a historical crisis state (default: 0.5)")
    p.add_argument("--warm-start-epochs", type=int, default=8)
    p.add_argument("--warm-start-batch-size", type=int, default=32)
    p.add_argument("--warm-start-lr", type=float, default=1e-3)
    p.add_argument("--use-subproc", action="store_true")
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--checkpoint-freq", type=int, default=10_000)
    p.add_argument("--smoke-test-episodes", type=int, default=50)
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()

    return TrainConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        seed=args.seed,
        mode=args.mode,
        n_envs=args.n_envs,
        n_magnitude_bins=args.n_magnitude_bins,
        portfolio_profile=args.portfolio_profile,
        actions_per_episode=args.actions_per_episode,
        use_family_templates=not args.no_family_templates,
        reward_mode=args.reward_mode,
        warm_start_beam=args.warm_start_beam,
        warm_start_dataset_path=args.warm_start_dataset_path,
        warm_start_seed_start=args.warm_start_seed_start,
        warm_start_beam_seeds=args.warm_start_beam_seeds,
        warm_start_beam_width=args.warm_start_beam_width,
        warm_start_beam_top_k=args.warm_start_beam_top_k,
        warm_start_min_portfolio_loss=args.warm_start_min_portfolio_loss,
        warm_start_use_crisis_seeds=not args.no_crisis_seeds,
        train_crisis_seed_prob=args.train_crisis_seed_prob,
        warm_start_epochs=args.warm_start_epochs,
        warm_start_batch_size=args.warm_start_batch_size,
        warm_start_lr=args.warm_start_lr,
        use_subproc=args.use_subproc,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        tensorboard=not args.no_tensorboard,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        smoke_test_episodes=args.smoke_test_episodes,
        resume_from=args.resume_from,
    )


def main():
    cfg = parse_args()
    summary = train(cfg)
    print(f"\n{'='*60}")
    print(f"Training complete. Results: {summary['smoke_test_passed']}")
    print(f"{'='*60}")
    sys.exit(0 if summary["smoke_test_passed"] else 1)


if __name__ == "__main__":
    main()
