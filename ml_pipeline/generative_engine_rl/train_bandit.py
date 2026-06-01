"""
train_bandit.py
===============
Neural Contextual Bandit — training and held-out evaluation.

Supports 1-step (--n-steps 1) and 2-step (--n-steps 2) training modes.

1-step mode (bandit v1):
    Collect (obs_0, action, r) for all actions × all training seeds.
    Train f(obs, action) → reward.  At inference: argmax over 250 actions.

2-step mode (bandit v2):
    Step-1 data: (obs_0, action_1, r_1) for all 250 actions × all seeds.
    Step-2 data: (obs_1, action_2, r_2) for all 250 actions × top-K step-1
                 branches × all seeds.  obs_1 is the post-shock state after
                 executing the chosen step-1 action.
    The combined dataset teaches the model to predict immediate reward at
    whichever step it's at (encoded by the progress dim in obs).
    At inference: greedy at step 1, then greedy again from the new obs at step 2.

Usage:
    # 1-step (v1):
    python -m generative_engine_rl.train_bandit \\
        --portfolio-profile balanced --n-steps 1 \\
        --n-train-seeds 50 --n-epochs 500 --out-dir runs/bandit_v1_balanced

    # 2-step (v2):
    python -m generative_engine_rl.train_bandit \\
        --portfolio-profile balanced --n-steps 2 \\
        --n-train-seeds 50 --n-step2-branches 6 \\
        --n-epochs 500 --eval-actions-per-episode 2 \\
        --out-dir runs/bandit_v2_balanced
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.neural_bandit import (
    BanditRewardNet,
    bandit_sequence,
    build_catalog_tensor,
)
from generative_engine_rl.portfolio_model import DEFAULT_PORTFOLIO_PROFILE
from generative_engine_rl.sequence_compare import (
    action_catalog as _action_catalog_list,
    beam_best_sequence,
    heuristic_sequences,
    random_sequence,
    run_sequence,
    sequence_to_str,
    summarize,
)


# ============================================================================
# Data collection
# ============================================================================

def collect_dataset(
    mode: str,
    portfolio_profile: str,
    n_magnitude_bins: int,
    seed_start: int,
    n_train_seeds: int,
    reward_mode: str,
    crisis_seed_prob: float = 0.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    For each training seed: evaluate ALL actions on a 1-step env.

    Returns dict with keys:
        obs     (N, obs_dim)  float32 — observation before the action
        actions (N, 3)        int64   — [target_idx, family_idx, mag_idx]
        rewards (N,)          float32 — immediate reward from env.step
        seeds   (N,)          int64   — which seed each row came from
    where N = n_train_seeds * n_actions.
    """
    t0 = time.time()
    env = make_env(
        mode=mode,
        seed=seed_start,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        reward_mode=reward_mode,
        crisis_seed_prob=crisis_seed_prob,
    )
    catalog_list = _action_catalog_list(env)   # list of (3,) int64 arrays
    n_actions = len(catalog_list)

    obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []
    seed_list: List[int] = []

    for i, seed in enumerate(range(seed_start, seed_start + n_train_seeds)):
        if verbose:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (n_train_seeds - i)
            print(
                f"  [collect] seed {seed}  ({i+1}/{n_train_seeds})  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )

        # Get obs for this seed (deterministic: same seed → same obs always)
        obs_seed, _ = env.reset(seed=seed)

        for action in catalog_list:
            # Reset to clean episode start (same seed → same obs)
            env.reset(seed=seed)
            _, reward, _, _, _ = env.step(action)
            obs_list.append(obs_seed.copy())
            action_list.append(action.copy())
            reward_list.append(float(reward))
            seed_list.append(seed)

    env.close()
    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [collect] done  total_rows={len(obs_list):,}  "
            f"({n_actions} actions × {n_train_seeds} seeds)  elapsed={elapsed:.0f}s",
            flush=True,
        )

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.int64),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "seeds": np.asarray(seed_list, dtype=np.int64),
    }


def collect_dataset_2step(
    mode: str,
    portfolio_profile: str,
    n_magnitude_bins: int,
    seed_start: int,
    n_train_seeds: int,
    reward_mode: str,
    n_step2_branches: int = 6,
    crisis_seed_prob: float = 0.0,
    force_minmag_branches: bool = False,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Collect 2-step training data for BanditRewardNet.

    Step-1 data (N1 = n_train_seeds * n_actions):
        (obs_0, action_1, r_1) — reward landscape from the initial state.

    Step-2 data (N2 = n_train_seeds * n_branches * n_actions):
        (obs_1, action_2, r_2) — reward landscape from the top-K post-shock
        states that follow from the best n_step2_branches step-1 actions,
        plus (if force_minmag_branches=True) one branch per target variable at
        the minimum shock magnitude (0.5σ).

    force_minmag_branches: when True, additionally expands one branch per target
        variable at magnitude index 0 (0.5σ adverse shock), regardless of its
        step-1 reward rank. Exhaustive search shows that tiny-first-shock sequences
        (e.g. BAMLC0A0CM +0.5σ → TLT -5σ) are among the highest joint-reward
        pathways, but they never appear in top-K greedy branches. This flag
        ensures the model sees the post-tiny-shock state in training.
        Adds up to n_targets extra branches per seed (after deduplication with top-K).

    The combined dataset teaches the model to map any obs → best action,
    whether obs is a fresh initial state (step 1) or a post-shock state (step 2).
    The progress dimension in obs encodes which phase the model is evaluating.
    """
    t0 = time.time()

    env_1step = make_env(
        mode=mode, seed=seed_start, n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile, actions_per_episode=1,
        reward_mode=reward_mode, crisis_seed_prob=crisis_seed_prob,
    )
    env_2step = make_env(
        mode=mode, seed=seed_start, n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile, actions_per_episode=2,
        reward_mode=reward_mode, crisis_seed_prob=crisis_seed_prob,
    )
    catalog_list = _action_catalog_list(env_1step)
    n_actions = len(catalog_list)

    # Pre-compute forced min-magnitude branch indices (computed once, reused per seed).
    # catalog_list ordering: target outermost → family → magnitude (innermost).
    # Index for (target=t, family=0, mag=0) = t * n_families * n_mags.
    if force_minmag_branches:
        _nvec = env_1step.action_space.nvec
        _n_tgt = int(_nvec[0])
        _n_fam = int(_nvec[1])
        _n_mag = int(_nvec[2])
        _forced_minmag_idxs: List[int] = [t * _n_fam * _n_mag for t in range(_n_tgt)]
        if verbose:
            print(
                f"  [collect2] force_minmag_branches=True: "
                f"{len(_forced_minmag_idxs)} candidate forced indices "
                f"(one per target at mag_idx=0, 0.5σ adverse shock)",
                flush=True,
            )
    else:
        _forced_minmag_idxs: List[int] = []

    obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []
    seed_list: List[int] = []

    for i, seed in enumerate(range(seed_start, seed_start + n_train_seeds)):
        if verbose:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (n_train_seeds - i)
            print(
                f"  [collect2] seed {seed}  ({i+1}/{n_train_seeds})  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )

        # ── Step-1: evaluate all actions from obs_0 ──────────────────────────
        obs_0, _ = env_1step.reset(seed=seed)
        step1_rewards: List[float] = []

        for action in catalog_list:
            env_1step.reset(seed=seed)
            _, r1, _, _, _ = env_1step.step(action)
            r1 = float(r1)
            step1_rewards.append(r1)
            obs_list.append(obs_0.copy())
            action_list.append(action.copy())
            reward_list.append(r1)
            seed_list.append(seed)

        # ── Step-2: evaluate all actions from each post-step-1 state ────────
        # Start with top-K greedy branches (highest immediate step-1 reward).
        top_k_indices: List[int] = list(np.argsort(step1_rewards)[-n_step2_branches:])

        # Append forced min-magnitude branches (if enabled), skipping any that
        # are already in the greedy top-K to avoid duplicate step-2 evaluations.
        if _forced_minmag_idxs:
            _seen: set = set(top_k_indices)
            for _idx in _forced_minmag_idxs:
                if _idx not in _seen:
                    top_k_indices.append(_idx)
                    _seen.add(_idx)

        for branch_rank, k in enumerate(top_k_indices):
            action_1 = catalog_list[k]
            # Unique context ID per (seed, branch) so BPR groups correctly
            ctx_id = seed + (branch_rank + 1) * n_train_seeds * 100

            for action_2 in catalog_list:
                env_2step.reset(seed=seed)
                obs_1, _, _, _, _ = env_2step.step(action_1)  # reach step-2 state
                _, r2, _, _, _ = env_2step.step(action_2)      # evaluate action_2
                obs_list.append(obs_1.copy())
                action_list.append(action_2.copy())
                reward_list.append(float(r2))
                seed_list.append(int(ctx_id))

    env_1step.close()
    env_2step.close()

    n_total = len(obs_list)
    n_step1 = n_train_seeds * n_actions
    n_step2 = n_total - n_step1
    elapsed = time.time() - t0
    if verbose:
        print(
            f"  [collect2] done  total={n_total:,}  "
            f"step1={n_step1:,}  step2={n_step2:,}  elapsed={elapsed:.0f}s",
            flush=True,
        )

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.int64),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "seeds": np.asarray(seed_list, dtype=np.int64),
    }


# ============================================================================
# Loss functions
# ============================================================================

def bpr_loss(
    pred: torch.Tensor,     # (N,) predicted rewards for a batch
    true: torch.Tensor,     # (N,) true rewards
    seed_ids: torch.Tensor, # (N,) which seed each row belongs to
    top_frac: float = 0.15,
    bot_frac: float = 0.15,
    n_pairs: int = 256,
) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.

    For each unique seed in the batch, sample (pos, neg) pairs where
    pos has higher true reward than neg. Loss: -log sigmoid(pred_pos - pred_neg).
    This trains the model to rank beam-optimal actions above suboptimal ones.
    """
    device = pred.device
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for sid in seed_ids.unique():
        mask = seed_ids == sid
        r_seed = true[mask]
        p_seed = pred[mask]
        n = int(mask.sum())
        if n < 4:
            continue
        k_top = max(1, int(n * top_frac))
        k_bot = max(1, int(n * bot_frac))
        top_idx = r_seed.argsort(descending=True)[:k_top]
        bot_idx = r_seed.argsort(descending=False)[:k_bot]

        # Sample n_pairs random (pos, neg) combinations
        n_sample = min(n_pairs, k_top * k_bot)
        pos_rand = torch.randint(k_top, (n_sample,), device=device)
        neg_rand = torch.randint(k_bot, (n_sample,), device=device)
        pos_scores = p_seed[top_idx[pos_rand]]
        neg_scores = p_seed[bot_idx[neg_rand]]

        diff = pos_scores - neg_scores
        total_loss = total_loss + (-F.logsigmoid(diff)).mean()
        n_terms += 1

    return total_loss / max(n_terms, 1)


def combined_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    seed_ids: torch.Tensor,
    alpha: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """alpha * Huber + (1-alpha) * BPR."""
    l_reg = F.huber_loss(pred, true)
    l_rank = bpr_loss(pred, true, seed_ids)
    l_total = alpha * l_reg + (1.0 - alpha) * l_rank
    return l_total, l_reg, l_rank


# ============================================================================
# Training loop
# ============================================================================

def train_net(
    dataset: Dict[str, np.ndarray],
    n_epochs: int = 300,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 128,
    dropout: float = 0.15,
    alpha: float = 0.6,
    patience: int = 40,
    verbose: bool = True,
) -> BanditRewardNet:
    """Train BanditRewardNet on the collected dataset."""
    obs_np = dataset["obs"]
    actions_np = dataset["actions"]
    rewards_np = dataset["rewards"]
    seeds_np = dataset["seeds"]

    obs_dim = obs_np.shape[1]
    # Infer discrete dims from action data
    n_targets = int(actions_np[:, 0].max()) + 1
    n_families = int(actions_np[:, 1].max()) + 1
    n_mags = int(actions_np[:, 2].max()) + 1

    if verbose:
        print(
            f"  [train] obs_dim={obs_dim}  actions=({n_targets},{n_families},{n_mags})"
            f"  N={len(obs_np):,}  n_epochs={n_epochs}",
            flush=True,
        )

    # Normalise rewards to zero-mean, unit-std (per-dataset, not per-seed)
    r_mean = float(rewards_np.mean())
    r_std = float(rewards_np.std()) + 1e-8
    rewards_norm = (rewards_np - r_mean) / r_std

    # Convert to tensors
    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    act_t = torch.tensor(actions_np, dtype=torch.long)
    rew_t = torch.tensor(rewards_norm, dtype=torch.float32)
    sid_t = torch.tensor(seeds_np, dtype=torch.long)

    net = BanditRewardNet(obs_dim, n_targets, n_families, n_mags, hidden, dropout)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    N = len(obs_t)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        net.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
            obs_b = obs_t[idx]
            t_b = act_t[idx, 0]
            f_b = act_t[idx, 1]
            m_b = act_t[idx, 2]
            r_b = rew_t[idx]
            s_b = sid_t[idx]

            opt.zero_grad()
            pred = net(obs_b, t_b, f_b, m_b)
            loss, _, _ = combined_loss(pred, r_b, s_b, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        sched.step()
        epoch_loss /= max(n_batches, 1)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 50 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  [train] epoch {epoch:4d}/{n_epochs}  loss={epoch_loss:.4f}"
                f"  best={best_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}"
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )

        if no_improve >= patience:
            if verbose:
                print(f"  [train] early stop at epoch {epoch} (patience={patience})", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


# ============================================================================
# Held-out evaluation
# ============================================================================

def run_heldout_eval(
    net: BanditRewardNet,
    mode: str,
    portfolio_profile: str,
    n_magnitude_bins: int,
    reward_mode: str,
    seed_start: int,
    n_seeds: int,
    beam_width: int,
    eval_actions_per_episode: int = 1,
    ucb_beta: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Mirror of heldout_generalization.py — adds bandit_greedy and bandit_ucb.

    eval_actions_per_episode: set to 1 to match PPO v5 benchmark setup exactly;
        set to 2 for the full 2-step adversarial scenario (longer horizon, more
        powerful beam, harder for the bandit).
    """
    seq_env = make_env(
        mode=mode,
        seed=0,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=eval_actions_per_episode,
        reward_mode=reward_mode,
    )
    one_step_env = make_env(
        mode=mode,
        seed=0,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=portfolio_profile,
        actions_per_episode=1,
        reward_mode=reward_mode,
    )
    catalog_t = build_catalog_tensor(seq_env)
    seeds = list(range(seed_start, seed_start + n_seeds))
    heuristics = heuristic_sequences(seq_env, portfolio_profile)
    rng = np.random.default_rng(24680)

    bandit_records: List[Dict] = []
    ucb_records: List[Dict] = []
    random_records: List[Dict] = []
    heuristic_records: List[Dict] = []
    beam_records: List[Dict] = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  [eval] seed {seed}  ({i+1}/{n_seeds})", flush=True)

        bandit_records.append(
            bandit_sequence(net, catalog_t, seq_env, seed, ucb_beta=0.0)
        )
        if ucb_beta > 0.0:
            ucb_records.append(
                bandit_sequence(net, catalog_t, seq_env, seed, ucb_beta=ucb_beta)
            )
        random_records.append(random_sequence(seq_env, rng, seed))

        best_heur = None
        for name, actions in heuristics:
            result = run_sequence(seq_env, actions, seed)
            result["heuristic_name"] = name
            if best_heur is None or result["reward"] > best_heur["reward"]:
                best_heur = result
        heuristic_records.append(best_heur)

        beam_records.append(
            beam_best_sequence(seq_env, one_step_env, seed, beam_width=beam_width)
        )

    summary = {
        "portfolio_profile": portfolio_profile,
        "reward_mode": reward_mode,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "beam_width": beam_width,
        "bandit_greedy": summarize(bandit_records),
        "random": summarize(random_records),
        "heuristic": summarize(heuristic_records),
        "beam": summarize(beam_records),
        "top_bandit_sequence": sequence_to_str(max(bandit_records, key=lambda r: r["reward"])),
        "top_beam_sequence": sequence_to_str(max(beam_records, key=lambda r: r["reward"])),
    }
    if ucb_beta > 0.0:
        summary["bandit_ucb"] = summarize(ucb_records)
        summary["ucb_beta"] = ucb_beta

    # Compute RL/Beam ratios
    beam_mean = summary["beam"]["mean_reward"]
    summary["bandit_greedy_vs_beam_pct"] = (
        100.0 * summary["bandit_greedy"]["mean_reward"] / beam_mean
        if beam_mean > 0 else float("nan")
    )
    if "bandit_ucb" in summary:
        summary["bandit_ucb_vs_beam_pct"] = (
            100.0 * summary["bandit_ucb"]["mean_reward"] / beam_mean
            if beam_mean > 0 else float("nan")
        )

    seq_env.close()
    one_step_env.close()
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    beam = summary["beam"]["mean_reward"]
    greedy = summary["bandit_greedy"]["mean_reward"]
    rand = summary["random"]["mean_reward"]
    heur = summary["heuristic"]["mean_reward"]

    print()
    print("=" * 72)
    print(f"  Bandit held-out benchmark  [{summary['portfolio_profile']}]")
    print("=" * 72)
    print(f"  n_seeds={summary['n_seeds']}  beam_width={summary['beam_width']}")
    print(f"  Bandit-greedy: {greedy:+.4f} +/- {summary['bandit_greedy']['std_reward']:.4f}"
          f"  ({summary.get('bandit_greedy_vs_beam_pct', float('nan')):.1f}% of beam)")
    if "bandit_ucb" in summary:
        ucb = summary["bandit_ucb"]["mean_reward"]
        print(f"  Bandit-UCB:    {ucb:+.4f} +/- {summary['bandit_ucb']['std_reward']:.4f}"
              f"  ({summary.get('bandit_ucb_vs_beam_pct', float('nan')):.1f}% of beam)")
    print(f"  Random:        {rand:+.4f} +/- {summary['random']['std_reward']:.4f}")
    print(f"  Heuristic:     {heur:+.4f} +/- {summary['heuristic']['std_reward']:.4f}")
    print(f"  Beam:          {beam:+.4f} +/- {summary['beam']['std_reward']:.4f}")
    top_b = summary["top_bandit_sequence"].encode("ascii", "replace").decode("ascii")
    top_m = summary["top_beam_sequence"].encode("ascii", "replace").decode("ascii")
    print(f"  top bandit: {top_b}")
    print(f"  top beam:   {top_m}")
    print()


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural Contextual Bandit training")
    p.add_argument("--mode", default="real", choices=["fast", "real"])
    p.add_argument("--portfolio-profile", default=DEFAULT_PORTFOLIO_PROFILE)
    p.add_argument("--n-magnitude-bins", type=int, default=21)
    p.add_argument("--reward-mode", default="portfolio_adversarial")
    # Data collection
    p.add_argument("--n-steps", type=int, default=1, choices=[1, 2],
                   help="1=1-step bandit (v1), 2=2-step bandit (v2, collects step-2 data)")
    p.add_argument("--n-step2-branches", type=int, default=6,
                   help="Top-K step-1 actions to expand at step 2 (only used when --n-steps 2)")
    p.add_argument("--force-minmag-branches", action="store_true", default=False,
                   help="In 2-step mode: additionally expand one branch per target variable at "
                        "the minimum shock magnitude (0.5σ), regardless of step-1 reward rank. "
                        "Exposes the model to small-first-shock pathways that top-K selection "
                        "systematically misses. Adds up to n_targets extra branches per seed "
                        "after deduplication with the top-K greedy set.")
    p.add_argument("--train-seed-start", type=int, default=1000,
                   help="First seed for training data collection (must not overlap "
                        "with warm-start seeds 10000-10011 or eval seeds 20000+)")
    p.add_argument("--n-train-seeds", type=int, default=10)
    # Network
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    # Training
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Weight on Huber loss (1-alpha on BPR ranking loss)")
    p.add_argument("--patience", type=int, default=40)
    # Evaluation
    p.add_argument("--eval-seed-start", type=int, default=20000)
    p.add_argument("--n-eval-seeds", type=int, default=16)
    p.add_argument("--beam-width", type=int, default=6)
    p.add_argument("--eval-actions-per-episode", type=int, default=1,
                   help="Set to 1 to match PPO v5 benchmark (1-step); 2 for full 2-step eval")
    p.add_argument("--ucb-beta", type=float, default=0.5,
                   help="UCB exploration coefficient (0 = greedy only)")
    p.add_argument("--crisis-seed-prob", type=float, default=0.0,
                   help="Probability of sampling a crisis-regime starting state (0=never, 0.5=50%)")
    # Output
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vstr = f"v{args.n_steps}"
    run_name = args.run_name or f"bandit_{vstr}_{args.portfolio_profile}_{ts}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    version = "v2" if args.n_steps == 2 else "v1"
    print(f"\n{'='*72}")
    print(f"  Neural Contextual Bandit — {version} ({args.n_steps}-step)")
    print(f"  portfolio: {args.portfolio_profile}")
    print(f"  train_seeds: {args.train_seed_start}..{args.train_seed_start + args.n_train_seeds - 1}")
    if args.n_steps == 2:
        print(f"  step2_branches: {args.n_step2_branches}  (top-K step-1 actions expanded at step 2)")
        if args.force_minmag_branches:
            print(f"  force_minmag_branches: True  (+ up to 25 forced 0.5σ branches, deduplicated)")
    print(f"  n_magnitude_bins: {args.n_magnitude_bins}  (actual dims resolved at runtime)")
    print(f"  out_dir: {out_dir}")
    print(f"{'='*72}\n")

    # ── Step 1: collect training data (skip if dataset.npz already exists) ──
    dataset_path = out_dir / "dataset.npz"
    if dataset_path.exists():
        print(f"[Step 1/3] Loading cached dataset from {dataset_path}")
        _d = np.load(dataset_path)
        dataset = {k: _d[k] for k in _d.files}
        print(f"  loaded: {dataset['obs'].shape[0]:,} rows")
    elif args.n_steps == 2:
        _branch_desc = f"{args.n_step2_branches} greedy"
        if args.force_minmag_branches:
            _branch_desc += " + up to 25 forced min-mag (0.5σ)"
        print(f"[Step 1/3] Collecting 2-step training data "
              f"({args.n_train_seeds} seeds × [{_branch_desc}] branches × all actions)")
        dataset = collect_dataset_2step(
            mode=args.mode,
            portfolio_profile=args.portfolio_profile,
            n_magnitude_bins=args.n_magnitude_bins,
            seed_start=args.train_seed_start,
            n_train_seeds=args.n_train_seeds,
            reward_mode=args.reward_mode,
            n_step2_branches=args.n_step2_branches,
            crisis_seed_prob=args.crisis_seed_prob,
            force_minmag_branches=args.force_minmag_branches,
            verbose=True,
        )
        np.savez_compressed(dataset_path, **dataset)
    else:
        print("[Step 1/3] Collecting 1-step training data (all actions × all seeds)")
        dataset = collect_dataset(
            mode=args.mode,
            portfolio_profile=args.portfolio_profile,
            n_magnitude_bins=args.n_magnitude_bins,
            seed_start=args.train_seed_start,
            n_train_seeds=args.n_train_seeds,
            reward_mode=args.reward_mode,
            crisis_seed_prob=args.crisis_seed_prob,
            verbose=True,
        )
        np.savez_compressed(dataset_path, **dataset)

    # ── Step 2: train (skip if bandit.pt already exists) ─────────────────────
    model_path = out_dir / "bandit.pt"
    if model_path.exists():
        print(f"\n[Step 2/3] Loading cached model from {model_path}")
        net = BanditRewardNet.load(str(model_path))
        net.eval()
    else:
        print("\n[Step 2/3] Training BanditRewardNet")
        net = train_net(
            dataset=dataset,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
            alpha=args.alpha,
            patience=args.patience,
            verbose=True,
        )
        net.save(
            str(model_path),
            meta={
                "portfolio_profile": args.portfolio_profile,
                "n_train_seeds": args.n_train_seeds,
                "train_seed_start": args.train_seed_start,
                "n_magnitude_bins": args.n_magnitude_bins,
                "reward_mode": args.reward_mode,
                "run_name": run_name,
            },
        )

    # ── Step 3: held-out evaluation (skip if results already exist) ──────────
    out_json = out_dir / "heldout_results.json"
    if out_json.exists():
        print(f"\n[Step 3/3] Skipping eval — results already exist at {out_json}")
        print_summary(json.loads(out_json.read_text()))
        return

    print("\n[Step 3/3] Held-out evaluation (16 seeds, 20000-20015)")
    summary = run_heldout_eval(
        net=net,
        mode=args.mode,
        portfolio_profile=args.portfolio_profile,
        n_magnitude_bins=args.n_magnitude_bins,
        reward_mode=args.reward_mode,
        seed_start=args.eval_seed_start,
        n_seeds=args.n_eval_seeds,
        beam_width=args.beam_width,
        eval_actions_per_episode=args.eval_actions_per_episode,
        ucb_beta=args.ucb_beta,
        verbose=True,
    )

    out_json = out_dir / "heldout_results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print_summary(summary)
    print(f"  saved: {out_json}")

    # Save config
    cfg = vars(args)
    cfg["run_name"] = run_name
    cfg["n_train_total"] = int(dataset["obs"].shape[0])
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
