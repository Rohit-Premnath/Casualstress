"""
exhaustive_search.py
====================
Evaluate ALL |A|^2 two-step sequences for each held-out seed to compute
the true global-maximum adversarial reward for the bond_heavy profile.

Why this matters:
    The existing beam-search reference (beam_width=6) explores only
    top-6^2 = 36 of 62,500 possible 2-step sequences.
    Reviewer critique: the denominator is not the global optimum.
    This script computes the exhaustive upper bound so the paper can report
    bandit quality relative to the true worst case, not a truncated search.

Method:
    For each seed, evaluate all n_actions^2 sequences with the 2-step env
    using the same run_sequence() call that beam search uses for its final
    candidate scoring.  The per-seed maximum is the true worst-case reward
    discoverable by any deterministic 2-step policy.

    Alongside exhaustive, the script re-computes beam (beam_width=6) and
    bandit-greedy in the same environment, so all three metrics are
    internally consistent.

Profile:  bond_heavy  (the only profile above the 85% deployment gate)
Seeds:    20000-20015 (16 held-out seeds, identical to heldout_results.json)
Output:   runs/bandit_v2_bond_heavy/heldout_results_exhaustive.json
Partial:  runs/bandit_v2_bond_heavy/exhaustive_partial/seed_XXXXX.json

Run from ml_pipeline directory:
    python exhaustive_search.py [--profile bond_heavy] [--dry-run]
Or from Docker:
    docker exec causalstress-api bash -c "cd /ml_pipeline && python exhaustive_search.py"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.env_factory import make_env
from generative_engine_rl.neural_bandit import (
    BanditRewardNet,
    bandit_sequence,
    build_catalog_tensor,
)


# ---------------------------------------------------------------------------
# Helpers inlined from sequence_compare.py to avoid stable_baselines3 import
# ---------------------------------------------------------------------------

def _action_catalog_list(env) -> List[np.ndarray]:
    """Return a list of every valid (target, family, magnitude) action tuple."""
    nvec = env.action_space.nvec
    catalog: List[np.ndarray] = []
    for t in range(int(nvec[0])):
        for f in range(int(nvec[1])):
            for m in range(int(nvec[2])):
                catalog.append(np.array([t, f, m], dtype=np.int64))
    return catalog


def run_sequence(env, actions: List[np.ndarray], seed: int) -> Dict[str, Any]:
    """Reset env to seed, execute each action in order, return result dict."""
    obs, reset_info = env.reset(seed=seed)
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


def sequence_to_str(record: Dict[str, Any]) -> str:
    return " -> ".join(
        f"{step['target_var']} {step['magnitude']:+.2f}σ"
        for step in record["sequence"]
    )


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


def beam_best_sequence(
    seq_env,
    one_step_env,
    catalog_list: List[np.ndarray],
    seed: int,
    beam_width: int = 6,
) -> Dict[str, Any]:
    """
    Beam search: score all 250 actions at step 1 using one_step_env,
    take the top beam_width, then evaluate all beam_width^2 combinations
    at step 2 using seq_env (2-step env).  Mirrors sequence_compare.py.
    """
    scored: List[Tuple[float, np.ndarray]] = []
    for action in catalog_list:
        result = run_sequence(one_step_env, [action], seed)
        scored.append((result["reward"], action))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [a for _, a in scored[:beam_width]]

    best: Optional[Dict[str, Any]] = None
    for a1 in candidates:
        for a2 in candidates:
            result = run_sequence(seq_env, [a1, a2], seed)
            if best is None or result["reward"] > best["reward"]:
                best = result
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------

def exhaustive_best_sequence(
    env_2step,
    action1_catalog: List[np.ndarray],
    action2_catalog: List[np.ndarray],
    seed: int,
    t_overall_start: float,
    seed_index: int,
    n_seeds_total: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate all len(action1_catalog) x len(action2_catalog) two-step
    sequences for one seed and return the best.

    In full mode:  action1_catalog == action2_catalog == full 250-action catalog.
    In dry-run:    action1_catalog is truncated to 5; action2_catalog is full 250.
    """
    n1, n2 = len(action1_catalog), len(action2_catalog)
    best: Optional[Dict[str, Any]] = None
    all_rewards: List[float] = []
    t_seed = time.time()

    for i, action_1 in enumerate(action1_catalog):
        if i % 25 == 0 and i > 0:
            elapsed = time.time() - t_seed
            eta_seed = (elapsed / i) * (n1 - i)
            elapsed_total = time.time() - t_overall_start
            eta_total = (
                (elapsed_total / seed_index) * (n_seeds_total - seed_index)
                if seed_index > 0 else float("nan")
            )
            print(
                f"    a1 {i+1:3d}/{n1}  "
                f"seed_elapsed={elapsed:.0f}s  seed_ETA={eta_seed:.0f}s  "
                f"total_ETA={eta_total:.0f}s",
                flush=True,
            )

        for action_2 in action2_catalog:
            result = run_sequence(env_2step, [action_1, action_2], seed)
            all_rewards.append(result["reward"])
            if best is None or result["reward"] > best["reward"]:
                best = result

    arr = np.asarray(all_rewards, dtype=np.float64)
    per_seed_stats = {
        "seed": seed,
        "n_sequences": len(all_rewards),
        "max_reward": float(arr.max()),
        "mean_reward_all_seqs": float(arr.mean()),
        "std_reward_all_seqs": float(arr.std()),
        "best_sequence": sequence_to_str(best),  # type: ignore[arg-type]
        "elapsed_s": float(time.time() - t_seed),
    }
    return best, per_seed_stats  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Exhaustive 2-step search — true global max for reviewer response"
    )
    p.add_argument("--profile", default="bond_heavy")
    p.add_argument(
        "--run-dir", default=None,
        help="Override run directory (default: runs/bandit_v2_{profile})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Run only the first 5 action_1s × all 250 action_2s per seed "
            "(1,250 sequences/seed) to verify correctness and estimate runtime"
        ),
    )
    args = p.parse_args()

    runs_dir = ROOT / "runs"
    run_dir = Path(args.run_dir) if args.run_dir else runs_dir / f"bandit_v2_{args.profile}"

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        print(f"[ERROR] config.json not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text())
    mode              = cfg["mode"]
    n_magnitude_bins  = cfg["n_magnitude_bins"]
    reward_mode       = cfg["reward_mode"]
    seed_start        = cfg["eval_seed_start"]
    n_seeds           = cfg["n_eval_seeds"]
    beam_width        = cfg["beam_width"]          # 6 → 36 sequences

    partial_dir = run_dir / "exhaustive_partial"
    partial_dir.mkdir(exist_ok=True)

    # ── Load stored results for informational comparison ─────────────────────
    stored: Optional[Dict] = None
    stored_path = run_dir / "heldout_results.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())

    print(f"\n{'='*72}")
    print(f"  Exhaustive 2-step search  [{args.profile}]")
    print(f"  run_dir:  {run_dir}")
    print(f"  mode={mode}  |  reward_mode={reward_mode}  |  n_magnitude_bins={n_magnitude_bins}")
    print(f"  seeds: {seed_start}..{seed_start + n_seeds - 1}  (n={n_seeds})")
    print(f"  beam_width={beam_width}  ({beam_width**2} sequences per seed)")
    if args.dry_run:
        print("  MODE: DRY-RUN (first 5 action_1s × all 250 action_2s per seed)")
    else:
        print(f"  MODE: FULL EXHAUSTIVE ({250}×{250}=62,500 sequences per seed)")
    if stored:
        print(f"\n  Stored heldout_results.json (informational, NOT used as reference):")
        print(f"    bandit_greedy  = {stored['bandit_greedy']['mean_reward']:.6f}")
        print(f"    beam           = {stored['beam']['mean_reward']:.6f}")
        print(f"    bandit vs beam = {stored.get('bandit_greedy_vs_beam_pct', float('nan')):.2f}%")
    print(f"{'='*72}\n")

    # ── Create environments ─────────────────────────────────────────────────
    make_env_kwargs = dict(
        mode=mode,
        seed=0,
        n_magnitude_bins=n_magnitude_bins,
        portfolio_profile=args.profile,
        reward_mode=reward_mode,
    )
    env_2step   = make_env(actions_per_episode=2, **make_env_kwargs)
    env_1step   = make_env(actions_per_episode=1, **make_env_kwargs)

    full_catalog = _action_catalog_list(env_2step)
    n_actions = len(full_catalog)

    action1_catalog = full_catalog[:5] if args.dry_run else full_catalog
    action2_catalog = full_catalog   # always the full 250

    n_sequences_per_seed = len(action1_catalog) * n_actions
    total_sequences = n_sequences_per_seed * n_seeds

    print(f"  n_actions:           {n_actions}")
    print(f"  sequences per seed:  {n_sequences_per_seed:,}  "
          f"({'5×250 dry-run' if args.dry_run else f'{n_actions}×{n_actions} exhaustive'})")
    print(f"  total sequences:     {total_sequences:,}")
    print()

    # ── Load bandit ─────────────────────────────────────────────────────────
    bandit_path = run_dir / "bandit.pt"
    if not bandit_path.exists():
        print(f"[ERROR] bandit.pt not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    net = BanditRewardNet.load(str(bandit_path))
    catalog_t = build_catalog_tensor(env_2step)

    # ── Per-seed loop ───────────────────────────────────────────────────────
    seeds = list(range(seed_start, seed_start + n_seeds))
    bandit_records:    List[Dict] = []
    beam_records:      List[Dict] = []
    exhaustive_records: List[Dict] = []
    exhaustive_per_seed_stats: List[Dict] = []

    t_overall = time.time()

    for seed_idx, seed in enumerate(seeds):
        partial_path = partial_dir / f"seed_{seed:05d}.json"

        # ── Resume from checkpoint (skip on dry-run) ─────────────────────
        if partial_path.exists() and not args.dry_run:
            cached = json.loads(partial_path.read_text())
            print(
                f"[seed {seed}  ({seed_idx+1}/{n_seeds})]  "
                f"loaded from cache  "
                f"exhaustive_best={cached['exhaustive']['reward']:.4f}  "
                f"bandit={cached['bandit']['reward']:.4f}",
                flush=True,
            )
            bandit_records.append(cached["bandit"])
            beam_records.append(cached["beam"])
            exhaustive_records.append(cached["exhaustive"])
            exhaustive_per_seed_stats.append(cached["stats"])
            continue

        print(
            f"\n[seed {seed}  ({seed_idx+1}/{n_seeds})]  "
            f"total elapsed={time.time()-t_overall:.0f}s",
            flush=True,
        )

        # ── Bandit (greedy) ───────────────────────────────────────────────
        bandit_rec = bandit_sequence(net, catalog_t, env_2step, seed, ucb_beta=0.0)
        print(
            f"  bandit greedy:  {sequence_to_str(bandit_rec)}  "
            f"reward={bandit_rec['reward']:.4f}",
            flush=True,
        )

        # ── Beam (recomputed in same environment) ─────────────────────────
        beam_rec = beam_best_sequence(env_2step, env_1step, full_catalog, seed, beam_width)
        print(
            f"  beam (w={beam_width}):    {sequence_to_str(beam_rec)}  "
            f"reward={beam_rec['reward']:.4f}",
            flush=True,
        )

        # ── Exhaustive ────────────────────────────────────────────────────
        n_to_eval = len(action1_catalog) * n_actions
        print(f"  exhaustive ({n_to_eval:,} sequences)...", flush=True)
        best_record, per_seed_stats = exhaustive_best_sequence(
            env_2step=env_2step,
            action1_catalog=action1_catalog,
            action2_catalog=action2_catalog,
            seed=seed,
            t_overall_start=t_overall,
            seed_index=seed_idx + 1,
            n_seeds_total=n_seeds,
        )
        print(
            f"  exhaustive best: {sequence_to_str(best_record)}  "
            f"reward={best_record['reward']:.4f}  ({per_seed_stats['elapsed_s']:.1f}s)",
            flush=True,
        )

        bandit_records.append(bandit_rec)
        beam_records.append(beam_rec)
        exhaustive_records.append(best_record)
        exhaustive_per_seed_stats.append(per_seed_stats)

        # ── Checkpoint (skip on dry-run) ──────────────────────────────────
        if not args.dry_run:
            partial_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "bandit": bandit_rec,
                        "beam": beam_rec,
                        "exhaustive": best_record,
                        "stats": per_seed_stats,
                    },
                    indent=2,
                    default=str,
                )
            )

    env_2step.close()
    env_1step.close()

    # ── Aggregate ───────────────────────────────────────────────────────────
    bandit_sum     = summarize(bandit_records)
    beam_sum       = summarize(beam_records)
    exhaustive_sum = summarize(exhaustive_records)

    bandit_mean    = bandit_sum["mean_reward"]
    beam_mean      = beam_sum["mean_reward"]
    exhaustive_mean = exhaustive_sum["mean_reward"]

    def pct(num: float, den: float) -> float:
        return 100.0 * num / den if den > 0 else float("nan")

    bandit_vs_beam        = pct(bandit_mean, beam_mean)
    bandit_vs_exhaustive  = pct(bandit_mean, exhaustive_mean)
    beam_vs_exhaustive    = pct(beam_mean, exhaustive_mean)

    top_bandit     = max(bandit_records,    key=lambda r: r["reward"])
    top_beam       = max(beam_records,      key=lambda r: r["reward"])
    top_exhaustive = max(exhaustive_records, key=lambda r: r["reward"])

    # ── Cross-check: compare our bandit and beam against stored values ───────
    cross_check: Dict[str, Any] = {}
    if stored:
        stored_bandit = stored["bandit_greedy"]["mean_reward"]
        stored_beam   = stored["beam"]["mean_reward"]
        delta_bandit  = abs(bandit_mean - stored_bandit)
        delta_beam    = abs(beam_mean - stored_beam)
        tol = 1e-3
        cross_check = {
            "stored_bandit_mean":   stored_bandit,
            "stored_beam_mean":     stored_beam,
            "stored_bandit_vs_beam_pct": stored.get("bandit_greedy_vs_beam_pct"),
            "this_run_bandit_mean": bandit_mean,
            "this_run_beam_mean":   beam_mean,
            "delta_bandit":         delta_bandit,
            "delta_beam":           delta_beam,
            "consistent":           delta_bandit < tol and delta_beam < tol,
        }
        if not cross_check["consistent"]:
            print(
                f"\n[NOTE] This run's bandit/beam differ from stored results.\n"
                f"  bandit:  stored={stored_bandit:.4f}  this_run={bandit_mean:.4f}  "
                f"delta={delta_bandit:.4f}\n"
                f"  beam:    stored={stored_beam:.4f}  this_run={beam_mean:.4f}  "
                f"delta={delta_beam:.4f}\n"
                f"  Most likely cause: DB dataset grew since training (new rows shift\n"
                f"  the seed→historical-state mapping).  All three metrics (bandit,\n"
                f"  beam, exhaustive) in THIS run are computed in the same environment,\n"
                f"  so the bandit-vs-exhaustive ratio is internally consistent.\n"
                f"  The paper should report results from this run, not the stored file.",
                flush=True,
            )
        else:
            print(
                f"\n[OK] Cross-check passed: stored and this-run results match within {tol}.",
                flush=True,
            )

    summary = {
        "profile": args.profile,
        "reward_mode": reward_mode,
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "n_actions": n_actions,
        "n_sequences_per_seed": n_sequences_per_seed,
        "beam_width": beam_width,
        "beam_sequences_per_seed": beam_width ** 2,
        "dry_run": args.dry_run,
        # ── Per-method summaries ──────────────────────────────────────────
        "bandit_greedy": bandit_sum,
        "beam_recomputed": beam_sum,
        "exhaustive": exhaustive_sum,
        # ── Ratios ───────────────────────────────────────────────────────
        "bandit_vs_beam_pct":       bandit_vs_beam,
        "bandit_vs_exhaustive_pct": bandit_vs_exhaustive,
        "beam_vs_exhaustive_pct":   beam_vs_exhaustive,
        # ── Top sequences ─────────────────────────────────────────────────
        "top_bandit_sequence":     sequence_to_str(top_bandit),
        "top_beam_sequence":       sequence_to_str(top_beam),
        "top_exhaustive_sequence": sequence_to_str(top_exhaustive),
        # ── Cross-check ───────────────────────────────────────────────────
        "cross_check": cross_check,
        # ── Per-seed detail ───────────────────────────────────────────────
        "per_seed_stats": exhaustive_per_seed_stats,
        "total_elapsed_s": float(time.time() - t_overall),
    }

    # ── Print final summary ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Exhaustive search {'[DRY-RUN] ' if args.dry_run else ''}complete  [{args.profile}]")
    print(f"  n_seeds={n_seeds}  n_actions={n_actions}  "
          f"sequences/seed={n_sequences_per_seed:,}")
    print()
    print(f"  Bandit-greedy:          {bandit_mean:+.4f}  "
          f"(+/- {bandit_sum['std_reward']:.4f})")
    print(f"  Beam (recomputed w={beam_width}): {beam_mean:+.4f}  "
          f"(+/- {beam_sum['std_reward']:.4f})")
    print(f"  Exhaustive global max:  {exhaustive_mean:+.4f}  "
          f"(+/- {exhaustive_sum['std_reward']:.4f})")
    print()
    print(f"  Bandit vs beam (recomputed):  {bandit_vs_beam:.2f}%")
    print(f"  Bandit vs exhaustive:         {bandit_vs_exhaustive:.2f}%  ← PRIMARY METRIC")
    print(f"  Beam vs exhaustive:           {beam_vs_exhaustive:.2f}%  "
          f"[how beam underestimates true max]")
    if stored:
        print(f"\n  Stored bandit vs beam:        "
              f"{stored.get('bandit_greedy_vs_beam_pct', float('nan')):.2f}%  "
              f"(from heldout_results.json)")
    print()
    print(f"  top bandit:     {sequence_to_str(top_bandit)}")
    print(f"  top beam:       {sequence_to_str(top_beam)}")
    print(f"  top exhaustive: {sequence_to_str(top_exhaustive)}")
    print(f"  total time:     {summary['total_elapsed_s']:.0f}s")
    print(f"{'='*72}\n")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = run_dir / "heldout_results_exhaustive.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved: {out_path}")

    if args.dry_run:
        print(
            "\n  [DRY-RUN] Not representative — only 5 action_1s were evaluated.\n"
            "  Check timing above and run without --dry-run for the full search.\n"
            "  Full run: 62,500 sequences/seed × 16 seeds = 1,000,000 total."
        )


if __name__ == "__main__":
    main()
