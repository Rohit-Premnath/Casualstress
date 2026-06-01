"""
portfolio_comparison.py
=======================
Exhaustive adversarial scan across all four portfolio profiles to find
portfolio-specific worst-case scenarios.

Core product insight: different portfolios have structurally different
worst cases. An equity-heavy book is most vulnerable to market crash
scenarios; a credit-heavy book is most vulnerable to spread widening;
a bond-heavy book is most vulnerable to rate shocks. RL finds these
automatically without a human having to specify which scenarios to test.

Method: for each portfolio, we run the FULL action catalog (all combinations
of target variable x event family x magnitude), evaluate portfolio_loss on
each, and rank them. This is a deterministic exhaustive search — no policy,
no randomness. It directly answers "what is the worst single shock for this
portfolio under this causal model?"

Usage (from ml_pipeline/):
    python -m generative_engine_rl.portfolio_comparison
    python -m generative_engine_rl.portfolio_comparison --n-seeds 3 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.env_factory import make_env
from generative_engine_rl.portfolio_model import PORTFOLIO_PROFILES


# ============================================================================
# EXHAUSTIVE SCAN
# ============================================================================

def exhaustive_scan(
    env,
    n_seeds: int,
    seed_start: int,
) -> List[Dict[str, Any]]:
    """Run every action in the catalog for n_seeds episodes. Return all results."""
    nvec = env.action_space.nvec
    n_targets, n_families, n_mag = int(nvec[0]), int(nvec[1]), int(nvec[2])
    results: List[Dict[str, Any]] = []
    total_actions = n_targets * n_families * n_mag

    for seed_offset in range(n_seeds):
        seed = seed_start + seed_offset * 1000
        obs, _ = env.reset(seed=seed)
        count = 0
        for t in range(n_targets):
            for f in range(n_families):
                for m in range(n_mag):
                    action = np.array([t, f, m], dtype=np.int64)
                    obs_ep, _ = env.reset(seed=seed)
                    obs_ep, reward, terminated, truncated, info = env.step(action)
                    if info.get("action_rejected"):
                        continue
                    br = info.get("reward_breakdown")
                    decoded = dict(env.decode_action(action))
                    results.append({
                        "seed": seed,
                        "action": action.tolist(),
                        "target_var": info.get("target_var", decoded.get("target_var", "?")),
                        "family_name": info.get("family_name", decoded.get("family_name", "?")),
                        "magnitude": float(info.get("magnitude", decoded.get("magnitude", 0.0))),
                        "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
                        "dfast_breach": float(br.dfast_breach) if br else 0.0,
                        "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
                        "total": float(br.total) if br else float(reward),
                    })
                    count += 1
        print(f"    seed={seed}: scanned {count}/{total_actions} valid actions")

    return results


def summarize_top_k(results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Average portfolio_loss across seeds for each (target_var, family, magnitude), return top-k."""
    from collections import defaultdict
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    meta: Dict[tuple, Dict] = {}
    for r in results:
        key = (r["target_var"], r["family_name"], round(r["magnitude"], 2))
        grouped[key].append(r["portfolio_loss"])
        meta[key] = {
            "target_var": r["target_var"],
            "family_name": r["family_name"],
            "magnitude": r["magnitude"],
            "dfast_breach": r["dfast_breach"],
            "causal_fidelity": r["causal_fidelity"],
        }

    scored = [
        {
            **meta[key],
            "portfolio_loss_mean": float(np.mean(vals)),
            "portfolio_loss_max": float(np.max(vals)),
            "n_seeds": len(vals),
        }
        for key, vals in grouped.items()
    ]
    scored.sort(key=lambda x: -x["portfolio_loss_mean"])
    return scored[:k]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="real", choices=["fast", "real"])
    parser.add_argument("--n-seeds", type=int, default=2,
                        help="Seeds per portfolio (more seeds = more reliable ranking)")
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top scenarios to report per portfolio")
    parser.add_argument("--output-dir", default="runs/portfolio_comparison")
    args = parser.parse_args()

    profiles = list(PORTFOLIO_PROFILES.keys())
    print(f"\n{'='*72}")
    print(f"Portfolio Comparison: Adversarial Worst-Case Scan")
    print(f"{'='*72}")
    print(f"  mode:     {args.mode}")
    print(f"  profiles: {profiles}")
    print(f"  n_seeds:  {args.n_seeds}")
    print(f"  top_k:    {args.top_k}")
    print()

    spec = load_spec()
    all_results: Dict[str, Any] = {}
    t0_total = time.time()

    for profile in profiles:
        print(f"\n--- Portfolio: {profile} ---")
        t0 = time.time()
        env = make_env(
            mode=args.mode,
            seed=args.seed_start,
            spec=spec,
            portfolio_profile=profile,
            actions_per_episode=1,
            use_family_templates=True,
            reward_mode="portfolio_adversarial",
        )
        nvec = env.action_space.nvec
        n_actions = int(nvec[0]) * int(nvec[1]) * int(nvec[2])
        print(f"  action space: {int(nvec[0])} targets x {int(nvec[1])} families x {int(nvec[2])} mags = {n_actions} actions")
        print(f"  scanning {n_actions * args.n_seeds} episodes...")

        raw = exhaustive_scan(env, n_seeds=args.n_seeds, seed_start=args.seed_start)
        top = summarize_top_k(raw, k=args.top_k)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

        all_results[profile] = {
            "top_scenarios": top,
            "total_scanned": len(raw),
            "mean_portfolio_loss_all": float(np.mean([r["portfolio_loss"] for r in raw])),
            "max_portfolio_loss_all": float(np.max([r["portfolio_loss"] for r in raw])),
        }

    total_elapsed = time.time() - t0_total
    print(f"\nTotal scan time: {total_elapsed:.1f}s")

    # ---- Print comparison table ----
    print(f"\n{'='*72}")
    print("WORST-CASE SCENARIO BY PORTFOLIO TYPE")
    print(f"{'='*72}")
    print(f"  Each row = worst single shock (target x family x magnitude) for that portfolio.")
    print(f"  P/L = portfolio_loss (higher = worse for the portfolio).")
    print(f"  CF  = causal_fidelity (fraction of causal edges respected).")
    print()

    for profile, data in all_results.items():
        top1 = data["top_scenarios"][0] if data["top_scenarios"] else {}
        print(f"  [{profile}]")
        print(f"    Worst scenario: {top1.get('target_var','?')} x {top1.get('family_name','?')} "
              f"x {top1.get('magnitude', 0):+.1f}")
        print(f"    Portfolio loss: {top1.get('portfolio_loss_mean', 0):.4f} (mean across {top1.get('n_seeds',0)} seeds)")
        print(f"    Causal fidelity: {top1.get('causal_fidelity', 0):.4f}")
        print(f"    DFAST breach:    {top1.get('dfast_breach', 0):.4f} (logged, not in reward)")
        print()

    print(f"--- Top-{args.top_k} per portfolio ---")
    for profile, data in all_results.items():
        print(f"\n  {profile}:")
        print(f"    {'#':<3} {'Target Var':<14} {'Family':<28} {'Mag':>6}  {'P/L mean':>9}  {'CF':>6}")
        print(f"    {'-'*3} {'-'*14} {'-'*28} {'-'*6}  {'-'*9}  {'-'*6}")
        for i, s in enumerate(data["top_scenarios"], 1):
            print(f"    {i:<3} {s['target_var']:<14} {s['family_name']:<28} "
                  f"{s['magnitude']:>+6.1f}  {s['portfolio_loss_mean']:>9.4f}  {s['causal_fidelity']:>6.4f}")

    # ---- Save results ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    out_path.write_text(json.dumps({
        "mode": args.mode,
        "n_seeds": args.n_seeds,
        "top_k": args.top_k,
        "profiles": all_results,
    }, indent=2, default=str))
    print(f"\n  Results saved to: {out_path}")

    # ---- Differentiation analysis ----
    print(f"\n{'='*72}")
    print("DIFFERENTIATION ANALYSIS")
    print(f"{'='*72}")
    worst_vars = {p: d["top_scenarios"][0].get("target_var", "?")
                  for p, d in all_results.items() if d["top_scenarios"]}
    unique_vars = set(worst_vars.values())
    if len(unique_vars) > 1:
        print(f"  PASS: Different portfolios have different worst-case variables.")
        print(f"  {' | '.join(f'{p}={v}' for p, v in worst_vars.items())}")
        print(f"  This confirms the core RL product claim: worst-case is portfolio-specific.")
    else:
        v = list(unique_vars)[0]
        print(f"  NOTE: All portfolios share the same worst variable: {v}")
        # Check if magnitudes or families differ
        worst_fams = {p: d["top_scenarios"][0].get("family_name", "?")
                      for p, d in all_results.items() if d["top_scenarios"]}
        if len(set(worst_fams.values())) > 1:
            print(f"  But event families differ: {' | '.join(f'{p}={f}' for p, f in worst_fams.items())}")
        # Compare portfolio_loss magnitudes
        losses = {p: d["top_scenarios"][0].get("portfolio_loss_mean", 0)
                  for p, d in all_results.items() if d["top_scenarios"]}
        max_p = max(losses, key=losses.get)
        min_p = min(losses, key=losses.get)
        print(f"  Severity differs: {max_p} loses {losses[max_p]:.3f} vs {min_p} loses {losses[min_p]:.3f}")
        print(f"  ({losses[max_p]/max(losses[min_p], 1e-6):.1f}x more severe for {max_p})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
