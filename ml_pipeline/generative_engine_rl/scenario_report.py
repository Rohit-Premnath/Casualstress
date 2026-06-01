"""
scenario_report.py
==================
Production-grade adversarial scenario report for risk managers.

Loads a trained PPO policy and runs it stochastically to produce a ranked
table of diverse worst-case scenarios for a given portfolio. This is the
primary output surface of the RL system — the deliverable handed to a
risk manager or submitted to a regulatory review.

Methodology
-----------
1. Run the trained policy in STOCHASTIC mode for N episodes (default 500).
   Stochastic sampling is deliberate: it reveals the full distribution of
   high-value actions the policy has learned, not just the greedy argmax.
2. Group results by (target_var, event_family, magnitude_bin).
3. Average portfolio_loss and causal_fidelity across seeds per group.
4. Filter to groups seen at least min_occurrences times (statistical noise guard).
5. Rank by mean portfolio_loss descending.
6. Print and save the top-K ranked scenario table.

Why stochastic?
---------------
A deterministic policy collapses to one action (the argmax). For a risk
report, you want a RANKED LIST of distinct adversarial scenarios — each one
representing a different macro regime that causes severe losses. Stochastic
sampling naturally produces this ranked list from the learned distribution.

Usage (from ml_pipeline/):
    python -m generative_engine_rl.scenario_report
    python -m generative_engine_rl.scenario_report --run-dir runs/ppo_v3_diverse_<id>
    python -m generative_engine_rl.scenario_report --n-episodes 1000 --top-k 10
    python -m generative_engine_rl.scenario_report --portfolio-profile tech_heavy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# HELPERS
# ============================================================================

def find_latest_run(base: Path) -> Path:
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "final.zip").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories with final.zip found in {base}")
    return candidates[0]


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def collect_stochastic_episodes(
    model,
    env,
    n_episodes: int,
    seed_start: int = 30000,
) -> List[Dict[str, Any]]:
    """Run policy stochastically, return all non-rejected episode records."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=False)
            action = np.asarray(action, dtype=np.int64).flatten()
            decoded = dict(env.decode_action(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("action_rejected"):
            continue
        br = info.get("reward_breakdown")
        results.append({
            "ep": ep,
            "target_var": info.get("target_var", decoded.get("target_var", "?")),
            "family_name": info.get("family_name", decoded.get("family_name", "?")),
            "magnitude": float(info.get("magnitude", decoded.get("magnitude", 0.0))),
            "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
            "dfast_breach": float(br.dfast_breach) if br else 0.0,
            "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
            "action_novelty": float(info.get("action_novelty", 0.0)),
            "total": float(br.total) if br else float(reward),
        })
    return results


def aggregate_scenarios(
    episodes: List[Dict[str, Any]],
    min_occurrences: int = 3,
) -> List[Dict[str, Any]]:
    """Group episodes by (target_var, family_name), average key metrics, rank by P/L."""
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    meta: Dict[tuple, Dict] = {}

    for r in episodes:
        key = (r["target_var"], r["family_name"])
        grouped[key].append(r["portfolio_loss"])
        if key not in meta or r["portfolio_loss"] > meta[key].get("best_pl", -1):
            meta[key] = {
                "target_var": r["target_var"],
                "family_name": r["family_name"],
                "magnitude": r["magnitude"],
                "causal_fidelity": r["causal_fidelity"],
                "dfast_breach": r["dfast_breach"],
                "best_pl": r["portfolio_loss"],
            }

    scenarios = []
    for key, losses in grouped.items():
        if len(losses) < min_occurrences:
            continue
        m = meta[key]
        scenarios.append({
            "target_var": m["target_var"],
            "family_name": m["family_name"],
            "magnitude": m["magnitude"],
            "occurrences": len(losses),
            "portfolio_loss_mean": float(np.mean(losses)),
            "portfolio_loss_max": float(np.max(losses)),
            "portfolio_loss_p75": float(np.percentile(losses, 75)),
            "causal_fidelity": m["causal_fidelity"],
            "dfast_breach": m["dfast_breach"],
        })

    scenarios.sort(key=lambda x: -x["portfolio_loss_mean"])
    return scenarios


def print_report(
    scenarios: List[Dict[str, Any]],
    top_k: int,
    portfolio_profile: str,
    n_episodes: int,
    run_name: str,
    random_mean_pl: Optional[float] = None,
):
    """Print the formatted risk manager report to stdout."""
    k = min(top_k, len(scenarios))
    divider = "=" * 88

    print()
    print(divider)
    print("  ADVERSARIAL STRESS SCENARIO REPORT")
    print(divider)
    print(f"  Portfolio:      {portfolio_profile}")
    print(f"  Model:          {run_name}")
    print(f"  Episodes run:   {n_episodes} (stochastic policy sampling)")
    print(f"  Scenarios found: {len(scenarios)} distinct (target_var x event_family) combinations")
    if random_mean_pl is not None:
        print(f"  Random baseline mean P/L: {random_mean_pl:.4f}")
    print()
    print(f"  Top-{k} worst-case scenarios ranked by mean portfolio loss:")
    print()
    print(f"  {'Rank':<5}  {'Target Var':<16}  {'Event Family':<26}  {'Mag':>5}  "
          f"{'P/L Mean':>9}  {'P/L Max':>8}  {'CF':>6}  {'n':>5}")
    print(f"  {'-'*5}  {'-'*16}  {'-'*26}  {'-'*5}  "
          f"{'-'*9}  {'-'*8}  {'-'*6}  {'-'*5}")

    for i, s in enumerate(scenarios[:k], 1):
        mag_str = f"{s['magnitude']:+.1f}s"
        print(
            f"  {i:<5}  {s['target_var']:<16}  {s['family_name']:<26}  "
            f"{mag_str:>5}  {s['portfolio_loss_mean']:>9.4f}  "
            f"{s['portfolio_loss_max']:>8.4f}  {s['causal_fidelity']:>6.4f}  "
            f"{s['occurrences']:>5}"
        )

    print()
    print("  Columns:")
    print("    P/L Mean  = average max-drawdown across episodes with this scenario")
    print("    P/L Max   = worst single realisation (tail risk)")
    print("    CF        = causal fidelity (fraction of causal edges respected, 0-1)")
    print("    n         = number of times this scenario was sampled")
    print()

    # Macro summary
    if scenarios:
        top = scenarios[0]
        print(f"  PRIMARY FINDING:")
        print(f"    Worst-case scenario: {top['target_var']} x {top['family_name']} "
              f"x {top['magnitude']:+.1f}sigma")
        print(f"    Expected drawdown:   {top['portfolio_loss_mean']:.1%} (mean), "
              f"{top['portfolio_loss_max']:.1%} (worst case)")
        unique_vars = len(set(s["target_var"] for s in scenarios[:k]))
        unique_fams = len(set(s["family_name"] for s in scenarios[:k]))
        print(f"    Scenario diversity:  {unique_vars} distinct target variables, "
              f"{unique_fams} distinct event families in top-{k}")

    print()
    print(divider)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate an adversarial scenario report from a trained PPO policy."
    )
    parser.add_argument("--run-dir", default=None,
                        help="Path to a training run directory. Default: latest in ml_pipeline/runs/.")
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="Stochastic episodes to run (more = better statistics, slower).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of scenarios to include in the ranked report.")
    parser.add_argument("--min-occurrences", type=int, default=3,
                        help="Minimum times a scenario must appear to be included.")
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save report JSON. Default: <run_dir>/scenario_report/.")
    parser.add_argument("--portfolio-profile", default=None,
                        help="Override portfolio profile from run config.")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from generative_engine_rl.action_space_loader import load_spec
    from generative_engine_rl.env_factory import make_env

    runs_base = ROOT / "runs"

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run(runs_base)

    model_path = run_dir / "final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"final.zip not found in {run_dir}")

    run_cfg = load_run_config(run_dir)
    mode = run_cfg.get("mode", "real")
    portfolio_profile = args.portfolio_profile or run_cfg.get("portfolio_profile", "balanced")
    actions_per_episode = int(run_cfg.get("actions_per_episode", 1))
    reward_mode = run_cfg.get("reward_mode", "portfolio_adversarial")

    print(f"\nAdversarial Scenario Report")
    print(f"  run_dir:   {run_dir.name}")
    print(f"  portfolio: {portfolio_profile}")
    print(f"  mode:      {mode}")
    print(f"  episodes:  {args.n_episodes}")
    print()

    spec = load_spec()
    env = make_env(
        mode=mode,
        seed=args.seed,
        spec=spec,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=True,
        reward_mode=reward_mode,
    )

    print("Loading model...")
    model = PPO.load(str(model_path))

    # ---- Collect stochastic episodes ----
    print(f"Running {args.n_episodes} stochastic episodes...")
    episodes = collect_stochastic_episodes(model, env, args.n_episodes, seed_start=args.seed)
    print(f"  Collected {len(episodes)} valid episodes ({args.n_episodes - len(episodes)} rejected)")

    # ---- Random baseline for comparison ----
    print("Running 100 random-baseline episodes for comparison...")
    nvec = env.action_space.nvec
    rng = np.random.default_rng(args.seed + 99999)
    random_pls = []
    for ep in range(100):
        obs, _ = env.reset(seed=args.seed + 99999 + ep)
        done = False
        while not done:
            action = np.array([
                rng.integers(nvec[0]), rng.integers(nvec[1]), rng.integers(nvec[2])
            ], dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if not info.get("action_rejected"):
            br = info.get("reward_breakdown")
            if br:
                random_pls.append(float(br.portfolio_loss))
    random_mean_pl = float(np.mean(random_pls)) if random_pls else None

    # ---- Aggregate + rank ----
    scenarios = aggregate_scenarios(episodes, min_occurrences=args.min_occurrences)

    # ---- Print report ----
    print_report(
        scenarios=scenarios,
        top_k=args.top_k,
        portfolio_profile=portfolio_profile,
        n_episodes=args.n_episodes,
        run_name=run_dir.name,
        random_mean_pl=random_mean_pl,
    )

    # ---- Distribution summary ----
    from collections import Counter
    var_counts = Counter(e["target_var"] for e in episodes)
    fam_counts = Counter(e["family_name"] for e in episodes)
    print("  Policy action distribution (target variables, top-5):")
    for var, count in var_counts.most_common(5):
        pct = count / len(episodes) * 100
        print(f"    {var:<20} {pct:>6.1f}%")
    print()
    print("  Policy action distribution (event families):")
    for fam, count in fam_counts.most_common():
        pct = count / len(episodes) * 100
        print(f"    {fam:<30} {pct:>6.1f}%")
    print()

    # ---- Performance lift ----
    trained_mean_pl = float(np.mean([e["portfolio_loss"] for e in episodes]))
    print(f"  Portfolio loss — trained: {trained_mean_pl:.4f}  random: {random_mean_pl:.4f}  "
          f"lift: {(trained_mean_pl / max(random_mean_pl, 1e-6) - 1) * 100:.0f}%")
    print()

    # ---- Save ----
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "scenario_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenarios.json"
    out_path.write_text(json.dumps({
        "run_name": run_dir.name,
        "portfolio_profile": portfolio_profile,
        "mode": mode,
        "n_episodes": args.n_episodes,
        "n_valid": len(episodes),
        "random_mean_portfolio_loss": random_mean_pl,
        "trained_mean_portfolio_loss": trained_mean_pl,
        "lift_pct": round((trained_mean_pl / max(random_mean_pl, 1e-6) - 1) * 100, 1),
        "ranked_scenarios": scenarios[:args.top_k],
    }, indent=2))
    print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
