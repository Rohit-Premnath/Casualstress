"""
Run the locked canonical model across multiple seeds on the 11-event backtest.

This is a lightweight paper-rigor utility that answers:
"How stable is the canonical winner across random seeds?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from all_paper_experiments import (
    CANONICAL_EVENTS,
    CANONICAL_PAPER_NAME,
    CORE_VARS,
    COVERAGE_METRIC,
    EVENT_SHOCK_TEMPLATES,
    evaluate_scenarios,
    gen_full_model_soft_filtered,
    get_canonical_signature,
    load_all_data,
    load_causal_graph_from_db,
    load_regime_series,
    load_canonical_graph,
    score_canonical_plausibility,
    select_training_window,
    store_experiment,
    CANONICAL_TRAIN_REGIMES,
    get_canonical_target_scenarios,
)


def run_canonical_seed_stability(seed_runs: int = 5) -> dict:
    print("=" * 90)
    print(f"  {CANONICAL_PAPER_NAME} SEED STABILITY")
    print(f"  Metric: {COVERAGE_METRIC}")
    print(f"  Seed runs: {seed_runs}")
    print("=" * 90)

    all_data = load_all_data()
    regime_series = load_regime_series()
    all_data = all_data.join(regime_series, how="left")
    discovery_adj = load_causal_graph_from_db()
    canonical_adj = load_canonical_graph(str(Path(__file__).resolve().parent))
    causal_adj = canonical_adj or discovery_adj

    print(f"  Canonical signature: {get_canonical_signature()}")
    print(f"  Canonical graph edges: {len(causal_adj) if causal_adj else 0}")

    event_results = []
    seed_averages = [[] for _ in range(seed_runs)]

    for i, event in enumerate(CANONICAL_EVENTS):
        cutoff = np.datetime64(event["cutoff"])
        ev_start = np.datetime64(event["start"])
        ev_end = np.datetime64(event["end"])

        train_full = all_data[all_data.index < cutoff]
        train_regime = select_training_window(train_full, train_regimes=CANONICAL_TRAIN_REGIMES)
        actual = all_data[(all_data.index >= ev_start) & (all_data.index <= ev_end)]
        train_plain = train_full.drop(columns=["regime_name"], errors="ignore")
        avail = [v for v in CORE_VARS if v in train_plain.columns]
        template = {v: s for v, s in EVENT_SHOCK_TEMPLATES.get(event["type"], {"^GSPC": -3.0}).items() if v in avail}

        coverages = []
        directions = []
        pairwise = []
        plausibilities = []

        print(f"\n  [{i+1}/11] {event['name']}")

        for seed in range(seed_runs):
            np.random.seed(seed)
            scenarios = gen_full_model_soft_filtered(
                train_regime,
                avail,
                get_canonical_target_scenarios(),
                60,
                causal_adj=causal_adj,
                shock_template=template,
                event_type=event["type"],
            )
            coverage, direction, details, _ = evaluate_scenarios(scenarios, actual, avail, event["window"])
            plausibility_scores = score_canonical_plausibility(
                scenarios,
                avail,
                train_regime[avail].std().to_numpy(),
                event["type"],
                causal_adj=causal_adj,
            )
            pairwise_scores = []
            for var, direction_hint in EVENT_SHOCK_TEMPLATES.get(event["type"], {}).items():
                if var in details:
                    if direction_hint >= 0:
                        pairwise_scores.append(100.0 if details[var]["median"] >= 0 else 0.0)
                    else:
                        pairwise_scores.append(100.0 if details[var]["median"] < 0 else 0.0)

            coverages.append(float(coverage))
            directions.append(float(direction))
            pairwise.append(float(np.mean(pairwise_scores)) if pairwise_scores else 0.0)
            plausibilities.append(float(np.mean(plausibility_scores)))
            seed_averages[seed].append(float(coverage))

        mean_cov = float(np.mean(coverages))
        std_cov = float(np.std(coverages, ddof=0))
        mean_dir = float(np.mean(directions))
        mean_pair = float(np.mean(pairwise))
        print(f"    coverage mean±std: {mean_cov:.1f}% ± {std_cov:.1f}% | direction: {mean_dir:.1f}% | pairwise: {mean_pair:.1f}%")

        event_results.append(
            {
                "event": event["name"],
                "type": event["type"],
                "coverage_by_seed": [round(x, 1) for x in coverages],
                "coverage_mean": round(mean_cov, 1),
                "coverage_std": round(std_cov, 1),
                "direction_mean": round(mean_dir, 1),
                "pairwise_mean": round(mean_pair, 1),
                "plausibility_mean": round(float(np.mean(plausibilities)), 4),
            }
        )

    seed_means = [float(np.mean(rows)) for rows in seed_averages]
    overall_mean = float(np.mean(seed_means))
    overall_std = float(np.std(seed_means, ddof=0))
    overall_direction = float(np.mean([row["direction_mean"] for row in event_results]))
    overall_pairwise = float(np.mean([row["pairwise_mean"] for row in event_results]))

    print("\n" + "-" * 90)
    print(f"  Overall mean coverage across seeds: {overall_mean:.1f}% ± {overall_std:.1f}%")
    print(f"  Overall mean direction accuracy: {overall_direction:.1f}%")
    print(f"  Overall mean pairwise accuracy: {overall_pairwise:.1f}%")
    print(f"  Per-seed average coverages: {', '.join(f'{x:.1f}%' for x in seed_means)}")

    results = {
        "model": CANONICAL_PAPER_NAME,
        "signature": get_canonical_signature(),
        "metric": COVERAGE_METRIC,
        "seed_runs": seed_runs,
        "per_seed_avg_coverage": [round(x, 1) for x in seed_means],
        "overall_mean_coverage": round(overall_mean, 1),
        "overall_std_coverage": round(overall_std, 1),
        "overall_mean_direction": round(overall_direction, 1),
        "overall_mean_pairwise": round(overall_pairwise, 1),
        "per_event": event_results,
    }
    store_experiment("Exp4_Canonical_Seed_Stability", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-runs", type=int, default=5)
    args = parser.parse_args()
    run_canonical_seed_stability(seed_runs=max(1, args.seed_runs))


if __name__ == "__main__":
    main()
