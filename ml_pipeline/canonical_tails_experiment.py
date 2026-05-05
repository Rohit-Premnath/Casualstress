"""
Tail-aware variant of the canonical model.

Keeps the same regime-aware + causal graph + soft-filtered structure,
but swaps Gaussian innovations for Student-t innovations to better
capture crisis tail behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from all_paper_experiments import (
    CANONICAL_EVENTS,
    CANONICAL_PAPER_NAME,
    CANONICAL_TRAIN_REGIMES,
    CORE_VARS,
    COVERAGE_METRIC,
    EVENT_SHOCK_TEMPLATES,
    _fit_var,
    evaluate_scenarios,
    get_canonical_signature,
    load_all_data,
    load_causal_graph_from_db,
    load_canonical_graph,
    load_regime_series,
    score_canonical_plausibility,
    select_training_window,
    soft_filter_weights,
    store_experiment,
)


TAIL_MODEL_NAME = "Full Model (Soft Filtered + Student-t Tails)"
TAIL_DF_NORMAL = 7.0
TAIL_DF_CRISIS = 4.0
TAIL_DEPTH = 3
TAIL_PROPAGATION_DECAY = 0.4
TAIL_PROPAGATION_MIN = 0.12
TAIL_PROPAGATION_CLIP = 2.5


def _build_adj(causal_adj: dict | None) -> dict[str, list[tuple[str, float]]]:
    adj: dict[str, list[tuple[str, float]]] = {}
    if not causal_adj:
        return adj
    for edge_key, edge_data in causal_adj.items():
        cause, effect = edge_key.split("->")
        adj.setdefault(cause, []).append((effect, edge_data.get("weight", 0.0)))
    return adj


def _student_t_noise(cholesky: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    draws = student_t.rvs(df, size=cholesky.shape[0], random_state=rng)
    # Standardize to unit variance so the covariance scaling remains meaningful.
    draws = draws * np.sqrt((df - 2.0) / df)
    return cholesky @ draws


def _simulate_tail_aware(
    B,
    cov_n,
    cov_c,
    means,
    stds,
    lag,
    d,
    avail,
    n,
    horizon,
    rng: np.random.Generator,
    shock_template=None,
    causal_adj=None,
    clip=6.0,
):
    L_n = np.linalg.cholesky(cov_n)
    L_c = np.linalg.cholesky(cov_c)
    template = shock_template or {"^GSPC": -3.0}
    anchor_var = "^GSPC" if "^GSPC" in template else next(iter(template))
    anchor_sigma = template.get(anchor_var, -3.0)

    sign = 1.0 if anchor_sigma >= 0 else -1.0
    shock_levels = []
    for sigma, pct in [(3.0, 0.35), (4.0, 0.25), (5.0, 0.20), (6.0, 0.12), (7.0, 0.08)]:
        shock_levels.extend([sign * sigma] * max(1, int(n * pct)))
    while len(shock_levels) < n:
        shock_levels.append(anchor_sigma)
    shock_levels = shock_levels[:n]
    rng.shuffle(shock_levels)

    adj = _build_adj(causal_adj)
    scenarios = []

    for s in range(n):
        current_shock = shock_levels[s]
        scale = current_shock / anchor_sigma if anchor_sigma != 0 else 1.0

        initial = np.zeros(d)
        for var, sigma in template.items():
            if var in avail:
                initial[avail.index(var)] += sigma * scale

        if adj:
            visited = {v for v in template if v in avail}
            layer = [(v, template[v] * scale) for v in template if v in avail]
            for depth in range(TAIL_DEPTH):
                nxt = []
                decay = TAIL_PROPAGATION_DECAY ** (depth + 1)
                for src, src_shock in layer:
                    for tgt, w in adj.get(src, []):
                        if tgt in avail and tgt not in visited:
                            prop = float(np.clip(src_shock * w * decay, -TAIL_PROPAGATION_CLIP, TAIL_PROPAGATION_CLIP))
                            if abs(prop) > TAIL_PROPAGATION_MIN:
                                initial[avail.index(tgt)] += prop
                                visited.add(tgt)
                                nxt.append((tgt, prop))
                layer = nxt

        path = np.zeros((horizon + lag, d))
        path[lag, :] = initial

        for t in range(lag + 1, horizon + lag):
            x = [1.0]
            for l_idx in range(1, lag + 1):
                x.extend(path[t - l_idx])
            x = np.array(x)

            if abs(current_shock) >= 5.0:
                noise = _student_t_noise(L_c, TAIL_DF_CRISIS, rng) * 1.2
            elif abs(current_shock) >= 4.0:
                blend = 0.5 * cov_c + 0.5 * cov_n
                eigvals = np.linalg.eigvalsh(blend)
                if eigvals.min() < 0:
                    blend += np.eye(d) * (abs(eigvals.min()) + 0.001)
                noise = _student_t_noise(np.linalg.cholesky(blend), 5.0, rng) * 1.1
            else:
                noise = _student_t_noise(L_n, TAIL_DF_NORMAL, rng)

            path[t] = np.clip(x @ B + noise, -clip, clip)

        real = path[lag:] * stds + means
        scenarios.append(pd.DataFrame(real, columns=avail, index=range(horizon)))

    return scenarios


def gen_full_model_soft_filtered_tails(train, avail, n=200, horizon=60, **kw):
    B, cov_n, cov_c, means, stds, lag, d = _fit_var(train, avail)
    rng = kw["rng"]
    shock_template = kw.get("shock_template", {"^GSPC": -3.0})
    causal_adj = kw.get("causal_adj")
    event_type = kw.get("event_type", "market_crash")

    raw = _simulate_tail_aware(
        B, cov_n, cov_c, means, stds, lag, d, avail,
        n=n * 2,
        horizon=horizon,
        rng=rng,
        shock_template=shock_template,
        causal_adj=causal_adj,
        clip=6.0,
    )
    scores = score_canonical_plausibility(raw, avail, train[avail].std().to_numpy(), event_type, causal_adj=causal_adj)
    weights = soft_filter_weights(scores)
    ranked = sorted(zip(scores, weights, raw), key=lambda item: (item[0], item[1]), reverse=True)
    return [scenario for _, _, scenario in ranked[:n]]


def run_tail_model(seed_runs: int = 5) -> dict:
    print("=" * 90)
    print(f"  {TAIL_MODEL_NAME}")
    print(f"  Metric: {COVERAGE_METRIC}")
    print(f"  Seed runs: {seed_runs}")
    print("=" * 90)

    all_data = load_all_data()
    regime_series = load_regime_series()
    all_data = all_data.join(regime_series, how="left")
    discovery_adj = load_causal_graph_from_db()
    canonical_adj = load_canonical_graph(str(Path(__file__).resolve().parent))
    causal_adj = canonical_adj or discovery_adj

    print(f"  Base canonical signature: {get_canonical_signature()}")
    print(f"  Variant: Student-t innovations with causal graph retained")
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
            rng = np.random.default_rng(seed)
            scenarios = gen_full_model_soft_filtered_tails(
                train_regime,
                avail,
                200,
                60,
                causal_adj=causal_adj,
                shock_template=template,
                event_type=event["type"],
                rng=rng,
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
        "model": TAIL_MODEL_NAME,
        "base_signature": get_canonical_signature(),
        "variant": "student_t_innovations_with_causal_graph",
        "metric": COVERAGE_METRIC,
        "seed_runs": seed_runs,
        "per_seed_avg_coverage": [round(x, 1) for x in seed_means],
        "overall_mean_coverage": round(overall_mean, 1),
        "overall_std_coverage": round(overall_std, 1),
        "overall_mean_direction": round(overall_direction, 1),
        "overall_mean_pairwise": round(overall_pairwise, 1),
        "per_event": event_results,
    }
    store_experiment("Exp4_Canonical_Tail_Variant", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-runs", type=int, default=5)
    args = parser.parse_args()
    run_tail_model(seed_runs=max(1, args.seed_runs))


if __name__ == "__main__":
    main()
