"""
COVID Fan Chart Data Extract (for Figure 6)
============================================
Runs the canonical scenario generator once on the 2020 COVID event and dumps
all 200 scenario paths, the actual market trajectory, and percentile bands
in a single JSON file. The plotting script consumes this JSON to produce
Figure 6 in the paper.

This is a data extraction, not an evaluation. The coverage/direction metrics
for COVID are already locked in canonical_paper_numbers.py (53.3% / 100%).

What's in the output:
  - Metadata: event dates, seed, signature
  - For each of 6 key variables:
    - actual_path:     actual market trajectory (event_start -> event_end)
    - scenario_paths:  200 x horizon_days simulated paths (for fan plot)
    - percentiles:     p5, p25, p50, p75, p95 arrays (for shaded bands)
  - Display units: % for log-return vars, bps for credit/yield vars

Runs in ~10 seconds using seed 20260407 (first canonical seed).
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from all_paper_experiments import (
        gen_full_model_soft_filtered_tails,
        load_all_data,
        load_regime_series,
        load_causal_graph_from_db,
        select_training_window,
        CANONICAL_EVENTS,
        CORE_VARS,
        KEY_VARS,
        EVENT_SHOCK_TEMPLATES,
        LOG_RETURN_VARS,
        PAPER_SEEDS,
        PAPER_DF_NORMAL,
        PAPER_DF_CRISIS,
        PAPER_MID_DF,
    )
except ImportError as e:
    print(f"ERROR importing from all_paper_experiments: {e}")
    print("Make sure this script is in the same folder as all_paper_experiments.py")
    sys.exit(1)

try:
    from ml_pipeline.canonical_best_model import (
        CANONICAL_TRAIN_REGIMES,
        get_canonical_signature,
        get_canonical_target_scenarios,
        load_canonical_graph,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from canonical_best_model import (
        CANONICAL_TRAIN_REGIMES,
        get_canonical_signature,
        get_canonical_target_scenarios,
        load_canonical_graph,
    )

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

COVID_EVENT_NAME = "2020 COVID"
DATA_SEED = 20260407                 # first canonical seed, matches backtest


def find_covid_event():
    """Find the 2020 COVID event in CANONICAL_EVENTS."""
    for event in CANONICAL_EVENTS:
        if event["name"] == COVID_EVENT_NAME:
            return event
    raise ValueError(f"Event {COVID_EVENT_NAME} not found in CANONICAL_EVENTS")


def cumulative_path(daily_returns: pd.Series, var: str) -> np.ndarray:
    """
    Convert a daily returns series into a cumulative path for plotting.
    For log-return vars: cumulative log return, converted to (%) price change
    For yield/spread vars: cumulative change in bps
    """
    cum = daily_returns.cumsum()
    if var in LOG_RETURN_VARS:
        # Convert log-returns to % price change: (e^x - 1) * 100
        return (np.exp(cum.values) - 1) * 100
    else:
        # Yield/spread vars are already in decimal form; convert to bps
        return cum.values * 100


def cumulative_path_array(daily_returns_arr: np.ndarray, var: str) -> np.ndarray:
    """Same as cumulative_path but for numpy arrays."""
    cum = np.cumsum(daily_returns_arr)
    if var in LOG_RETURN_VARS:
        return (np.exp(cum) - 1) * 100
    else:
        return cum * 100


def extract_unit(var: str) -> str:
    """Return display unit for a variable."""
    return "%" if var in LOG_RETURN_VARS else "bps"


# ============================================================
# MAIN
# ============================================================

def build_fan_chart_data():
    print("=" * 90)
    print("  COVID FAN CHART DATA EXTRACT (Figure 6)")
    print(f"  Event: {COVID_EVENT_NAME}")
    print(f"  Canonical: Student-t data-fit df (normal={PAPER_DF_NORMAL}, crisis={PAPER_DF_CRISIS})")
    print(f"  Seed: {DATA_SEED}")
    print("=" * 90)

    # Load data + graph
    all_data = load_all_data()
    regime_series = load_regime_series()
    all_data = all_data.join(regime_series, how="left")
    discovery_adj = load_causal_graph_from_db()
    canonical_adj = load_canonical_graph(os.path.dirname(__file__))
    causal_adj = canonical_adj or discovery_adj

    print(f"  Data: {len(all_data)} days | Graph edges: {len(causal_adj) if causal_adj else 0}")

    # Find COVID event
    event = find_covid_event()
    cutoff = pd.to_datetime(event["cutoff"])
    ev_start = pd.to_datetime(event["start"])
    ev_end = pd.to_datetime(event["end"])

    # Training data: everything before cutoff, filtered by canonical train regimes
    train_full = all_data[all_data.index < cutoff]
    train_regime = select_training_window(train_full, train_regimes=CANONICAL_TRAIN_REGIMES)
    actual = all_data[(all_data.index >= ev_start) & (all_data.index <= ev_end)]
    avail = [v for v in CORE_VARS if v in train_regime.columns]

    print(f"\n  Event window: {ev_start.date()} to {ev_end.date()}  ({event['window']} days)")
    print(f"  Horizon:      60 days")
    print(f"  Training:     {len(train_regime)} regime-filtered days")
    print(f"  Actual obs:   {len(actual)} days")
    print(f"  Variables in CORE_VARS & train: {len(avail)}")

    # Generate 200 scenarios
    template = {
        v: s for v, s in
        EVENT_SHOCK_TEMPLATES.get(event["type"], {"^GSPC": -3.0}).items()
        if v in avail
    }
    print(f"\n  Shock template: {template}")

    np.random.seed(DATA_SEED)
    scenarios = gen_full_model_soft_filtered_tails(
        train_regime, avail,
        get_canonical_target_scenarios(), 60,
        shock_template=template,
        event_type=event["type"],
        causal_adj=causal_adj,
    )

    print(f"  Generated {len(scenarios)} scenarios, each {len(scenarios[0])} days")

    # Build per-variable payload
    variables_payload = {}

    print(f"\n  Extracting per-variable data for {len(KEY_VARS)} key variables...")
    for var in KEY_VARS:
        if var not in avail:
            print(f"    {var}: SKIP (not in avail)")
            continue

        unit = extract_unit(var)

        # Actual trajectory (event-window days only)
        actual_path = cumulative_path(actual[var], var).tolist()

        # Scenario paths (horizon = 60 days)
        scenario_paths = []
        for s in scenarios:
            if var not in s.columns:
                continue
            path = cumulative_path_array(s[var].values, var)
            scenario_paths.append(path.tolist())

        scenario_paths_arr = np.array(scenario_paths)  # shape: (200, 60)

        # Percentile bands (p5, p25, p50, p75, p95) at each day
        percentiles = {
            "p5":  np.percentile(scenario_paths_arr, 5,  axis=0).tolist(),
            "p25": np.percentile(scenario_paths_arr, 25, axis=0).tolist(),
            "p50": np.percentile(scenario_paths_arr, 50, axis=0).tolist(),
            "p75": np.percentile(scenario_paths_arr, 75, axis=0).tolist(),
            "p95": np.percentile(scenario_paths_arr, 95, axis=0).tolist(),
        }

        # Quick sanity check: where does actual end up vs the bands?
        actual_final = actual_path[-1] if actual_path else None
        p5_at_end = percentiles["p5"][event["window"] - 1] if event["window"] - 1 < 60 else None
        p50_at_end = percentiles["p50"][event["window"] - 1] if event["window"] - 1 < 60 else None
        p95_at_end = percentiles["p95"][event["window"] - 1] if event["window"] - 1 < 60 else None

        in_band = (
            p5_at_end is not None and p95_at_end is not None
            and p5_at_end <= actual_final <= p95_at_end
        )

        variables_payload[var] = {
            "display_unit": unit,
            "actual_path": [round(float(x), 4) for x in actual_path],
            "scenario_paths": [[round(float(x), 4) for x in path] for path in scenario_paths],
            "percentiles": {
                k: [round(float(x), 4) for x in v]
                for k, v in percentiles.items()
            },
            "sanity_check": {
                "actual_end": round(float(actual_final), 4) if actual_final is not None else None,
                "p5_at_end": round(float(p5_at_end), 4) if p5_at_end is not None else None,
                "p50_at_end": round(float(p50_at_end), 4) if p50_at_end is not None else None,
                "p95_at_end": round(float(p95_at_end), 4) if p95_at_end is not None else None,
                "in_band_at_end": bool(in_band),
            },
        }

        marker = "[COVERED]" if in_band else "[MISS]"
        print(
            f"    {var:<14}  actual_end={actual_final:>+9.2f}{unit:<1}  "
            f"p5={p5_at_end:>+9.2f}  p50={p50_at_end:>+9.2f}  p95={p95_at_end:>+9.2f}  {marker}"
        )

    # Build final payload
    payload = {
        "event_name": event["name"],
        "event_type": event["type"],
        "cutoff": str(cutoff.date()),
        "event_start": str(ev_start.date()),
        "event_end": str(ev_end.date()),
        "window_days": event["window"],
        "horizon_days": 60,
        "n_scenarios": len(scenarios),
        "seed": DATA_SEED,
        "canonical_signature": get_canonical_signature(),
        "key_variables": KEY_VARS,
        "graph_edges": len(causal_adj) if causal_adj else 0,
        "training_days": len(train_regime),
        "variables": variables_payload,
    }

    # Write output
    out_path = Path(__file__).parent / "covid_fan_chart.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    file_kb = out_path.stat().st_size / 1024
    print(f"\n  JSON export: {out_path}")
    print(f"  File size:   {file_kb:.1f} KB")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("  PER-VARIABLE COVERAGE AT EVENT END (sanity check vs canonical backtest)")
    print("=" * 90)

    n_covered = sum(1 for v in variables_payload.values() if v["sanity_check"]["in_band_at_end"])
    n_total = len(variables_payload)
    covered_pct = (n_covered / n_total * 100) if n_total else 0.0

    print(f"\n  Covered at event end: {n_covered}/{n_total} ({covered_pct:.1f}%)")
    print(f"  Canonical backtest (5-seed avg) for COVID: 53.3%")
    print()
    print(f"  Single-seed result may differ from 5-seed average; this is expected.")
    print(f"  The JSON contains per-day percentile bands and all 200 scenario paths,")
    print(f"  so the plotting script can visualize the full distribution regardless.")

    print("\n" + "=" * 90)
    print("  Figure 6 is ready to plot. Sample script:")
    print("=" * 90)
    print("""
    import json, matplotlib.pyplot as plt, numpy as np

    data = json.load(open("covid_fan_chart.json"))
    var_data = data["variables"]["^GSPC"]
    horizon = range(data["horizon_days"])
    actual_x = range(data["window_days"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(horizon, var_data["percentiles"]["p5"], var_data["percentiles"]["p95"],
                    alpha=0.25, label="5-95% scenario range")
    ax.fill_between(horizon, var_data["percentiles"]["p25"], var_data["percentiles"]["p75"],
                    alpha=0.35, label="25-75% scenario range")
    ax.plot(horizon, var_data["percentiles"]["p50"], "--", label="scenario median", lw=2)
    ax.plot(actual_x, var_data["actual_path"], "k-", label="actual", lw=2.5)
    ax.set_xlabel("Days from event start")
    ax.set_ylabel(f"Cumulative change ({var_data['display_unit']})")
    ax.legend(); plt.show()
    """)
    print("=" * 90)

    return payload


if __name__ == "__main__":
    build_fan_chart_data()
