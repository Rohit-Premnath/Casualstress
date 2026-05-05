"""
Per-Event x Per-Variable Coverage Matrix
==========================================
Extracts the canonical model's coverage breakdown at the most granular level:
  - 11 events x 6 key variables = 66 cells
  - Each cell: actual outcome, p5/median/p95, in-range (bool), direction (bool)

This is the data behind Figure 8 (the per-event heatmap) and Table 9
(per-event detailed results).

Uses the CANONICAL model (Full Model Soft Filtered + Student-t data-fit df)
with the 5 canonical seeds. For each (event, variable) cell, reports the mean
across seeds.

Output:
  - Console table: 11 rows x 6 variables showing coverage + direction
  - JSON/parquet export for figure generation scripts
  - Stored in models.per_variable_coverage_matrix table
"""

import os
import sys
import json
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Import generators from all_paper_experiments (same folder)
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
        VALIDATION_INDICES,
        TEST_INDICES,
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
# DB
# ============================================================

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def store_matrix(payload):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models.per_variable_coverage_matrix (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            matrix_name VARCHAR(200),
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    clean = json.loads(json.dumps(payload, default=lambda x:
        None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
        else bool(x) if isinstance(x, (np.bool_,))
        else float(x) if isinstance(x, (np.floating,))
        else int(x) if isinstance(x, (np.integer,))
        else x))
    cursor.execute(
        "INSERT INTO models.per_variable_coverage_matrix (id, matrix_name, results) VALUES (%s, %s, %s)",
        (str(uuid.uuid4()), "Canonical per-event per-variable breakdown", Json(clean)),
    )
    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# PER-CELL EVALUATION
# ============================================================

def evaluate_single_cell(scenarios, actual, var, window):
    """
    Evaluate scenarios for a SINGLE variable on a single event.
    Returns: dict with actual_cum, p5, p50, p95, in_range (bool), direction (bool)
    """
    if var not in actual.columns:
        return None

    days = min(window, len(actual), 60)
    actual_cum = actual[var].iloc[:days].sum()

    try:
        pred_cums = np.array([s[var].iloc[:days].sum() for s in scenarios if var in s.columns])
    except Exception:
        return None

    if len(pred_cums) == 0:
        return None

    p5 = float(np.percentile(pred_cums, 5))
    p50 = float(np.median(pred_cums))
    p95 = float(np.percentile(pred_cums, 95))

    in_range = bool(p5 <= actual_cum <= p95)
    same_dir = bool(
        (actual_cum >= 0 and p50 >= 0) or (actual_cum < 0 and p50 < 0)
    )

    # Convert to display units
    if var in LOG_RETURN_VARS:
        a_d = (np.exp(actual_cum) - 1) * 100
        p5_d = (np.exp(p5) - 1) * 100
        p50_d = (np.exp(p50) - 1) * 100
        p95_d = (np.exp(p95) - 1) * 100
        unit = "%"
    else:
        a_d = actual_cum * 100
        p5_d = p5 * 100
        p50_d = p50 * 100
        p95_d = p95 * 100
        unit = "bps"

    return {
        "variable": var,
        "actual": round(float(a_d), 2),
        "p5": round(float(p5_d), 2),
        "p50": round(float(p50_d), 2),
        "p95": round(float(p95_d), 2),
        "in_range": in_range,
        "direction_ok": same_dir,
        "unit": unit,
    }


# ============================================================
# MAIN
# ============================================================

def build_matrix(seed_runs: int = 5):
    print("=" * 100)
    print("  PER-EVENT x PER-VARIABLE COVERAGE MATRIX (CANONICAL MODEL)")
    print(f"  Canonical: Student-t data-fit df (normal={PAPER_DF_NORMAL}, crisis={PAPER_DF_CRISIS})")
    print(f"  Seeds: {PAPER_SEEDS[:seed_runs]}")
    print("=" * 100)

    # Load data + graph
    all_data = load_all_data()
    regime_series = load_regime_series()
    all_data = all_data.join(regime_series, how="left")
    discovery_adj = load_causal_graph_from_db()
    canonical_adj = load_canonical_graph(os.path.dirname(__file__))
    causal_adj = canonical_adj or discovery_adj

    print(f"  Data: {len(all_data)} days | Graph edges: {len(causal_adj) if causal_adj else 0}")
    print()

    # Results structure:
    # matrix[event_name][var_name] = {
    #   "in_range_rate": float (0..1 across seeds),
    #   "direction_rate": float (0..1 across seeds),
    #   "actual": mean across seeds,
    #   "p5": mean across seeds, "p50": mean, "p95": mean,
    #   "split": "VAL" or "TEST"
    # }
    matrix = {}

    for event_idx, event in enumerate(CANONICAL_EVENTS):
        split_tag = "VAL" if event_idx in VALIDATION_INDICES else "TEST"

        cutoff = pd.to_datetime(event["cutoff"])
        ev_start = pd.to_datetime(event["start"])
        ev_end = pd.to_datetime(event["end"])

        train_full = all_data[all_data.index < cutoff]
        train_regime = select_training_window(train_full, train_regimes=CANONICAL_TRAIN_REGIMES)
        actual = all_data[(all_data.index >= ev_start) & (all_data.index <= ev_end)]
        avail = [v for v in CORE_VARS if v in train_regime.columns]

        if len(train_regime) < 500:
            print(f"  [{event_idx+1}/11] [{split_tag}] {event['name']}: SKIP (insufficient data)")
            continue

        template = {
            v: s for v, s in
            EVENT_SHOCK_TEMPLATES.get(event["type"], {"^GSPC": -3.0}).items()
            if v in avail
        }

        # Per-seed storage: for each key var, track across-seed lists
        per_var_seed: dict = {var: [] for var in KEY_VARS if var in avail}

        for seed in PAPER_SEEDS[:seed_runs]:
            np.random.seed(seed)
            scenarios = gen_full_model_soft_filtered_tails(
                train_regime, avail,
                get_canonical_target_scenarios(), 60,
                shock_template=template,
                event_type=event["type"],
                causal_adj=causal_adj,
            )

            for var in KEY_VARS:
                if var not in avail:
                    continue
                cell = evaluate_single_cell(scenarios, actual, var, event["window"])
                if cell:
                    per_var_seed[var].append(cell)

        # Aggregate per-variable across seeds
        event_row = {
            "event": event["name"],
            "event_type": event["type"],
            "split": split_tag,
            "window_days": event["window"],
            "variables": {},
        }

        print(f"  [{event_idx+1}/11] [{split_tag}] {event['name']}")
        header = f"     {'Variable':<15} {'Actual':>10} {'P5':>10} {'Median':>10} {'P95':>10}  {'In-Range':>10}  {'Dir-OK':>8}"
        print(header)

        for var in KEY_VARS:
            if var not in per_var_seed or not per_var_seed[var]:
                continue
            cells = per_var_seed[var]
            in_range_rate = float(np.mean([c["in_range"] for c in cells]))
            direction_rate = float(np.mean([c["direction_ok"] for c in cells]))
            mean_actual = float(np.mean([c["actual"] for c in cells]))
            mean_p5 = float(np.mean([c["p5"] for c in cells]))
            mean_p50 = float(np.mean([c["p50"] for c in cells]))
            mean_p95 = float(np.mean([c["p95"] for c in cells]))
            unit = cells[0]["unit"]

            event_row["variables"][var] = {
                "in_range_rate": round(in_range_rate, 3),
                "direction_rate": round(direction_rate, 3),
                "actual": round(mean_actual, 2),
                "p5": round(mean_p5, 2),
                "p50": round(mean_p50, 2),
                "p95": round(mean_p95, 2),
                "unit": unit,
            }

            ir = "YES" if in_range_rate >= 0.5 else "NO"
            do = "YES" if direction_rate >= 0.5 else "NO"
            print(
                f"     {var:<15} {mean_actual:>+9.1f}{unit:<1} "
                f"{mean_p5:>+9.1f}{unit:<1} {mean_p50:>+9.1f}{unit:<1} {mean_p95:>+9.1f}{unit:<1}  "
                f"{ir:>10}  {do:>8}"
            )

        matrix[event["name"]] = event_row
        print()

    # ---------------------------------------------------------
    # SUMMARY TABLE: 11 events x 6 vars coverage heatmap source
    # ---------------------------------------------------------
    print("=" * 100)
    print("  SUMMARY: 11 x 6 COVERAGE MATRIX (for Figure 8 heatmap)")
    print("  Cell value = in-range rate across 5 seeds (1.0 = always covered, 0.0 = never)")
    print("=" * 100)

    header = f"  {'Event':<28} " + "".join(f"{v:>10}" for v in KEY_VARS) + f" {'Row%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for event in CANONICAL_EVENTS:
        name = event["name"]
        if name not in matrix:
            continue
        row = matrix[name]
        line = f"  {name[:28]:<28} "
        row_vals = []
        for var in KEY_VARS:
            if var in row["variables"]:
                v = row["variables"][var]["in_range_rate"]
                row_vals.append(v)
                line += f"{v:>10.2f}"
            else:
                line += f"{'--':>10}"
        row_pct = float(np.mean(row_vals)) * 100 if row_vals else 0.0
        line += f" {row_pct:>7.1f}%"
        print(line)

    # Column averages
    print("  " + "-" * (len(header) - 2))
    col_line = f"  {'COLUMN AVG (all 11 events)':<28} "
    for var in KEY_VARS:
        col_vals = [
            matrix[ev["name"]]["variables"][var]["in_range_rate"]
            for ev in CANONICAL_EVENTS
            if ev["name"] in matrix and var in matrix[ev["name"]]["variables"]
        ]
        col_avg = float(np.mean(col_vals)) if col_vals else 0.0
        col_line += f"{col_avg:>10.2f}"
    print(col_line)

    # ---------------------------------------------------------
    # DIRECTION MATRIX
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("  DIRECTION MATRIX: same shape, direction-correct rate")
    print("=" * 100)

    print(header)
    print("  " + "-" * (len(header) - 2))
    for event in CANONICAL_EVENTS:
        name = event["name"]
        if name not in matrix:
            continue
        row = matrix[name]
        line = f"  {name[:28]:<28} "
        row_vals = []
        for var in KEY_VARS:
            if var in row["variables"]:
                v = row["variables"][var]["direction_rate"]
                row_vals.append(v)
                line += f"{v:>10.2f}"
            else:
                line += f"{'--':>10}"
        row_pct = float(np.mean(row_vals)) * 100 if row_vals else 0.0
        line += f" {row_pct:>7.1f}%"
        print(line)

    # Column averages
    print("  " + "-" * (len(header) - 2))
    col_line = f"  {'COLUMN AVG (all 11 events)':<28} "
    for var in KEY_VARS:
        col_vals = [
            matrix[ev["name"]]["variables"][var]["direction_rate"]
            for ev in CANONICAL_EVENTS
            if ev["name"] in matrix and var in matrix[ev["name"]]["variables"]
        ]
        col_avg = float(np.mean(col_vals)) if col_vals else 0.0
        col_line += f"{col_avg:>10.2f}"
    print(col_line)

    # ---------------------------------------------------------
    # STORE
    # ---------------------------------------------------------
    payload = {
        "canonical_signature": get_canonical_signature(),
        "seeds": PAPER_SEEDS[:seed_runs],
        "n_events": len(matrix),
        "key_variables": KEY_VARS,
        "matrix": matrix,
    }
    try:
        store_matrix(payload)
        print(f"\n  Stored in models.per_variable_coverage_matrix")
    except Exception as e:
        print(f"\n  WARNING: DB store failed: {e}")

    # Also save to JSON file for direct figure-script consumption
    out_path = Path(__file__).parent / "per_variable_coverage_matrix.json"
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  JSON export: {out_path}")
    except Exception as e:
        print(f"  WARNING: JSON write failed: {e}")

    print("=" * 100)
    return matrix


if __name__ == "__main__":
    build_matrix(seed_runs=5)
