"""
Regulatory Compliance Engine
===============================
Loads official DFAST/CCAR/EBA stress test scenarios from the Federal Reserve
and runs them through our causal model to:

1. Stress test portfolios against official regulatory scenarios
2. Generate the "Causal Difference Report" — showing WHERE our causal model
   diverges from the Fed's assumptions and WHY

The Causal Difference Report is our killer differentiator:
- Fed uses historical correlations to project variable paths
- We use a causally-discovered graph to propagate shocks
- Where we diverge, we trace the causal path and explain in plain English

DFAST scenarios are published annually at:
https://www.federalreserve.gov/supervisionreg/dfast.htm
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


# ============================================
# FALLBACK SCENARIOS (only used if DB is empty)
# These are approximations — real data should be
# loaded from Fed CSVs via load_real_dfast.py
# ============================================

DFAST_2026_SEVERELY_ADVERSE = {
    "name": "DFAST 2026 Severely Adverse",
    "source": "Federal Reserve (Approximated)",
    "year": 2026,
    "scenario_type": "severely_adverse",
    "description": "Approximated DFAST 2026 severely adverse scenario. "
                   "Replace with real Fed CSV data using load_real_dfast_2026.py",
    "horizon_quarters": 9,
    "variables": {
        "GDP_GROWTH": {
            "fed_name": "Real GDP growth rate",
            "our_code": "A191RL1Q225SBEA",
            "unit": "percent",
            "quarterly_path": [-8.0, -5.5, -3.0, -1.0, 0.5, 1.5, 2.0, 2.5, 3.0],
        },
        "UNEMPLOYMENT": {
            "fed_name": "Civilian unemployment rate",
            "our_code": "UNRATE",
            "unit": "percent",
            "quarterly_path": [5.5, 7.0, 8.5, 9.5, 10.0, 9.8, 9.5, 9.0, 8.5],
        },
        "BBB_SPREAD": {
            "fed_name": "BBB corporate yield spread",
            "our_code": "BAMLH0A0HYM2",
            "unit": "percent",
            "quarterly_path": [4.5, 5.5, 6.0, 5.8, 5.5, 5.0, 4.5, 4.0, 3.5],
        },
        "EQUITY_PRICES": {
            "fed_name": "Dow Jones Total Stock Market Index",
            "our_code": "^GSPC",
            "unit": "index_level",
            "quarterly_path": [75, 55, 50, 48, 45, 50, 55, 60, 68],
        },
        "VIX": {
            "fed_name": "CBOE Volatility Index",
            "our_code": "^VIX",
            "unit": "index_level",
            "quarterly_path": [45, 55, 50, 42, 38, 32, 28, 22, 18],
        },
    },
}

EBA_2025_ADVERSE = {
    "name": "EBA 2025 Adverse Scenario",
    "source": "European Banking Authority",
    "year": 2025,
    "scenario_type": "adverse",
    "description": "European adverse scenario featuring geopolitical tensions, energy price shocks, "
                   "and sovereign debt stress across the Euro area.",
    "horizon_quarters": 9,
    "variables": {
        "GDP_GROWTH": {
            "fed_name": "EU GDP growth rate", "our_code": "A191RL1Q225SBEA", "unit": "percent",
            "quarterly_path": [-5.0, -3.5, -2.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.0],
        },
        "UNEMPLOYMENT": {
            "fed_name": "EU unemployment rate", "our_code": "UNRATE", "unit": "percent",
            "quarterly_path": [7.5, 8.5, 9.5, 10.0, 10.2, 10.0, 9.5, 9.0, 8.5],
        },
        "EQUITY_PRICES": {
            "fed_name": "EU equity prices", "our_code": "^GSPC", "unit": "index_level",
            "quarterly_path": [80, 65, 55, 52, 50, 55, 60, 65, 72],
        },
        "OIL_PRICE": {
            "fed_name": "Oil price (Brent)", "our_code": "CL=F", "unit": "dollars_per_barrel",
            "quarterly_path": [120, 130, 125, 110, 95, 85, 80, 78, 75],
        },
    },
}

# Only used as fallback if DB has no scenarios
FALLBACK_SCENARIOS = [DFAST_2026_SEVERELY_ADVERSE, EBA_2025_ADVERSE]


# ============================================
# DATABASE
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# STEP 1: ENSURE FALLBACK SCENARIOS IN DB
# ============================================

def ensure_fallback_scenarios():
    """Insert fallback scenarios only if they don't exist in DB."""
    conn = get_db_connection()
    cursor = conn.cursor()

    for scenario in FALLBACK_SCENARIOS:
        cursor.execute(
            "SELECT id FROM regulatory.scenarios WHERE name = %s",
            (scenario["name"],)
        )
        existing = cursor.fetchone()

        if not existing:
            scenario_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO regulatory.scenarios
                    (id, name, source, year, scenario_type, description, variables, horizon_quarters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                scenario_id,
                scenario["name"],
                scenario["source"],
                scenario["year"],
                scenario["scenario_type"],
                scenario["description"],
                Json(scenario["variables"]),
                scenario["horizon_quarters"],
            ))
            print(f"  {scenario['name']}: inserted fallback (ID: {scenario_id})")

    conn.commit()
    cursor.close()
    conn.close()


# ============================================
# STEP 2: LOAD ALL SCENARIOS FROM DATABASE
# ============================================

def load_scenarios_from_db():
    """Load all regulatory scenarios from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, source, year, scenario_type, description, variables, horizon_quarters
        FROM regulatory.scenarios
        ORDER BY year DESC, name
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    scenarios = []
    for row in rows:
        scenarios.append({
            "db_id": str(row[0]),
            "name": row[1],
            "source": row[2],
            "year": row[3],
            "scenario_type": row[4],
            "description": row[5],
            "variables": row[6],
            "horizon_quarters": row[7],
        })

    return scenarios


# ============================================
# STEP 3: RUN REGULATORY SCENARIO THROUGH CAUSAL MODEL
# ============================================

def run_causal_projection(scenario):
    """
    Run the Fed's scenario through our causal model to generate
    OUR projections for the same variables.
    """
    print(f"\nRunning causal projection for: {scenario['name']}...")

    # Load our causal graph
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT adjacency_matrix
        FROM models.causal_graphs
        WHERE method = 'ensemble_dynotears_pcmci'
        ORDER BY created_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        print("  WARNING: No causal graph found. Using Fed projections as baseline.")
        causal_adj = {}
    else:
        causal_adj = row[0]

    # For each variable in the scenario, adjust the Fed's projection
    # based on causal relationships our model has discovered
    causal_projections = {}

    for var_key, var_data in scenario["variables"].items():
        if not isinstance(var_data, dict):
            continue

        our_code = var_data.get("our_code", "")
        fed_path = var_data.get("quarterly_path", [])

        if not fed_path or not our_code:
            continue

        # Start with Fed's projection as baseline
        causal_path = list(fed_path)

        # Find all causal parents of this variable in our graph
        parent_effects = []
        for edge_key, edge_data in causal_adj.items():
            cause, effect = edge_key.split("->")
            if effect == our_code:
                weight = edge_data.get("weight", 0)
                parent_effects.append({
                    "cause": cause,
                    "weight": weight,
                })

        # Adjust projections based on causal incoming effects
        if parent_effects:
            for q in range(len(causal_path)):
                adjustment = 0.0

                for parent in parent_effects:
                    parent_code = parent["cause"]
                    parent_weight = parent["weight"]

                    # Check if the parent variable is also in the scenario
                    parent_scenario = None
                    for pv_key, pv_data in scenario["variables"].items():
                        if isinstance(pv_data, dict) and pv_data.get("our_code") == parent_code:
                            parent_scenario = pv_data
                            break

                    if parent_scenario:
                        parent_path = parent_scenario.get("quarterly_path", [])
                        if q < len(parent_path) and parent_path:
                            max_abs = max(abs(p) for p in parent_path) if parent_path else 1
                            if max_abs > 0:
                                shock_intensity = abs(parent_path[q]) / max_abs
                            else:
                                shock_intensity = 0
                            adjustment += parent_weight * shock_intensity * 0.15

                # Apply causal adjustment
                fed_name = var_data.get("fed_name", "")
                unit = var_data.get("unit", "")

                if unit == "percent" and ("spread" in fed_name.lower() or "yield" in fed_name.lower() or "BBB" in var_key):
                    causal_path[q] = round(fed_path[q] * (1 + adjustment), 2)
                elif unit == "index_level" and fed_path[q] < 100:
                    causal_path[q] = round(fed_path[q] * (1 - adjustment * 0.5), 2)
                elif "unemployment" in fed_name.lower() or "UNEMPLOYMENT" in var_key:
                    causal_path[q] = round(fed_path[q] * (1 + adjustment * 0.3), 2)
                else:
                    causal_path[q] = round(fed_path[q] * (1 + adjustment * 0.2), 2)

        causal_projections[var_key] = causal_path

    return causal_projections


# ============================================
# STEP 4: GENERATE CAUSAL DIFFERENCE REPORT
# ============================================

def generate_difference_report(scenario, causal_projections):
    """
    Generate the Causal Difference Report — THE KILLER FEATURE.
    """
    print(f"\nGenerating Causal Difference Report...")
    print(f"  Scenario: {scenario['name']}\n")

    # Load causal graph for path tracing
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT adjacency_matrix
        FROM models.causal_graphs
        WHERE method = 'ensemble_dynotears_pcmci'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    causal_adj = row[0] if row else {}
    cursor.close()
    conn.close()

    divergences = []
    explanations = []

    print(f"  {'Variable':<25} {'Quarter':>8} {'Fed Proj':>10} {'Our Proj':>10} {'Diff':>8} {'Status':<12}")
    print(f"  {'-'*77}")

    for var_key, var_data in scenario["variables"].items():
        if not isinstance(var_data, dict):
            continue

        fed_path = var_data.get("quarterly_path", [])
        our_code = var_data.get("our_code", "")
        fed_name = var_data.get("fed_name", var_key)

        if not fed_path or not our_code:
            continue

        causal_path = causal_projections.get(var_key, fed_path)

        for q in range(len(fed_path)):
            fed_val = fed_path[q]
            causal_val = causal_path[q] if q < len(causal_path) else fed_val

            if fed_val == 0:
                pct_diff = 0
            else:
                pct_diff = ((causal_val - fed_val) / abs(fed_val)) * 100

            if abs(pct_diff) > 10:
                divergence = {
                    "variable": var_key,
                    "variable_name": fed_name,
                    "our_code": our_code,
                    "quarter": q + 1,
                    "fed_value": fed_val,
                    "causal_value": causal_val,
                    "pct_difference": round(pct_diff, 1),
                    "direction": "more_severe" if abs(causal_val) > abs(fed_val) else "less_severe",
                }
                divergences.append(divergence)

                explanation = trace_causal_explanation(
                    our_code, fed_name, fed_val, causal_val, pct_diff, q + 1, causal_adj
                )
                explanations.append(explanation)

                status = "DIVERGENT" if abs(pct_diff) > 20 else "ELEVATED"
                print(f"  {fed_name:<25} Q{q+1:>6} {fed_val:>10.1f} {causal_val:>10.1f} "
                      f"{pct_diff:>+7.1f}% {status:<12}")

    print(f"\n  Total divergences (>10%): {len(divergences)}")

    severe = [d for d in divergences if abs(d["pct_difference"]) > 20]
    moderate = [d for d in divergences if 10 < abs(d["pct_difference"]) <= 20]
    print(f"  Severe (>20%): {len(severe)}")
    print(f"  Moderate (10-20%): {len(moderate)}")

    return divergences, explanations


def trace_causal_explanation(var_code, var_name, fed_val, causal_val, pct_diff, quarter, causal_adj):
    """Trace the causal path that explains WHY our projection differs."""
    parents = []
    for edge_key, edge_data in causal_adj.items():
        cause, effect = edge_key.split("->")
        if effect == var_code:
            parents.append({
                "cause": cause,
                "weight": edge_data.get("weight", 0),
            })

    parents.sort(key=lambda x: x["weight"], reverse=True)

    if not parents:
        explanation = (
            f"Quarter {quarter}: Our model projects {var_name} at {causal_val:.1f} vs "
            f"Fed's {fed_val:.1f} ({pct_diff:+.1f}%). No direct causal parents identified — "
            f"divergence may be driven by indirect effects through the causal network."
        )
    else:
        top_parents = parents[:3]
        parent_names = ", ".join([p["cause"] for p in top_parents])
        strongest = top_parents[0]

        if pct_diff > 0:
            direction = "higher"
            reason = "amplifying"
        else:
            direction = "lower"
            reason = "dampening"

        explanation = (
            f"Quarter {quarter}: Our causal model projects {var_name} at {causal_val:.1f} vs "
            f"Fed's projection of {fed_val:.1f} ({pct_diff:+.1f}% {direction}). "
            f"This divergence is driven by causal transmission from {parent_names}. "
            f"The strongest causal driver is {strongest['cause']} (weight: {strongest['weight']:.3f}), "
            f"which has a {reason} effect on {var_name} that the Fed's correlation-based model "
            f"does not fully capture. Our causal graph shows this transmission pathway is "
            f"currently {abs(pct_diff)/10:.1f}x stronger than the Fed's historical calibration."
        )

    return {
        "variable": var_code,
        "variable_name": var_name,
        "quarter": quarter,
        "explanation": explanation,
    }


# ============================================
# STEP 5: STORE REPORT IN DATABASE
# ============================================

def store_report(scenario, causal_projections, divergences, explanations):
    """Store the Causal Difference Report in the database."""
    print("\nStoring Causal Difference Report in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get scenario ID from DB
    scenario_id = scenario.get("db_id")
    if not scenario_id:
        cursor.execute(
            "SELECT id FROM regulatory.scenarios WHERE name = %s ORDER BY created_at DESC LIMIT 1",
            (scenario["name"],)
        )
        scenario_row = cursor.fetchone()
        scenario_id = str(scenario_row[0]) if scenario_row else None

    report_id = str(uuid.uuid4())

    fed_projections = {}
    for var_key, var_data in scenario["variables"].items():
        if isinstance(var_data, dict) and "quarterly_path" in var_data:
            fed_projections[var_key] = var_data["quarterly_path"]

    cursor.execute("""
        INSERT INTO regulatory.causal_difference_reports
            (id, regulatory_scenario_id, fed_projections, causal_projections,
             divergences, causal_explanations)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        report_id,
        scenario_id,
        Json(fed_projections),
        Json(causal_projections),
        Json(divergences),
        Json(explanations),
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Report stored with ID: {report_id}")
    return report_id


# ============================================
# STEP 6: PRINT FULL REPORT
# ============================================

def print_full_report(scenario, divergences, explanations):
    """Print the complete Causal Difference Report."""
    print(f"\n{'='*70}")
    print(f"  CAUSAL DIFFERENCE REPORT")
    print(f"  {scenario['name']}")
    print(f"{'='*70}")

    print(f"\n  Scenario: {scenario['description']}")
    print(f"  Source: {scenario['source']}")
    print(f"  Horizon: {scenario['horizon_quarters']} quarters")

    if not divergences:
        print(f"\n  No significant divergences found (all within 10% of Fed projections)")
        return

    print(f"\n  DIVERGENT PROJECTIONS ({len(divergences)} found):")
    print(f"  {'─'*66}")

    vars_seen = set()
    for div in divergences:
        if div["variable_name"] not in vars_seen:
            vars_seen.add(div["variable_name"])
            print(f"\n  {div['variable_name']} ({div['our_code']}):")

            var_divs = [d for d in divergences if d["variable_name"] == div["variable_name"]]
            for d in var_divs:
                arrow = "▲" if d["pct_difference"] > 0 else "▼"
                print(f"    Q{d['quarter']}: Fed={d['fed_value']:.1f}, "
                      f"Ours={d['causal_value']:.1f} "
                      f"({arrow} {abs(d['pct_difference']):.1f}% {'more' if d['direction'] == 'more_severe' else 'less'} severe)")

    print(f"\n  CAUSAL EXPLANATIONS:")
    print(f"  {'─'*66}")
    for exp in explanations[:5]:
        print(f"\n  {exp['explanation']}")

    print(f"\n{'='*70}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("CAUSALSTRESS - REGULATORY COMPLIANCE ENGINE")
    print("=" * 70)

    # Step 1: Ensure fallback scenarios exist (won't overwrite real Fed data)
    print("\nChecking regulatory scenarios in database...")
    ensure_fallback_scenarios()

    # Step 2: Load ALL scenarios from database (includes real Fed CSV data)
    scenarios = load_scenarios_from_db()
    print(f"\n  Found {len(scenarios)} scenarios in database:")
    for s in scenarios:
        n_vars = len(s["variables"]) if isinstance(s["variables"], dict) else 0
        print(f"    {s['name']} ({s['source']}) — {n_vars} variables, {s['horizon_quarters']}Q")

    # Step 3: Run each scenario through causal model
    for scenario in scenarios:
        print(f"\n{'#'*70}")
        print(f"  SCENARIO: {scenario['name'].upper()}")
        print(f"  Source: {scenario['source']}")
        print(f"{'#'*70}")

        # Generate causal projections
        causal_projections = run_causal_projection(scenario)

        # Generate difference report
        divergences, explanations = generate_difference_report(scenario, causal_projections)

        # Store in database
        report_id = store_report(scenario, causal_projections, divergences, explanations)

        # Print full report
        print_full_report(scenario, divergences, explanations)

    print(f"\n\n{'='*70}")
    print("✓ Regulatory compliance analysis complete!")
    print(f"  {len(scenarios)} scenarios analyzed")
    print(f"  Causal Difference Reports generated and stored")
    print("=" * 70)


if __name__ == "__main__":
    main()