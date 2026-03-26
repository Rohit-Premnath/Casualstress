"""
Regime-Conditional Causal Graphs
==================================
Re-runs causal discovery WITHIN each regime to capture how
the causal structure of the economy changes across market states.

This is our KEY INNOVATION that nobody else has built.
The January 2025 ATSCM paper explicitly states:
"Time series causal methods do not handle regime changes
in learned causal graphs." — We fill that gap.

Key insight: During a crisis, new causal links APPEAR that
don't exist in calm times (contagion effects), and some
calm-time links DISAPPEAR or reverse.

For example:
- In CALM: oil prices → energy stocks (sector-specific)
- In CRISIS: oil prices → ALL stocks (contagion, everything correlated)
- In CALM: VIX has weak links to most assets
- In CRISIS: VIX becomes a central driver of everything
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import networkx as nx
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

MAX_LAG = 3  # Shorter lag for regime-specific (less data per regime)
EDGE_THRESHOLD = 0.05

# Minimum number of days needed in a regime to run causal discovery
MIN_REGIME_DAYS = 200


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
# STEP 1: LOAD DATA WITH REGIME LABELS
# ============================================

def load_data_with_regimes():
    """Load processed time-series data and merge with regime labels."""
    print("Loading processed data with regime labels...")

    conn = get_db_connection()

    # Load processed data
    ts_df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)

    # Load regime labels
    regime_df = pd.read_sql("""
        SELECT date, regime_label, regime_name
        FROM models.regimes
        ORDER BY date
    """, conn)

    conn.close()

    # Pivot time-series to wide format
    pivoted = ts_df.pivot_table(
        index="date",
        columns="variable_code",
        values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)

    # Drop columns with too many NaN
    threshold = len(pivoted) * 0.7
    pivoted = pivoted.dropna(axis=1, thresh=int(threshold))
    pivoted = pivoted.dropna()

    # Merge with regime labels
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df = regime_df.set_index("date")

    merged = pivoted.join(regime_df, how="inner")

    print(f"  Loaded {len(merged)} days x {len(pivoted.columns)} variables")
    print(f"  Regime distribution:")
    regime_counts = merged["regime_name"].value_counts()
    for name, count in regime_counts.items():
        print(f"    {name}: {count} days ({count/len(merged)*100:.1f}%)")

    return merged, list(pivoted.columns)


# ============================================
# STEP 2: RUN CAUSAL DISCOVERY PER REGIME
# ============================================

def discover_regime_graph(data, variable_names, regime_name):
    """
    Run Lasso-based causal discovery on data from a single regime.

    Same approach as DYNOTEARS but on regime-specific data only.
    """
    d = len(variable_names)
    values = data[variable_names].values

    # Standardize
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    # Create current and lagged matrices
    T = len(standardized) - MAX_LAG
    X = standardized[MAX_LAG:]

    X_lags = []
    for lag in range(1, MAX_LAG + 1):
        X_lag = standardized[MAX_LAG - lag: len(standardized) - lag]
        X_lags.append(X_lag)
    X_lag = np.hstack(X_lags)

    # Run Lasso for each target variable
    W_contemp = np.zeros((d, d))
    W_lagged = np.zeros((MAX_LAG, d, d))

    for j in range(d):
        y = X[:, j]
        contemp_indices = [i for i in range(d) if i != j]
        X_contemp = X[:, contemp_indices]
        features = np.hstack([X_contemp, X_lag])

        try:
            lasso = LassoCV(cv=3, max_iter=10000, n_jobs=-1, random_state=42)
            lasso.fit(features, y)

            n_contemp = len(contemp_indices)
            contemp_coefs = lasso.coef_[:n_contemp]
            for idx, i in enumerate(contemp_indices):
                W_contemp[i, j] = contemp_coefs[idx]

            lag_coefs = lasso.coef_[n_contemp:]
            for lag_idx in range(MAX_LAG):
                start = lag_idx * d
                end = start + d
                W_lagged[lag_idx, :, j] = lag_coefs[start:end]
        except Exception:
            continue

    # Threshold
    W_contemp[np.abs(W_contemp) < EDGE_THRESHOLD] = 0
    for lag_idx in range(MAX_LAG):
        W_lagged[lag_idx][np.abs(W_lagged[lag_idx]) < EDGE_THRESHOLD] = 0

    # Build NetworkX graph
    G = nx.DiGraph()
    for name in variable_names:
        G.add_node(name)

    for i in range(d):
        for j in range(d):
            if W_contemp[i, j] != 0:
                G.add_edge(variable_names[i], variable_names[j],
                           weight=float(abs(W_contemp[i, j])),
                           raw_weight=float(W_contemp[i, j]),
                           lag=0, edge_type="contemporaneous")

    for lag_idx in range(MAX_LAG):
        for i in range(d):
            for j in range(d):
                if W_lagged[lag_idx, i, j] != 0:
                    if not G.has_edge(variable_names[i], variable_names[j]):
                        G.add_edge(variable_names[i], variable_names[j],
                                   weight=float(abs(W_lagged[lag_idx, i, j])),
                                   raw_weight=float(W_lagged[lag_idx, i, j]),
                                   lag=lag_idx + 1, edge_type="lagged")
                    elif abs(W_lagged[lag_idx, i, j]) > G[variable_names[i]][variable_names[j]]["weight"]:
                        G[variable_names[i]][variable_names[j]].update({
                            "weight": float(abs(W_lagged[lag_idx, i, j])),
                            "raw_weight": float(W_lagged[lag_idx, i, j]),
                            "lag": lag_idx + 1, "edge_type": "lagged"})

    return G


def run_all_regimes(merged_data, variable_names):
    """Run causal discovery for each regime that has enough data."""
    print("\nRunning causal discovery per regime...\n")

    regime_graphs = {}
    regimes = merged_data["regime_name"].unique()

    for regime_name in sorted(regimes):
        regime_data = merged_data[merged_data["regime_name"] == regime_name]

        if len(regime_data) < MIN_REGIME_DAYS:
            print(f"  {regime_name.upper()}: {len(regime_data)} days — SKIPPED (need {MIN_REGIME_DAYS}+)")
            continue

        print(f"  {regime_name.upper()}: {len(regime_data)} days — running causal discovery...", end=" ")
        G = discover_regime_graph(regime_data, variable_names, regime_name)
        regime_graphs[regime_name] = G
        print(f"found {G.number_of_edges()} edges")

    return regime_graphs


# ============================================
# STEP 3: ANALYZE STRUCTURAL DIFFERENCES
# ============================================

def analyze_differences(regime_graphs, variable_names):
    """
    Compare causal structures across regimes to find:
    1. Edges that APPEAR during crises (contagion)
    2. Edges that DISAPPEAR during crises
    3. Edges that STRENGTHEN during crises
    4. Edges that REVERSE during crises
    """
    print("\n" + "=" * 60)
    print("  REGIME-DEPENDENT STRUCTURAL CHANGES")
    print("=" * 60)

    regime_names = sorted(regime_graphs.keys())

    if len(regime_names) < 2:
        print("  Need at least 2 regimes for comparison")
        return {}

    # Find the calmest and most stressed regimes
    calm_regime = regime_names[0]  # sorted alphabetically, "calm" comes first
    stress_regime = regime_names[-1]  # "stressed" comes last

    # Check if we have specific names
    for r in regime_names:
        if "calm" in r:
            calm_regime = r
        if "stressed" in r or "crisis" in r:
            stress_regime = r

    G_calm = regime_graphs.get(calm_regime)
    G_stress = regime_graphs.get(stress_regime)

    if G_calm is None or G_stress is None:
        print("  Cannot compare: missing calm or stressed graph")
        return {}

    calm_edges = set(G_calm.edges())
    stress_edges = set(G_stress.edges())

    # Edges in both
    shared = calm_edges & stress_edges

    # Contagion edges: appear ONLY in stress
    contagion = stress_edges - calm_edges

    # Disappearing edges: exist in calm, gone in stress
    disappearing = calm_edges - stress_edges

    print(f"\n  Comparing {calm_regime.upper()} vs {stress_regime.upper()}:\n")
    print(f"  {calm_regime.upper()} edges:       {len(calm_edges)}")
    print(f"  {stress_regime.upper()} edges:     {len(stress_edges)}")
    print(f"  Shared edges:          {len(shared)}")
    print(f"  Contagion (stress-only): {len(contagion)}  ← NEW links during stress")
    print(f"  Disappearing (calm-only): {len(disappearing)}")

    # Print top contagion edges (strongest new links during stress)
    if contagion:
        contagion_details = []
        for cause, effect in contagion:
            data = G_stress[cause][effect]
            contagion_details.append({
                "cause": cause,
                "effect": effect,
                "weight": data["weight"],
                "lag": data["lag"],
            })
        contagion_details.sort(key=lambda x: x["weight"], reverse=True)

        print(f"\n  Top 15 CONTAGION edges (appear during {stress_regime}):")
        print(f"  {'Cause':<20} {'Effect':<20} {'Weight':>8} {'Lag':>5}")
        print("  " + "-" * 56)
        for e in contagion_details[:15]:
            print(f"  {e['cause']:<20} {e['effect']:<20} {e['weight']:>8.4f} {e['lag']:>5}")

    # Print strengthening edges (exist in both but much stronger in stress)
    if shared:
        strengthening = []
        for cause, effect in shared:
            calm_weight = G_calm[cause][effect]["weight"]
            stress_weight = G_stress[cause][effect]["weight"]
            if stress_weight > calm_weight * 1.5:  # 50% stronger
                strengthening.append({
                    "cause": cause,
                    "effect": effect,
                    "calm_weight": calm_weight,
                    "stress_weight": stress_weight,
                    "ratio": stress_weight / calm_weight if calm_weight > 0 else 0,
                })
        strengthening.sort(key=lambda x: x["ratio"], reverse=True)

        if strengthening:
            print(f"\n  Top 10 STRENGTHENING edges (stronger during {stress_regime}):")
            print(f"  {'Cause':<18} {'Effect':<18} {'Calm':>8} {'Stress':>8} {'Ratio':>7}")
            print("  " + "-" * 63)
            for e in strengthening[:10]:
                print(f"  {e['cause']:<18} {e['effect']:<18} "
                      f"{e['calm_weight']:>8.4f} {e['stress_weight']:>8.4f} {e['ratio']:>6.1f}x")

    # Compare all regimes edge counts
    print(f"\n  Edge counts across all regimes:")
    print(f"  {'Regime':<15} {'Edges':>7} {'Avg Weight':>12}")
    print("  " + "-" * 37)
    for regime_name in regime_names:
        G = regime_graphs[regime_name]
        edges = G.number_of_edges()
        if edges > 0:
            avg_w = np.mean([d["weight"] for _, _, d in G.edges(data=True)])
        else:
            avg_w = 0
        print(f"  {regime_name:<15} {edges:>7} {avg_w:>12.4f}")

    analysis = {
        "calm_regime": calm_regime,
        "stress_regime": stress_regime,
        "shared_edges": len(shared),
        "contagion_edges": len(contagion),
        "disappearing_edges": len(disappearing),
    }

    return analysis


# ============================================
# STEP 4: STORE REGIME GRAPHS IN DATABASE
# ============================================

def store_regime_graphs(regime_graphs, variable_names):
    """Store each regime's causal graph in the database."""
    print("\nStoring regime-conditional graphs in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MIN(date), MAX(date) FROM processed.time_series_data")
    date_range = cursor.fetchone()

    graph_ids = {}

    for regime_name, G in regime_graphs.items():
        graph_id = str(uuid.uuid4())

        adjacency = {}
        for cause, effect, data in G.edges(data=True):
            key = f"{cause}->{effect}"
            adjacency[key] = {
                "weight": data["weight"],
                "raw_weight": data["raw_weight"],
                "lag": data["lag"],
                "edge_type": data["edge_type"],
            }

        cursor.execute("""
            INSERT INTO models.causal_graphs
                (id, method, variables, adjacency_matrix, confidence_scores,
                 structural_constraints, date_range_start, date_range_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            graph_id,
            f"regime_{regime_name}",
            Json(variable_names),
            Json(adjacency),
            Json({}),
            Json({"regime": regime_name, "min_days": MIN_REGIME_DAYS, "max_lag": MAX_LAG}),
            date_range[0],
            date_range[1],
        ))

        graph_ids[regime_name] = graph_id
        print(f"  {regime_name}: stored with ID {graph_id} ({G.number_of_edges()} edges)")

    conn.commit()
    cursor.close()
    conn.close()

    return graph_ids


# ============================================
# STEP 5: EXPORT FOR FRONTEND
# ============================================

def export_regime_graphs(regime_graphs, analysis):
    """Export all regime graphs as a single JSON for the frontend."""
    filepath = "regime_causal_graphs.json"
    print(f"\nExporting regime-conditional graphs to {filepath}...")

    export_data = {
        "regimes": {},
        "analysis": analysis,
        "created_at": datetime.now().isoformat(),
    }

    for regime_name, G in regime_graphs.items():
        nodes = [{"id": n, "in_degree": G.in_degree(n), "out_degree": G.out_degree(n)}
                 for n in G.nodes()]

        edges = [{"source": u, "target": v, "weight": d["weight"],
                  "lag": d["lag"], "edge_type": d["edge_type"]}
                 for u, v, d in G.edges(data=True)]

        export_data["regimes"][regime_name] = {
            "nodes": nodes,
            "edges": edges,
            "n_edges": len(edges),
        }

    with open(filepath, "w") as f:
        json.dump(export_data, f)

    print(f"  Exported {len(regime_graphs)} regime graphs")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - REGIME-CONDITIONAL CAUSAL GRAPHS")
    print("=" * 60)

    # Step 1: Load data with regime labels
    merged_data, variable_names = load_data_with_regimes()

    # Step 2: Run causal discovery per regime
    regime_graphs = run_all_regimes(merged_data, variable_names)

    # Step 3: Analyze structural differences
    analysis = analyze_differences(regime_graphs, variable_names)

    # Step 4: Store in database
    graph_ids = store_regime_graphs(regime_graphs, variable_names)

    # Step 5: Export for frontend
    export_regime_graphs(regime_graphs, analysis)

    print("\n✓ Regime-conditional causal graphs complete!")
    print(f"  Regimes analyzed: {len(regime_graphs)}")
    for regime_name, G in regime_graphs.items():
        print(f"    {regime_name}: {G.number_of_edges()} edges")
    if analysis:
        print(f"  Contagion edges (stress-only): {analysis.get('contagion_edges', 0)}")
        print(f"  Disappearing edges (calm-only): {analysis.get('disappearing_edges', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()