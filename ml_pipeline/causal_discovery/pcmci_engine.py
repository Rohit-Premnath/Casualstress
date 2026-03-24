"""
PCMCI Causal Discovery Engine
===============================
Discovers causal relationships using the PCMCI algorithm from tigramite.

PCMCI is specifically designed for time-series causal discovery.
It uses conditional independence testing to find causal links,
which is a fundamentally different approach from DYNOTEARS (Lasso).

By comparing both methods, edges that appear in BOTH are highly
reliable — this is our ensemble approach.

Key advantage of PCMCI over DYNOTEARS:
- Handles autocorrelation properly (common in financial data)
- Uses rigorous statistical testing (not just regularization)
- Provides p-values for each edge
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

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

# Maximum time lag (in trading days)
MAX_LAG = 5

# Significance level for conditional independence tests
ALPHA = 0.05  # Only keep edges with p-value < 0.05

# Minimum absolute value of partial correlation to keep
MIN_EFFECT_SIZE = 0.03


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
# STEP 1: LOAD DATA
# ============================================

def load_processed_data():
    """Load transformed time-series data from the processed schema."""
    print("Loading processed data...")

    conn = get_db_connection()

    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)

    conn.close()

    pivoted = df.pivot_table(
        index="date",
        columns="variable_code",
        values="transformed_value"
    )

    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()

    # Drop columns with too many NaNs
    threshold = len(pivoted) * 0.7
    pivoted = pivoted.dropna(axis=1, thresh=int(threshold))
    pivoted = pivoted.dropna()

    print(f"  Loaded {len(pivoted)} days x {len(pivoted.columns)} variables")

    return pivoted


# ============================================
# STEP 2: RUN PCMCI
# ============================================

def run_pcmci(df):
    """
    Run the PCMCI algorithm on the processed data.

    PCMCI works in two phases:
    1. PC phase: removes spurious links using conditional independence tests
    2. MCI phase: tests remaining links with momentary conditional independence

    This is more statistically rigorous than Lasso — it gives actual p-values.
    """
    print(f"\nRunning PCMCI (max_lag={MAX_LAG}, alpha={ALPHA})...")
    print("  This may take 10-20 minutes...\n")

    variable_names = list(df.columns)
    data_values = df.values

    # Standardize
    means = data_values.mean(axis=0)
    stds = data_values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (data_values - means) / stds

    # Create tigramite dataframe
    dataframe = pp.DataFrame(
        data=standardized,
        var_names=variable_names,
    )

    # Set up the conditional independence test
    # ParCorr = Partial Correlation (assumes linear relationships)
    # This is the most common choice for financial data
    parcorr = ParCorr(significance="analytic")

    # Create PCMCI object
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=1,  # 0=silent, 1=basic, 2=detailed
    )

    # Run PCMCI
    # tau_min=0 allows contemporaneous links
    # tau_max=MAX_LAG sets maximum lag
    print("  Phase 1: PC algorithm (removing spurious links)...")
    print("  Phase 2: MCI tests (testing remaining links)...\n")

    results = pcmci.run_pcmci(
        tau_min=0,
        tau_max=MAX_LAG,
        pc_alpha=ALPHA,       # Significance level for PC phase
        alpha_level=ALPHA,    # Significance level for MCI phase
    )

    # Extract results
    p_matrix = results["p_matrix"]       # (d x d x (max_lag+1)) p-values
    val_matrix = results["val_matrix"]   # (d x d x (max_lag+1)) partial correlations

    return p_matrix, val_matrix, variable_names, pcmci


# ============================================
# STEP 3: EXTRACT SIGNIFICANT EDGES
# ============================================

def extract_edges(p_matrix, val_matrix, variable_names):
    """
    Extract significant causal edges from PCMCI results.

    An edge is kept if:
    - p-value < ALPHA (statistically significant)
    - |partial correlation| > MIN_EFFECT_SIZE (meaningful effect)
    - Not a self-loop at lag 0
    """
    print("\nExtracting significant edges...")

    d = len(variable_names)
    edges = []

    for i in range(d):
        for j in range(d):
            for lag in range(MAX_LAG + 1):
                p_val = p_matrix[i, j, lag]
                strength = val_matrix[i, j, lag]

                # Skip non-significant edges
                if p_val >= ALPHA:
                    continue

                # Skip weak effects
                if abs(strength) < MIN_EFFECT_SIZE:
                    continue

                # Skip self-loops at lag 0
                if i == j and lag == 0:
                    continue

                edges.append({
                    "cause": variable_names[i],
                    "effect": variable_names[j],
                    "lag": lag,
                    "strength": float(abs(strength)),
                    "raw_strength": float(strength),
                    "p_value": float(p_val),
                    "edge_type": "contemporaneous" if lag == 0 else "lagged",
                })

    # Sort by strength
    edges.sort(key=lambda x: x["strength"], reverse=True)

    print(f"  Found {len(edges)} significant edges")

    # Print top 20
    print(f"\n  Top 20 PCMCI causal links:")
    print(f"  {'Cause':<20} {'Effect':<20} {'Strength':>10} {'Lag':>5} {'p-value':>10}")
    print("  " + "-" * 68)
    for edge in edges[:20]:
        print(f"  {edge['cause']:<20} {edge['effect']:<20} "
              f"{edge['strength']:>10.4f} {edge['lag']:>5} {edge['p_value']:>10.6f}")

    return edges


# ============================================
# STEP 4: COMPARE WITH DYNOTEARS
# ============================================

def compare_with_dynotears(pcmci_edges, variable_names):
    """
    Load the DYNOTEARS graph from database and compare edges.
    Find consensus edges that appear in BOTH methods.
    """
    print("\nComparing PCMCI with DYNOTEARS results...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Load the most recent DYNOTEARS graph
    cursor.execute("""
        SELECT adjacency_matrix, confidence_scores
        FROM models.causal_graphs
        WHERE method = 'dynotears_lasso'
        ORDER BY created_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        print("  WARNING: No DYNOTEARS graph found. Run dynotears_engine.py first.")
        return pcmci_edges, [], []

    dyno_adj = row[0]   # adjacency matrix as dict
    dyno_conf = row[1]  # confidence scores as dict

    # Build set of DYNOTEARS edges (cause->effect pairs)
    dyno_edge_set = set()
    for key in dyno_adj:
        cause, effect = key.split("->")
        dyno_edge_set.add((cause, effect))

    # Build set of PCMCI edges
    pcmci_edge_set = set()
    for edge in pcmci_edges:
        pcmci_edge_set.add((edge["cause"], edge["effect"]))

    # Find consensus (in both)
    consensus = dyno_edge_set & pcmci_edge_set

    # Only in DYNOTEARS
    only_dyno = dyno_edge_set - pcmci_edge_set

    # Only in PCMCI
    only_pcmci = pcmci_edge_set - dyno_edge_set

    print(f"\n  DYNOTEARS edges:    {len(dyno_edge_set)}")
    print(f"  PCMCI edges:        {len(pcmci_edge_set)}")
    print(f"  ─────────────────────────────")
    print(f"  Consensus (both):   {len(consensus)}  ← HIGH confidence")
    print(f"  Only DYNOTEARS:     {len(only_dyno)}")
    print(f"  Only PCMCI:         {len(only_pcmci)}")
    print(f"  Agreement rate:     {len(consensus)/max(len(dyno_edge_set & pcmci_edge_set | dyno_edge_set | pcmci_edge_set), 1)*100:.1f}%")

    # Print consensus edges
    if consensus:
        print(f"\n  Top consensus edges (confirmed by both methods):")
        consensus_details = []
        for edge in pcmci_edges:
            if (edge["cause"], edge["effect"]) in consensus:
                # Get DYNOTEARS weight
                key = f"{edge['cause']}->{edge['effect']}"
                dyno_weight = dyno_adj.get(key, {}).get("weight", 0)
                dyno_confidence = dyno_conf.get(key, 0)
                consensus_details.append({
                    **edge,
                    "dyno_weight": dyno_weight,
                    "dyno_confidence": dyno_confidence,
                })

        consensus_details.sort(key=lambda x: x["strength"], reverse=True)

        print(f"  {'Cause':<18} {'Effect':<18} {'PCMCI':>8} {'DYNO':>8} {'DYNO Conf':>10}")
        print("  " + "-" * 66)
        for e in consensus_details[:25]:
            print(f"  {e['cause']:<18} {e['effect']:<18} "
                  f"{e['strength']:>8.4f} {e['dyno_weight']:>8.4f} "
                  f"{e['dyno_confidence']:>9.0%}")

    return pcmci_edges, list(consensus), consensus_details if consensus else []


# ============================================
# STEP 5: BUILD ENSEMBLE GRAPH
# ============================================

def build_ensemble_graph(pcmci_edges, consensus_edges, variable_names):
    """
    Build the final ensemble causal graph.

    Rules for inclusion:
    1. Edge appears in BOTH DYNOTEARS and PCMCI → automatically included
    2. Edge appears in only one method but with very high confidence → included
    3. Everything else → excluded
    """
    import networkx as nx

    print("\nBuilding ensemble causal graph...")

    G = nx.DiGraph()

    # Add all variables as nodes
    for name in variable_names:
        G.add_node(name)

    consensus_set = set()
    for edge in consensus_edges:
        consensus_set.add((edge["cause"], edge["effect"]))

    # Add consensus edges (highest priority)
    for edge in consensus_edges:
        G.add_edge(
            edge["cause"],
            edge["effect"],
            weight=float(edge["strength"]),
            pcmci_strength=float(edge["strength"]),
            pcmci_pvalue=float(edge["p_value"]),
            dyno_weight=float(edge.get("dyno_weight", 0)),
            dyno_confidence=float(edge.get("dyno_confidence", 0)),
            lag=edge["lag"],
            edge_type=edge["edge_type"],
            method="consensus",
        )

    # Add strong PCMCI-only edges (p-value very low and strong effect)
    for edge in pcmci_edges:
        pair = (edge["cause"], edge["effect"])
        if pair not in consensus_set:
            if edge["p_value"] < 0.01 and edge["strength"] > 0.1:
                if not G.has_edge(edge["cause"], edge["effect"]):
                    G.add_edge(
                        edge["cause"],
                        edge["effect"],
                        weight=float(edge["strength"]),
                        pcmci_strength=float(edge["strength"]),
                        pcmci_pvalue=float(edge["p_value"]),
                        dyno_weight=0.0,
                        dyno_confidence=0.0,
                        lag=edge["lag"],
                        edge_type=edge["edge_type"],
                        method="pcmci_only",
                    )

    # Count by method
    consensus_count = sum(1 for _, _, d in G.edges(data=True) if d["method"] == "consensus")
    pcmci_only_count = sum(1 for _, _, d in G.edges(data=True) if d["method"] == "pcmci_only")

    print(f"  Ensemble graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"    Consensus edges: {consensus_count}")
    print(f"    PCMCI-only strong edges: {pcmci_only_count}")

    return G


# ============================================
# STEP 6: STORE ENSEMBLE GRAPH
# ============================================

def store_ensemble_graph(G, variable_names):
    """Store the ensemble causal graph in the database."""
    print("\nStoring ensemble graph in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    graph_id = str(uuid.uuid4())

    adjacency = {}
    confidence = {}
    for cause, effect, data in G.edges(data=True):
        key = f"{cause}->{effect}"
        adjacency[key] = {
            "weight": data["weight"],
            "pcmci_strength": data["pcmci_strength"],
            "pcmci_pvalue": data["pcmci_pvalue"],
            "dyno_weight": data["dyno_weight"],
            "dyno_confidence": data["dyno_confidence"],
            "lag": data["lag"],
            "edge_type": data["edge_type"],
            "method": data["method"],
        }
        # Confidence = average of DYNOTEARS bootstrap confidence and PCMCI significance
        if data["method"] == "consensus":
            conf = (data["dyno_confidence"] + (1 - data["pcmci_pvalue"])) / 2
        else:
            conf = 1 - data["pcmci_pvalue"]
        confidence[key] = conf

    cursor.execute("""
        SELECT MIN(date), MAX(date) FROM processed.time_series_data
    """)
    date_range = cursor.fetchone()

    cursor.execute("""
        INSERT INTO models.causal_graphs
            (id, method, variables, adjacency_matrix, confidence_scores,
             structural_constraints, date_range_start, date_range_end)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        graph_id,
        "ensemble_dynotears_pcmci",
        Json(variable_names),
        Json(adjacency),
        Json(confidence),
        Json({"note": "Ensemble of DYNOTEARS Lasso and PCMCI ParCorr"}),
        date_range[0],
        date_range[1],
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Ensemble graph stored with ID: {graph_id}")
    return graph_id


# ============================================
# STEP 7: EXPORT
# ============================================

def export_ensemble_json(G, filepath="ensemble_causal_graph.json"):
    """Export ensemble graph as JSON for frontend."""
    print(f"\nExporting ensemble graph to {filepath}...")

    nodes = []
    for node in G.nodes():
        nodes.append({
            "id": node,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
        })

    edges = []
    for cause, effect, data in G.edges(data=True):
        edges.append({
            "source": cause,
            "target": effect,
            "weight": data["weight"],
            "lag": data["lag"],
            "method": data["method"],
            "pcmci_pvalue": data["pcmci_pvalue"],
            "dyno_confidence": data["dyno_confidence"],
            "edge_type": data["edge_type"],
        })

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "method": "ensemble_dynotears_pcmci",
            "created_at": datetime.now().isoformat(),
        }
    }

    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"  Exported {len(nodes)} nodes and {len(edges)} edges")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - PCMCI CAUSAL DISCOVERY ENGINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_processed_data()

    # Step 2: Run PCMCI
    p_matrix, val_matrix, variable_names, pcmci_obj = run_pcmci(df)

    # Step 3: Extract significant edges
    pcmci_edges = extract_edges(p_matrix, val_matrix, variable_names)

    # Step 4: Compare with DYNOTEARS
    pcmci_edges, consensus, consensus_details = compare_with_dynotears(
        pcmci_edges, variable_names
    )

    # Step 5: Build ensemble graph
    G = build_ensemble_graph(pcmci_edges, consensus_details, variable_names)

    # Step 6: Store in database
    graph_id = store_ensemble_graph(G, variable_names)

    # Step 7: Export for frontend
    export_ensemble_json(G)

    print("\n✓ PCMCI analysis and ensemble graph complete!")
    print(f"  Ensemble Graph ID: {graph_id}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print("=" * 60)


if __name__ == "__main__":
    main()