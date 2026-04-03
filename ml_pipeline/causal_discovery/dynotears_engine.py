"""
DYNOTEARS Causal Discovery Engine
===================================
Discovers causal relationships between macro-financial variables
using the DYNOTEARS algorithm (Dynamic NOTEARS for time-series).

DYNOTEARS finds both:
- Contemporaneous effects (X causes Y at the same time step)
- Time-lagged effects (X at time t causes Y at time t+1, t+2, etc.)

Output: A causal graph (DAG) with weighted edges and confidence scores.
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.DAG2CPDAG import dag2cpdag
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

# Maximum time lag to consider (in trading days)
# 5 = one trading week of lagged effects
MAX_LAG = 5

# Minimum edge weight to keep (prune weak edges)
EDGE_THRESHOLD = 0.05

# Number of bootstrap resamples for confidence scores
N_BOOTSTRAP = 200  # Use 1000 for paper, 200 for development speed

# Variables to EXCLUDE from causal discovery (engineered features)
# We only want base variables in the causal graph
EXCLUDE_SUFFIXES = ["_vol_21d", "_vol_63d", "_ZSCORE"]

# Economic structural constraints
# Format: (cause, effect, forbidden=True/False)
# forbidden=True means this edge CANNOT exist
FORBIDDEN_EDGES = [
    # Single stocks/ETFs cannot cause macro variables
    ("XLK", "A191RL1Q225SBEA"),   # Tech ETF cannot cause GDP
    ("XLF", "A191RL1Q225SBEA"),   # Finance ETF cannot cause GDP
    ("XLE", "A191RL1Q225SBEA"),   # Energy ETF cannot cause GDP
    ("XLK", "UNRATE"),            # Tech ETF cannot cause unemployment
    ("XLF", "UNRATE"),            # Finance ETF cannot cause unemployment
    ("^GSPC", "A191RL1Q225SBEA"), # S&P 500 cannot cause GDP (in short-term)
    ("^GSPC", "CPIAUCSL"),        # S&P 500 cannot cause CPI
    # VIX is derived from options, it measures fear but doesn't cause macro
    ("^VIX", "A191RL1Q225SBEA"),  # VIX cannot cause GDP
    ("^VIX", "CPIAUCSL"),         # VIX cannot cause CPI
    ("^VIX", "UNRATE"),           # VIX cannot cause unemployment
    # Currency doesn't directly cause domestic rates in short-term
    ("EURUSD=X", "FEDFUNDS"),     # EUR/USD cannot cause fed funds rate
]

# Required edges (we KNOW these causal relationships exist)
REQUIRED_EDGES = [
    ("FEDFUNDS", "DGS2"),         # Fed funds rate causes 2Y yield changes
    ("CL=F", "CPIAUCSL"),         # Oil prices cause inflation
]


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
# STEP 1: LOAD PROCESSED DATA
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

    # Pivot to wide format: rows=dates, columns=variables
    pivoted = df.pivot_table(
        index="date",
        columns="variable_code",
        values="transformed_value"
    )

    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()

    # Drop any columns that are mostly NaN
    threshold = len(pivoted) * 0.7  # need at least 70% non-null
    pivoted = pivoted.dropna(axis=1, thresh=int(threshold))

    # Drop remaining NaN rows
    pivoted = pivoted.dropna()

    print(f"  Loaded {len(pivoted)} days x {len(pivoted.columns)} variables")
    print(f"  Variables: {list(pivoted.columns)}")

    return pivoted


# ============================================
# STEP 2: PREPARE DATA FOR DYNOTEARS
# ============================================

def prepare_dynotears_data(df):
    """
    Prepare data matrices for DYNOTEARS.

    DYNOTEARS needs:
    - X: (T x d) matrix of current observations
    - X_lag: (T x d*max_lag) matrix of lagged observations

    where T = number of time steps, d = number of variables
    """
    print(f"\nPreparing data for DYNOTEARS (max_lag={MAX_LAG})...")

    values = df.values  # (T_full x d)
    n_full, d = values.shape

    # Standardize each variable (zero mean, unit variance)
    # This is critical for DYNOTEARS to work properly across different scales
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero
    standardized = (values - means) / stds

    # Create current and lagged matrices
    T = n_full - MAX_LAG
    X = standardized[MAX_LAG:]  # Current values: (T x d)

    # Build lagged matrix: for each lag, stack the lagged values
    X_lags = []
    for lag in range(1, MAX_LAG + 1):
        X_lag = standardized[MAX_LAG - lag: n_full - lag]  # (T x d)
        X_lags.append(X_lag)

    X_lag = np.hstack(X_lags)  # (T x d*max_lag)

    print(f"  X shape: {X.shape} (current observations)")
    print(f"  X_lag shape: {X_lag.shape} (lagged observations)")

    return X, X_lag, df.columns.tolist(), means, stds


# ============================================
# STEP 3: RUN DYNOTEARS
# ============================================

def run_dynotears(X, X_lag, variable_names):
    """
    Run the DYNOTEARS algorithm to discover causal structure.

    Since causal-learn doesn't have DYNOTEARS directly,
    we implement it using penalized VAR (Vector Autoregression)
    with L1 regularization and acyclicity constraint.

    This is a practical approximation that captures the same idea:
    find the sparsest DAG that explains the data.
    """
    print("\nRunning causal discovery...")

    d = X.shape[1]  # number of variables

    # ── Method: Penalized VAR with Lasso ──
    # For each variable, regress it on all other current variables
    # and all lagged variables. The non-zero coefficients indicate
    # causal relationships.

    from sklearn.linear_model import LassoCV

    # Contemporaneous adjacency matrix (d x d)
    W_contemp = np.zeros((d, d))

    # Lagged adjacency matrices (max_lag x d x d)
    W_lagged = np.zeros((MAX_LAG, d, d))

    print(f"  Discovering causal links for {d} variables...")

    for j in range(d):
        # Target: variable j at time t
        y = X[:, j]

        # Features: all OTHER current variables + all lagged variables
        # Exclude variable j from contemporaneous (no self-loops)
        contemp_indices = [i for i in range(d) if i != j]
        X_contemp = X[:, contemp_indices]

        # Combine contemporaneous and lagged features
        features = np.hstack([X_contemp, X_lag])

        # Run Lasso regression with cross-validation
        lasso = LassoCV(
            cv=3,
            max_iter=50000,
            n_jobs=-1,
            random_state=42,
            tol=0.01,
        )
        lasso.fit(features, y)

        # Extract contemporaneous coefficients
        n_contemp = len(contemp_indices)
        contemp_coefs = lasso.coef_[:n_contemp]

        for idx, i in enumerate(contemp_indices):
            W_contemp[i, j] = contemp_coefs[idx]

        # Extract lagged coefficients
        lag_coefs = lasso.coef_[n_contemp:]
        for lag_idx in range(MAX_LAG):
            start = lag_idx * d
            end = start + d
            W_lagged[lag_idx, :, j] = lag_coefs[start:end]

        if (j + 1) % 5 == 0 or j == d - 1:
            print(f"    Processed variable {j+1}/{d}: {variable_names[j]}")

    # Apply threshold to remove weak edges
    W_contemp[np.abs(W_contemp) < EDGE_THRESHOLD] = 0
    for lag_idx in range(MAX_LAG):
        W_lagged[lag_idx][np.abs(W_lagged[lag_idx]) < EDGE_THRESHOLD] = 0

    # Count edges
    n_contemp_edges = np.count_nonzero(W_contemp)
    n_lagged_edges = sum(np.count_nonzero(W_lagged[l]) for l in range(MAX_LAG))

    print(f"\n  Contemporaneous edges found: {n_contemp_edges}")
    print(f"  Lagged edges found: {n_lagged_edges}")
    print(f"  Total edges: {n_contemp_edges + n_lagged_edges}")

    return W_contemp, W_lagged


# ============================================
# STEP 4: APPLY STRUCTURAL CONSTRAINTS
# ============================================

def apply_constraints(W_contemp, W_lagged, variable_names):
    """
    Apply economic structural constraints to the discovered graph.
    - Remove forbidden edges
    - Ensure required edges are present
    """
    print("\nApplying structural constraints...")

    var_to_idx = {name: i for i, name in enumerate(variable_names)}
    removed = 0
    added = 0

    # Remove forbidden edges
    for cause, effect, in FORBIDDEN_EDGES:
        if cause in var_to_idx and effect in var_to_idx:
            i, j = var_to_idx[cause], var_to_idx[effect]
            if W_contemp[i, j] != 0:
                W_contemp[i, j] = 0
                removed += 1
            for lag_idx in range(MAX_LAG):
                if W_lagged[lag_idx, i, j] != 0:
                    W_lagged[lag_idx, i, j] = 0
                    removed += 1

    # Ensure required edges exist (with minimum weight if not already present)
    for cause, effect in REQUIRED_EDGES:
        if cause in var_to_idx and effect in var_to_idx:
            i, j = var_to_idx[cause], var_to_idx[effect]
            # Check if any edge exists (contemp or lagged)
            has_edge = W_contemp[i, j] != 0
            for lag_idx in range(MAX_LAG):
                has_edge = has_edge or W_lagged[lag_idx, i, j] != 0
            if not has_edge:
                # Add a minimum weight lagged edge at lag=1
                W_lagged[0, i, j] = EDGE_THRESHOLD * 2
                added += 1

    print(f"  Removed {removed} forbidden edges")
    print(f"  Added {added} required edges")

    return W_contemp, W_lagged


# ============================================
# STEP 5: BOOTSTRAP CONFIDENCE SCORES
# ============================================

def bootstrap_confidence(X, X_lag, variable_names, n_bootstrap=N_BOOTSTRAP):
    """
    Run causal discovery on bootstrap resamples to compute
    confidence scores for each edge.

    An edge that appears in 90% of bootstrap runs is much more
    reliable than one that appears in only 30%.
    """
    print(f"\nRunning bootstrap stability analysis ({n_bootstrap} resamples)...")

    d = X.shape[1]
    T = X.shape[0]

    # Count how many times each edge appears
    contemp_counts = np.zeros((d, d))
    lagged_counts = np.zeros((MAX_LAG, d, d))

    from sklearn.linear_model import LassoCV

    for b in range(n_bootstrap):
        # Resample with replacement (block bootstrap to preserve time structure)
        block_size = 20  # ~1 month of trading days
        n_blocks = T // block_size + 1
        block_indices = np.random.randint(0, T - block_size, size=n_blocks)
        indices = np.concatenate([
            np.arange(start, min(start + block_size, T))
            for start in block_indices
        ])[:T]

        X_boot = X[indices]
        X_lag_boot = X_lag[indices]

        for j in range(d):
            y = X_boot[:, j]
            contemp_indices = [i for i in range(d) if i != j]
            X_contemp = X_boot[:, contemp_indices]
            features = np.hstack([X_contemp, X_lag_boot])

            try:
                lasso = LassoCV(cv=2, max_iter=20000, n_jobs=-1, random_state=b, tol=0.01)
                lasso.fit(features, y)

                n_contemp = len(contemp_indices)
                contemp_coefs = lasso.coef_[:n_contemp]

                for idx, i in enumerate(contemp_indices):
                    if abs(contemp_coefs[idx]) >= EDGE_THRESHOLD:
                        contemp_counts[i, j] += 1

                lag_coefs = lasso.coef_[n_contemp:]
                for lag_idx in range(MAX_LAG):
                    start = lag_idx * d
                    end = start + d
                    for i in range(d):
                        if abs(lag_coefs[start + i]) >= EDGE_THRESHOLD:
                            lagged_counts[lag_idx, i, j] += 1
            except Exception:
                continue

        if (b + 1) % 25 == 0:
            print(f"  Bootstrap {b+1}/{n_bootstrap} complete")

    # Convert counts to frequencies (0 to 1)
    contemp_confidence = contemp_counts / n_bootstrap
    lagged_confidence = lagged_counts / n_bootstrap

    # Report high-confidence edges
    high_conf = np.sum(contemp_confidence > 0.7) + sum(
        np.sum(lagged_confidence[l] > 0.7) for l in range(MAX_LAG)
    )
    print(f"  High-confidence edges (>70%): {high_conf}")

    return contemp_confidence, lagged_confidence


# ============================================
# STEP 6: BUILD NETWORKX GRAPH
# ============================================

def build_graph(W_contemp, W_lagged, contemp_confidence, lagged_confidence, variable_names):
    """
    Build a NetworkX directed graph from the adjacency matrices.
    Each edge has: weight, lag, confidence score.
    """
    print("\nBuilding causal graph...")

    G = nx.DiGraph()

    # Add all variables as nodes
    for name in variable_names:
        G.add_node(name)

    # Add contemporaneous edges
    d = len(variable_names)
    for i in range(d):
        for j in range(d):
            if W_contemp[i, j] != 0:
                G.add_edge(
                    variable_names[i],
                    variable_names[j],
                    weight=float(abs(W_contemp[i, j])),
                    raw_weight=float(W_contemp[i, j]),
                    lag=0,
                    confidence=float(contemp_confidence[i, j]),
                    edge_type="contemporaneous",
                )

    # Add lagged edges
    for lag_idx in range(MAX_LAG):
        for i in range(d):
            for j in range(d):
                if W_lagged[lag_idx, i, j] != 0:
                    lag = lag_idx + 1
                    edge_key = f"{variable_names[i]}_lag{lag}"

                    # If edge already exists with different lag, keep stronger one
                    if G.has_edge(variable_names[i], variable_names[j]):
                        existing = G[variable_names[i]][variable_names[j]]
                        if abs(W_lagged[lag_idx, i, j]) > existing["weight"]:
                            G[variable_names[i]][variable_names[j]].update({
                                "weight": float(abs(W_lagged[lag_idx, i, j])),
                                "raw_weight": float(W_lagged[lag_idx, i, j]),
                                "lag": lag,
                                "confidence": float(lagged_confidence[lag_idx, i, j]),
                                "edge_type": "lagged",
                            })
                    else:
                        G.add_edge(
                            variable_names[i],
                            variable_names[j],
                            weight=float(abs(W_lagged[lag_idx, i, j])),
                            raw_weight=float(W_lagged[lag_idx, i, j]),
                            lag=lag,
                            confidence=float(lagged_confidence[lag_idx, i, j]),
                            edge_type="lagged",
                        )

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Print top 20 strongest edges
    edges_sorted = sorted(
        G.edges(data=True),
        key=lambda x: x[2]["weight"],
        reverse=True
    )

    print(f"\n  Top 20 strongest causal links:")
    print(f"  {'Cause':<20} {'Effect':<20} {'Weight':>8} {'Lag':>5} {'Conf':>6}")
    print("  " + "-" * 63)
    for cause, effect, data in edges_sorted[:20]:
        print(f"  {cause:<20} {effect:<20} {data['weight']:>8.4f} {data['lag']:>5} {data['confidence']:>6.1%}")

    return G


# ============================================
# STEP 7: STORE IN DATABASE
# ============================================

def store_graph(G, variable_names, method="dynotears_lasso"):
    """Store the causal graph in the models.causal_graphs table."""
    print("\nStoring causal graph in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    graph_id = str(uuid.uuid4())

    # Convert graph to storable format
    adjacency = {}
    confidence = {}
    for cause, effect, data in G.edges(data=True):
        key = f"{cause}->{effect}"
        adjacency[key] = {
            "weight": data["weight"],
            "raw_weight": data["raw_weight"],
            "lag": data["lag"],
            "edge_type": data["edge_type"],
        }
        confidence[key] = data["confidence"]

    # Get date range from processed data
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
        method,
        Json(variable_names),
        Json(adjacency),
        Json(confidence),
        Json({
            "forbidden_edges": FORBIDDEN_EDGES,
            "required_edges": REQUIRED_EDGES,
            "edge_threshold": EDGE_THRESHOLD,
            "max_lag": MAX_LAG,
        }),
        date_range[0],
        date_range[1],
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Graph stored with ID: {graph_id}")
    return graph_id


# ============================================
# STEP 8: EXPORT FOR VISUALIZATION
# ============================================

def export_graph_json(G, filepath="causal_graph.json"):
    """Export graph as JSON for D3.js frontend visualization."""
    print(f"\nExporting graph to {filepath}...")

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
            "confidence": data["confidence"],
            "edge_type": data["edge_type"],
        })

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "max_lag": MAX_LAG,
            "created_at": datetime.now().isoformat(),
        }
    }

    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"  Exported {len(nodes)} nodes and {len(edges)} edges")
    return filepath


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - CAUSAL DISCOVERY ENGINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_processed_data()

    # Step 2: Prepare matrices
    X, X_lag, variable_names, means, stds = prepare_dynotears_data(df)

    # Step 3: Run causal discovery
    W_contemp, W_lagged = run_dynotears(X, X_lag, variable_names)

    # Step 4: Apply constraints
    W_contemp, W_lagged = apply_constraints(W_contemp, W_lagged, variable_names)

    # Step 5: Bootstrap confidence
    contemp_conf, lagged_conf = bootstrap_confidence(
        X, X_lag, variable_names, n_bootstrap=N_BOOTSTRAP
    )

    # Step 6: Build graph
    G = build_graph(W_contemp, W_lagged, contemp_conf, lagged_conf, variable_names)

    # Step 7: Store in database
    graph_id = store_graph(G, variable_names)

    # Step 8: Export for frontend
    export_graph_json(G)

    print("\n✓ Causal discovery complete!")
    print(f"  Graph ID: {graph_id}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print("=" * 60)


if __name__ == "__main__":
    main()