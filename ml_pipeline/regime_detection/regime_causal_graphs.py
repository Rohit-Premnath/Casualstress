"""
Regime-Conditional Causal Graphs  (v2 — with per-regime PCMCI ensemble)
=========================================================================
Re-runs causal discovery WITHIN each regime using BOTH the Lasso VAR
approximation AND PCMCI conditional independence testing, then builds a
consensus ensemble per regime.

This is our KEY INNOVATION that nobody else has built.
The January 2025 ATSCM paper explicitly states:
"Time series causal methods do not handle regime changes
in learned causal graphs." — We fill that gap.

v2 adds regime-specific PCMCI to complement the Lasso VAR edges.
Edges confirmed by BOTH methods within the same regime carry the highest
confidence and are stored as `regime_ensemble_{name}` in models.causal_graphs.
The scenario generator preferentially loads regime_ensemble_stressed so that
the causal propagation step uses regime-specific, statistically validated edges.

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

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    PCMCI_AVAILABLE = True
except ImportError:
    PCMCI_AVAILABLE = False
    print("  WARNING: tigramite not available — regime PCMCI step will be skipped.")

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

MAX_LAG = 3          # Shorter lag for regime-specific (less data per regime)
PCMCI_MAX_LAG = 3    # Same for PCMCI in regime mode
EDGE_THRESHOLD = 0.05
ALPHA = 0.05         # PCMCI significance level
MIN_EFFECT_SIZE = 0.03

# Minimum trading days per regime to run Lasso discovery
MIN_REGIME_DAYS = 200

# Minimum trading days per regime to run PCMCI (needs more data than Lasso)
MIN_PCMCI_DAYS = 300

# PCMCI-only edge inclusion thresholds
PCMCI_ONLY_ALPHA = 0.01     # must be very significant
PCMCI_ONLY_EFFECT = 0.10    # must have meaningful partial correlation


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

    ts_df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)

    regime_df = pd.read_sql("""
        SELECT date, regime_label, regime_name
        FROM models.regimes
        ORDER BY date
    """, conn)

    conn.close()

    pivoted = ts_df.pivot_table(
        index="date",
        columns="variable_code",
        values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)

    threshold = len(pivoted) * 0.7
    pivoted = pivoted.dropna(axis=1, thresh=int(threshold))
    pivoted = pivoted.dropna()

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
# STEP 2A: LASSO VAR DISCOVERY PER REGIME
# ============================================

def discover_regime_graph_lasso(data, variable_names, regime_name):
    """
    Run Lasso-based causal discovery on data from a single regime.
    Returns a NetworkX DiGraph with edge weights.
    """
    d = len(variable_names)
    values = data[variable_names].values

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    T = len(standardized) - MAX_LAG
    X = standardized[MAX_LAG:]

    X_lags = []
    for lag in range(1, MAX_LAG + 1):
        X_lag = standardized[MAX_LAG - lag: len(standardized) - lag]
        X_lags.append(X_lag)
    X_lag = np.hstack(X_lags)

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

    W_contemp[np.abs(W_contemp) < EDGE_THRESHOLD] = 0
    for lag_idx in range(MAX_LAG):
        W_lagged[lag_idx][np.abs(W_lagged[lag_idx]) < EDGE_THRESHOLD] = 0

    G = nx.DiGraph()
    for name in variable_names:
        G.add_node(name)

    for i in range(d):
        for j in range(d):
            if W_contemp[i, j] != 0:
                G.add_edge(variable_names[i], variable_names[j],
                           weight=float(abs(W_contemp[i, j])),
                           raw_weight=float(W_contemp[i, j]),
                           lag=0, edge_type="contemporaneous",
                           method="lasso")

    for lag_idx in range(MAX_LAG):
        for i in range(d):
            for j in range(d):
                if W_lagged[lag_idx, i, j] != 0:
                    w = float(abs(W_lagged[lag_idx, i, j]))
                    if G.has_edge(variable_names[i], variable_names[j]):
                        if w > G[variable_names[i]][variable_names[j]]["weight"]:
                            G[variable_names[i]][variable_names[j]].update({
                                "weight": w,
                                "raw_weight": float(W_lagged[lag_idx, i, j]),
                                "lag": lag_idx + 1, "edge_type": "lagged",
                                "method": "lasso"})
                    else:
                        G.add_edge(variable_names[i], variable_names[j],
                                   weight=w,
                                   raw_weight=float(W_lagged[lag_idx, i, j]),
                                   lag=lag_idx + 1, edge_type="lagged",
                                   method="lasso")

    return G


# ============================================
# STEP 2B: PCMCI DISCOVERY PER REGIME
# ============================================

def discover_regime_graph_pcmci(data, variable_names, regime_name):
    """
    Run PCMCI on regime-filtered data.
    Returns a list of edge dicts: {cause, effect, lag, strength, p_value}.
    Returns empty list if tigramite is unavailable or data is insufficient.
    """
    if not PCMCI_AVAILABLE:
        return []

    d = len(variable_names)
    values = data[variable_names].values

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    dataframe = pp.DataFrame(data=standardized, var_names=variable_names)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    try:
        results = pcmci.run_pcmci(
            tau_min=0,
            tau_max=PCMCI_MAX_LAG,
            pc_alpha=ALPHA,
            alpha_level=ALPHA,
        )
    except Exception as e:
        print(f"    PCMCI failed for {regime_name}: {e}")
        return []

    p_matrix  = results["p_matrix"]
    val_matrix = results["val_matrix"]

    edges = []
    for i in range(d):
        for j in range(d):
            for lag in range(PCMCI_MAX_LAG + 1):
                p_val    = p_matrix[i, j, lag]
                strength = val_matrix[i, j, lag]

                if p_val >= ALPHA:
                    continue
                if abs(strength) < MIN_EFFECT_SIZE:
                    continue
                if i == j and lag == 0:
                    continue

                edges.append({
                    "cause":    variable_names[i],
                    "effect":   variable_names[j],
                    "lag":      lag,
                    "strength": float(abs(strength)),
                    "p_value":  float(p_val),
                    "edge_type": "contemporaneous" if lag == 0 else "lagged",
                })

    edges.sort(key=lambda x: x["strength"], reverse=True)
    return edges


# ============================================
# STEP 3: BUILD REGIME-SPECIFIC ENSEMBLE
# ============================================

def build_regime_ensemble(lasso_graph, pcmci_edges, variable_names, regime_name):
    """
    Build an ensemble causal graph for a single regime by combining:
    - Lasso VAR edges (from discover_regime_graph_lasso)
    - PCMCI edges (from discover_regime_graph_pcmci)

    Inclusion rules (same as the global ensemble in pcmci_engine.py):
    1. Edge in BOTH methods → CONSENSUS — highest confidence, always included
    2. PCMCI only with p < 0.01 AND |partial corr| > 0.10 → pcmci_only
    3. Lasso only → excluded (unconfirmed by statistical testing)

    Confidence for consensus edges:
        (lasso_weight / max_lasso_weight  +  (1 - pcmci_pvalue)) / 2
    """
    G = nx.DiGraph()
    for name in variable_names:
        G.add_node(name)

    # Build lookup sets
    lasso_edge_set = set(lasso_graph.edges())
    pcmci_edge_set = set()
    pcmci_lookup   = {}   # (cause, effect) -> edge dict
    for e in pcmci_edges:
        key = (e["cause"], e["effect"])
        pcmci_edge_set.add(key)
        # keep the best (strongest) lag for each pair
        if key not in pcmci_lookup or e["strength"] > pcmci_lookup[key]["strength"]:
            pcmci_lookup[key] = e

    # Normalise lasso weights for confidence calculation
    lasso_weights = [d["weight"] for _, _, d in lasso_graph.edges(data=True)]
    max_lasso_w   = max(lasso_weights) if lasso_weights else 1.0

    # 1. CONSENSUS edges — in both
    consensus_keys = lasso_edge_set & pcmci_edge_set
    for cause, effect in consensus_keys:
        lasso_data  = lasso_graph[cause][effect]
        pcmci_data  = pcmci_lookup[(cause, effect)]
        lasso_norm  = lasso_data["weight"] / max_lasso_w
        confidence  = (lasso_norm + (1.0 - pcmci_data["p_value"])) / 2.0
        G.add_edge(
            cause, effect,
            weight=lasso_data["weight"],
            raw_weight=lasso_data.get("raw_weight", lasso_data["weight"]),
            lag=lasso_data["lag"],
            edge_type=lasso_data["edge_type"],
            method="consensus",
            confidence=round(confidence, 4),
            pcmci_strength=pcmci_data["strength"],
            pcmci_pvalue=pcmci_data["p_value"],
        )

    # 2. Strong PCMCI-only edges
    for (cause, effect), e in pcmci_lookup.items():
        if (cause, effect) in consensus_keys:
            continue
        if e["p_value"] < PCMCI_ONLY_ALPHA and e["strength"] > PCMCI_ONLY_EFFECT:
            if not G.has_edge(cause, effect):
                G.add_edge(
                    cause, effect,
                    weight=e["strength"],
                    raw_weight=e["strength"],
                    lag=e["lag"],
                    edge_type=e["edge_type"],
                    method="pcmci_only",
                    confidence=round(1.0 - e["p_value"], 4),
                    pcmci_strength=e["strength"],
                    pcmci_pvalue=e["p_value"],
                )

    consensus_count  = sum(1 for _, _, d in G.edges(data=True) if d["method"] == "consensus")
    pcmci_only_count = sum(1 for _, _, d in G.edges(data=True) if d["method"] == "pcmci_only")

    print(f"    Ensemble [{regime_name}]: "
          f"{G.number_of_edges()} edges  "
          f"({consensus_count} consensus, {pcmci_only_count} pcmci-only, "
          f"{len(lasso_edge_set) - consensus_count} lasso-only excluded)")

    return G


# ============================================
# STEP 4: RUN ALL REGIMES
# ============================================

def run_all_regimes(merged_data, variable_names):
    """
    Run Lasso VAR + PCMCI (where sufficient data) per regime.
    Returns three dicts: lasso_graphs, pcmci_edges_map, ensemble_graphs.
    """
    print("\nRunning regime-specific causal discovery (Lasso + PCMCI ensemble)...\n")

    lasso_graphs    = {}
    pcmci_edges_map = {}
    ensemble_graphs = {}

    regimes = sorted(merged_data["regime_name"].unique())

    for regime_name in regimes:
        regime_data = merged_data[merged_data["regime_name"] == regime_name]
        n_days = len(regime_data)

        if n_days < MIN_REGIME_DAYS:
            print(f"  {regime_name.upper():15s}: {n_days} days — SKIPPED (need {MIN_REGIME_DAYS}+)")
            continue

        print(f"  {regime_name.upper():15s}: {n_days} days")

        # ── Lasso VAR ──────────────────────────────────────────
        print(f"    [Lasso VAR]  ...", end=" ", flush=True)
        G_lasso = discover_regime_graph_lasso(regime_data, variable_names, regime_name)
        lasso_graphs[regime_name] = G_lasso
        print(f"{G_lasso.number_of_edges()} edges")

        # ── PCMCI ──────────────────────────────────────────────
        pcmci_edges = []
        if n_days >= MIN_PCMCI_DAYS:
            print(f"    [PCMCI]      ...", end=" ", flush=True)
            pcmci_edges = discover_regime_graph_pcmci(
                regime_data, variable_names, regime_name
            )
            print(f"{len(pcmci_edges)} significant edges")
        else:
            print(f"    [PCMCI]      SKIPPED (need {MIN_PCMCI_DAYS}+ days, have {n_days})")

        pcmci_edges_map[regime_name] = pcmci_edges

        # ── Ensemble ───────────────────────────────────────────
        if pcmci_edges:
            print(f"    [Ensemble]   building consensus...", end=" ")
            G_ensemble = build_regime_ensemble(
                G_lasso, pcmci_edges, variable_names, regime_name
            )
        else:
            # Not enough data for PCMCI — fall back to Lasso-only graph
            G_ensemble = G_lasso
            print(f"    [Ensemble]   using Lasso-only (no PCMCI data)")

        ensemble_graphs[regime_name] = G_ensemble

    return lasso_graphs, pcmci_edges_map, ensemble_graphs


# ============================================
# STEP 5: ANALYZE STRUCTURAL DIFFERENCES
# ============================================

def analyze_differences(lasso_graphs, ensemble_graphs, variable_names):
    """Compare causal structures across regimes using the ensemble graphs."""
    print("\n" + "=" * 60)
    print("  REGIME-DEPENDENT STRUCTURAL CHANGES (Ensemble graphs)")
    print("=" * 60)

    regime_names = sorted(ensemble_graphs.keys())
    if len(regime_names) < 2:
        print("  Need at least 2 regimes for comparison")
        return {}

    calm_regime   = next((r for r in regime_names if "calm" in r), regime_names[0])
    stress_regime = next((r for r in reversed(regime_names)
                          if "stressed" in r or "crisis" in r), regime_names[-1])

    G_calm   = ensemble_graphs.get(calm_regime)
    G_stress = ensemble_graphs.get(stress_regime)

    if G_calm is None or G_stress is None:
        print("  Cannot compare: missing calm or stressed graph")
        return {}

    calm_edges   = set(G_calm.edges())
    stress_edges = set(G_stress.edges())
    shared       = calm_edges & stress_edges
    contagion    = stress_edges - calm_edges
    disappearing = calm_edges - stress_edges

    print(f"\n  Comparing {calm_regime.upper()} vs {stress_regime.upper()} (ensemble graphs):\n")
    print(f"  {calm_regime.upper()} edges:             {len(calm_edges)}")
    print(f"  {stress_regime.upper()} edges:           {len(stress_edges)}")
    print(f"  Shared:                        {len(shared)}")
    print(f"  Contagion (stress-only):       {len(contagion)}  ← NEW causal links during stress")
    print(f"  Disappearing (calm-only):      {len(disappearing)}")

    # Consensus edge counts
    stressed_consensus = sum(1 for _, _, d in G_stress.edges(data=True)
                             if d.get("method") == "consensus")
    print(f"\n  Stressed graph quality: {stressed_consensus}/{len(stress_edges)} "
          f"edges confirmed by both Lasso and PCMCI")

    if contagion:
        contagion_details = sorted(
            [{"cause": c, "effect": e, "weight": G_stress[c][e]["weight"],
              "lag": G_stress[c][e]["lag"],
              "method": G_stress[c][e].get("method","?")}
             for c, e in contagion],
            key=lambda x: x["weight"], reverse=True
        )
        print(f"\n  Top 15 CONTAGION edges (appear during {stress_regime}):")
        print(f"  {'Cause':<22} {'Effect':<22} {'Weight':>8} {'Lag':>5} {'Method'}")
        print("  " + "-" * 68)
        for ed in contagion_details[:15]:
            print(f"  {ed['cause']:<22} {ed['effect']:<22} "
                  f"{ed['weight']:>8.4f} {ed['lag']:>5}  {ed['method']}")

    print(f"\n  Edge counts across all regimes (ensemble):")
    print(f"  {'Regime':<15} {'Edges':>7} {'Consensus':>10} {'Avg Weight':>12}")
    print("  " + "-" * 47)
    for regime_name in regime_names:
        G = ensemble_graphs[regime_name]
        edges = G.number_of_edges()
        consensus = sum(1 for _, _, d in G.edges(data=True) if d.get("method") == "consensus")
        avg_w = (np.mean([d["weight"] for _, _, d in G.edges(data=True)])
                 if edges > 0 else 0)
        print(f"  {regime_name:<15} {edges:>7} {consensus:>10} {avg_w:>12.4f}")

    return {
        "calm_regime": calm_regime,
        "stress_regime": stress_regime,
        "shared_edges": len(shared),
        "contagion_edges": len(contagion),
        "disappearing_edges": len(disappearing),
    }


# ============================================
# STEP 6: STORE GRAPHS IN DATABASE
# ============================================

def store_all_regime_graphs(lasso_graphs, ensemble_graphs, variable_names):
    """
    Store both the Lasso-only and ensemble graphs for each regime.
    Lasso  → method = 'regime_{name}'
    Ensemble → method = 'regime_ensemble_{name}'
    """
    print("\nStoring regime graphs in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MIN(date), MAX(date) FROM processed.time_series_data")
    date_range = cursor.fetchone()

    graph_ids = {}

    def _graph_to_adjacency(G):
        adjacency = {}
        for cause, effect, data in G.edges(data=True):
            key = f"{cause}->{effect}"
            adjacency[key] = {
                "weight":     data["weight"],
                "raw_weight": data.get("raw_weight", data["weight"]),
                "lag":        data["lag"],
                "edge_type":  data["edge_type"],
                "method":     data.get("method", "lasso"),
                "confidence": data.get("confidence", None),
            }
        return adjacency

    # Store Lasso-only graphs (backwards compatible)
    for regime_name, G in lasso_graphs.items():
        graph_id = str(uuid.uuid4())
        adjacency = _graph_to_adjacency(G)
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
            Json({"regime": regime_name, "min_days": MIN_REGIME_DAYS,
                  "max_lag": MAX_LAG, "method": "lasso_var"}),
            date_range[0], date_range[1],
        ))
        graph_ids[f"lasso_{regime_name}"] = graph_id

    # Store ensemble graphs
    for regime_name, G in ensemble_graphs.items():
        graph_id = str(uuid.uuid4())
        adjacency = _graph_to_adjacency(G)

        # Build confidence scores dict for ensemble
        confidence = {}
        for cause, effect, data in G.edges(data=True):
            key = f"{cause}->{effect}"
            conf = data.get("confidence")
            if conf is None:
                conf = data["weight"] / max(
                    (d["weight"] for _, _, d in G.edges(data=True)), default=1.0
                )
            confidence[key] = round(float(conf), 4)

        cursor.execute("""
            INSERT INTO models.causal_graphs
                (id, method, variables, adjacency_matrix, confidence_scores,
                 structural_constraints, date_range_start, date_range_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            graph_id,
            f"regime_ensemble_{regime_name}",
            Json(variable_names),
            Json(adjacency),
            Json(confidence),
            Json({"regime": regime_name, "min_days": MIN_REGIME_DAYS,
                  "lasso_max_lag": MAX_LAG, "pcmci_max_lag": PCMCI_MAX_LAG,
                  "pcmci_alpha": ALPHA, "method": "lasso_pcmci_ensemble"}),
            date_range[0], date_range[1],
        ))
        graph_ids[f"ensemble_{regime_name}"] = graph_id
        print(f"  {regime_name}: lasso={lasso_graphs[regime_name].number_of_edges()} edges  "
              f"ensemble={G.number_of_edges()} edges  (stored {graph_id[:8]}…)")

    conn.commit()
    cursor.close()
    conn.close()

    return graph_ids


# ============================================
# STEP 7: EXPORT FOR FRONTEND + CANONICAL FILE
# ============================================

def export_regime_graphs(lasso_graphs, ensemble_graphs, analysis):
    """
    Export all regime graphs as a single JSON for the frontend and canonical model.
    The 'edges' list per regime contains the ENSEMBLE edges (Lasso + PCMCI consensus)
    when available, falling back to Lasso-only for small regimes.

    This file is what load_canonical_graph() reads for the scenario generator.
    """
    filepath = "regime_causal_graphs.json"
    print(f"\nExporting regime graphs to {filepath}...")

    export_data = {
        "regimes": {},
        "analysis": analysis,
        "created_at": datetime.now().isoformat(),
        "method": "lasso_pcmci_ensemble_per_regime",
    }

    for regime_name, G in ensemble_graphs.items():
        # Use ensemble graph as the primary source
        G_lasso = lasso_graphs.get(regime_name, G)

        nodes = [{"id": n,
                  "in_degree":  G.in_degree(n),
                  "out_degree": G.out_degree(n)}
                 for n in G.nodes()]

        # Edges in the format expected by build_edge_map in canonical_best_model.py
        edges = []
        for u, v, d in G.edges(data=True):
            edges.append({
                "source":     u,
                "target":     v,
                "weight":     d["weight"],
                "lag":        d["lag"],
                "edge_type":  d["edge_type"],
                "method":     d.get("method", "lasso"),
                "confidence": d.get("confidence", None),
            })

        # Also include lasso-only edges for reference (marked separately)
        lasso_edges = []
        for u, v, d in G_lasso.edges(data=True):
            lasso_edges.append({
                "source": u, "target": v,
                "weight": d["weight"], "lag": d["lag"],
            })

        export_data["regimes"][regime_name] = {
            "nodes":          nodes,
            "edges":          edges,        # ← ensemble edges (used by canonical model)
            "lasso_edges":    lasso_edges,  # ← for reference/comparison
            "n_edges":        len(edges),
            "n_lasso_edges":  len(lasso_edges),
            "n_consensus":    sum(1 for e in edges if e.get("method") == "consensus"),
        }

    with open(filepath, "w") as f:
        json.dump(export_data, f)

    print(f"  Exported {len(ensemble_graphs)} regime ensemble graphs to {filepath}")

    # Summary
    for regime_name, info in export_data["regimes"].items():
        print(f"    {regime_name}: {info['n_edges']} ensemble edges "
              f"({info['n_consensus']} consensus)")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - REGIME-CONDITIONAL ENSEMBLE CAUSAL GRAPHS")
    print("  (Lasso VAR + PCMCI per regime → consensus ensemble)")
    print("=" * 60)

    if not PCMCI_AVAILABLE:
        print("\nWARNING: tigramite not installed. PCMCI step will be skipped.")
        print("Install with: pip install tigramite")
        print("Falling back to Lasso-only regime graphs.\n")

    # Step 1: Load data with regime labels
    merged_data, variable_names = load_data_with_regimes()

    # Step 2: Run Lasso + PCMCI per regime, build ensembles
    lasso_graphs, pcmci_edges_map, ensemble_graphs = run_all_regimes(
        merged_data, variable_names
    )

    # Step 3: Analyze structural differences
    analysis = analyze_differences(lasso_graphs, ensemble_graphs, variable_names)

    # Step 4: Store in database (both lasso and ensemble)
    graph_ids = store_all_regime_graphs(lasso_graphs, ensemble_graphs, variable_names)

    # Step 5: Export for frontend and canonical model
    export_regime_graphs(lasso_graphs, ensemble_graphs, analysis)

    print("\n✓ Regime-conditional ensemble graphs complete!")
    print(f"  Regimes with Lasso graphs:    {len(lasso_graphs)}")
    print(f"  Regimes with ensemble graphs: {len(ensemble_graphs)}")
    for regime_name in ensemble_graphs:
        has_pcmci = len(pcmci_edges_map.get(regime_name, [])) > 0
        G_e = ensemble_graphs[regime_name]
        consensus = sum(1 for _, _, d in G_e.edges(data=True) if d.get("method") == "consensus")
        print(f"    {regime_name}: {G_e.number_of_edges()} ensemble edges  "
              f"({consensus} consensus {'✓ PCMCI' if has_pcmci else '⚠ Lasso-only fallback'})")

    stressed_id = graph_ids.get("ensemble_stressed")
    if stressed_id:
        print(f"\n  ★ regime_ensemble_stressed ID: {stressed_id}")
        print("    The scenario generator will prefer this graph for crisis simulation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
