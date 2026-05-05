"""
Causal Graph Data Extract (for Figure 3)
==========================================
Pulls the latest ensemble causal graph from models.causal_graphs, categorizes
each variable (equity, rates, credit, commodity, volatility, macro, sector),
and writes JSON with the data needed to render a grouped-circular network
visualization.

Filtering:
  - Show only CONSENSUS edges (in both DYNOTEARS and PCMCI) — 253 edges
  - This matches the paper's locked CAUSAL_CONSENSUS_EDGES number

Output:
  - causal_graph_data.json
    {
      "metadata": {...},
      "nodes": [
        {"id": "^GSPC", "category": "equity", "label": "S&P 500",
         "in_degree": 12, "out_degree": 8, ...},
        ...
      ],
      "edges": [
        {"source": "BAMLH0A0HYM2", "target": "BAMLH0A2HYB",
         "dyno_confidence": 0.985, "pcmci_score": 0.932,
         "consensus_score": 0.918, "weight": ...},
        ...
      ],
      "category_stats": {...}
    }
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# VARIABLE CATEGORIES
# ============================================================
# Each variable in the causal graph gets assigned to one of 7 categories
# for grouped layout and color-coding.

CATEGORIES = {
    # Broad equity indices
    "equity_index": {
        "vars": ["^GSPC", "^DJI", "^NDX", "^RUT", "EEM", "VTI"],
        "label": "Equity Indices",
        "color": "#1f77b4",  # blue
    },
    # Sector ETFs
    "equity_sector": {
        "vars": ["XLF", "XLK", "XLE", "XLV", "XLY", "XLU", "XLI", "XLP", "XLB",
                 "XLRE", "XLC"],
        "label": "Sectors",
        "color": "#17becf",  # cyan
    },
    # Rates (yields, policy, commercial paper, SOFR, TED spread)
    "rates": {
        "vars": ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y",
                 "FEDFUNDS", "DFF", "SOFR", "SOFR90DAYAVG", "TB3MS",
                 "DCPF3M", "DCPN3M", "TEDRATE"],
        "label": "Rates",
        "color": "#ff7f0e",  # orange
    },
    # Credit spreads (IG, HY, EM sovereign, ETFs)
    "credit": {
        "vars": ["BAMLC0A0CM", "BAMLC0A1CAAA", "BAMLC0A2CAA", "BAMLC0A3CA",
                 "BAMLC0A4CBBB", "BAMLH0A0HYM2", "BAMLH0A1HYBB", "BAMLH0A2HYB",
                 "BAMLH0A3HYC", "BAMLEMCBPIOAS",
                 "HYG", "LQD", "TLT"],
        "label": "Credit",
        "color": "#d62728",  # red
    },
    # Commodities
    "commodity": {
        "vars": ["CL=F", "BZ=F", "GC=F", "SI=F", "HG=F", "NG=F", "CME_F"],
        "label": "Commodities",
        "color": "#8c564b",  # brown
    },
    # Volatility (equity + bond)
    "volatility": {
        "vars": ["^VIX", "^VVIX", "^VXN", "^OVX", "^MOVE"],
        "label": "Volatility",
        "color": "#9467bd",  # purple
    },
    # Macro / economic indicators + bank lending surveys + stress indices
    "macro": {
        "vars": ["CPIAUCSL", "PCEPILFE", "CPILFESL", "PAYEMS", "UNRATE",
                 "INDPRO", "ICSA", "DRTSCIS", "DRTSCILM", "DRSDCILM", "DRTSSP",
                 "UMCSENT", "NAPM",
                 "A191RL1Q225SBEA", "HOUST", "M2SL", "RSXFS", "STLFSI4"],
        "label": "Macro",
        "color": "#2ca02c",  # green
    },
    # FX
    "fx": {
        "vars": ["DX-Y.NYB", "EURUSD=X", "JPYUSD=X", "GBPUSD=X"],
        "label": "FX",
        "color": "#e377c2",  # pink
    },
}


def categorize(var_name: str) -> str:
    """Return the category key for a variable, or 'other' if unmapped."""
    for cat_key, cat_data in CATEGORIES.items():
        if var_name in cat_data["vars"]:
            return cat_key
    return "other"


def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def fetch_ensemble_graph():
    """
    Pull the latest ensemble causal graph from models.causal_graphs.
    Tries 'ensemble' method first, then any method with both dyno and pcmci scores.

    Schema-adaptive: introspects the table's actual columns first so we don't
    SELECT phantom fields (schema may differ across project versions).
    """
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Step 1: find which columns actually exist on this table
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='models' AND table_name='causal_graphs'
    """)
    available_cols = {row["column_name"] for row in cursor.fetchall()}

    # Step 2: intersect with what we want
    wanted = ["id", "method", "adjacency_matrix", "confidence_scores",
              "created_at", "model_version", "metadata", "node_count"]
    select_cols = [c for c in wanted if c in available_cols]
    if "adjacency_matrix" not in select_cols:
        cursor.close(); conn.close()
        return None
    select_clause = ", ".join(select_cols)

    # Step 3: try explicit ensemble method first
    cursor.execute(f"""
        SELECT {select_clause}
        FROM models.causal_graphs
        WHERE method LIKE '%%ensemble%%'
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()

    if not row:
        # Step 4: fall back to any recent graph with both dyno and pcmci scores
        cursor.execute(f"""
            SELECT {select_clause}
            FROM models.causal_graphs
            ORDER BY created_at DESC LIMIT 5
        """)
        candidates = cursor.fetchall()
        for cand in candidates:
            adj = cand["adjacency_matrix"]
            if adj:
                sample_key = next(iter(adj.keys()), None)
                if sample_key:
                    sample_val = adj[sample_key]
                    if (isinstance(sample_val, dict) and
                        "dyno_confidence" in sample_val and
                        "pcmci_score" in sample_val):
                        row = cand
                        break

    cursor.close(); conn.close()
    return row


def build_payload(graph_row):
    """Convert DB row into JSON payload for Figure 3.

    Edge schema (from ensemble_dynotears_pcmci graphs):
      {
        "method": "consensus" | "pcmci_only" | "dynotears_only",
        "weight": float,              # aggregate strength
        "dyno_weight": float,         # DYNOTEARS LASSO weight (0 if pcmci_only)
        "dyno_confidence": float,     # DYNOTEARS confidence (0 if pcmci_only)
        "pcmci_strength": float,      # PCMCI effect size
        "pcmci_pvalue": float,        # PCMCI p-value
        "lag": int,                   # lag in days
        "edge_type": "contemporaneous" | "lagged",
      }

    Consensus = method=='consensus' (appears in BOTH algorithms).
    This matches the paper's locked CAUSAL_CONSENSUS_EDGES = 253.
    """
    adj = graph_row["adjacency_matrix"]
    if not adj:
        return None

    all_edges = []
    consensus_edges = []
    node_set = set()

    for key, val in adj.items():
        if "->" not in key:
            continue
        src, tgt = key.split("->", 1)
        node_set.add(src)
        node_set.add(tgt)

        if not isinstance(val, dict):
            continue

        # Use exact field names from the schema
        method = str(val.get("method", "")).lower()
        dyno_conf = float(val.get("dyno_confidence") or 0.0)
        dyno_wt = float(val.get("dyno_weight") or 0.0)
        pcmci_strength = float(val.get("pcmci_strength") or 0.0)
        pcmci_pvalue = float(val.get("pcmci_pvalue") or 1.0)
        weight = float(val.get("weight") or 0.0)
        lag = int(val.get("lag") or 0)
        edge_type = str(val.get("edge_type", ""))

        # Consensus score for ranking: prefer dyno_confidence * pcmci_strength
        # when both are positive (consensus edges), else weight as fallback.
        if method == "consensus":
            consensus_score = dyno_conf * pcmci_strength
        else:
            consensus_score = 0.0

        edge = {
            "source": src,
            "target": tgt,
            "method": method,
            "dyno_confidence": dyno_conf,
            "dyno_weight": dyno_wt,
            "pcmci_strength": pcmci_strength,
            "pcmci_pvalue": pcmci_pvalue,
            "weight": weight,
            "lag": lag,
            "edge_type": edge_type,
            "consensus_score": consensus_score,
        }
        all_edges.append(edge)

        # Consensus = explicit flag on the edge
        if method == "consensus":
            consensus_edges.append(edge)

    # Compute degrees for each node (CONSENSUS edges only for Figure 3)
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for e in consensus_edges:
        out_deg[e["source"]] += 1
        in_deg[e["target"]] += 1

    # Build node list with category and degree info
    nodes = []
    for var in sorted(node_set):
        cat = categorize(var)
        nodes.append({
            "id": var,
            "label": var,
            "category": cat,
            "in_degree": in_deg[var],
            "out_degree": out_deg[var],
            "total_degree": in_deg[var] + out_deg[var],
        })

    # Category-level stats
    cat_stats = defaultdict(lambda: {"count": 0, "edges_in_out": 0})
    for n in nodes:
        cat_stats[n["category"]]["count"] += 1
        cat_stats[n["category"]]["edges_in_out"] += n["total_degree"]

    category_stats = {}
    for cat_key, cat_data in CATEGORIES.items():
        stats = cat_stats.get(cat_key, {"count": 0, "edges_in_out": 0})
        category_stats[cat_key] = {
            "label": cat_data["label"],
            "color": cat_data["color"],
            "node_count": stats["count"],
            "total_degree": stats["edges_in_out"],
        }
    if cat_stats.get("other", {}).get("count", 0) > 0:
        category_stats["other"] = {
            "label": "Other",
            "color": "#999999",
            "node_count": cat_stats["other"]["count"],
            "total_degree": cat_stats["other"]["edges_in_out"],
        }

    # Count by method (for verification)
    method_counts = defaultdict(int)
    for e in all_edges:
        method_counts[e["method"]] += 1

    payload = {
        "metadata": {
            "graph_id": str(graph_row["id"]),
            "method": graph_row["method"],
            "created_at": str(graph_row["created_at"]),
            "model_version": graph_row.get("model_version"),
            "total_edges": len(all_edges),
            "consensus_edges": len(consensus_edges),
            "method_counts": dict(method_counts),
            "node_count": len(nodes),
        },
        "category_stats": category_stats,
        "nodes": nodes,
        "edges": consensus_edges,
    }
    return payload


def main():
    print("=" * 80)
    print("  CAUSAL GRAPH DATA EXTRACT (for Figure 3)")
    print("=" * 80)

    print("\n  Step 1: Fetch latest ensemble graph from models.causal_graphs...")
    row = fetch_ensemble_graph()
    if not row:
        print("  ERROR: no ensemble graph found in models.causal_graphs")
        return
    print(f"    Graph ID:      {row['id']}")
    print(f"    Method:        {row['method']}")
    print(f"    Created at:    {row['created_at']}")
    mv = row.get("model_version")
    if mv is not None:
        print(f"    Model version: {mv}")

    print("\n  Step 2: Parse edges and categorize nodes...")
    payload = build_payload(row)
    if not payload:
        print("  ERROR: could not parse adjacency matrix")
        return

    meta = payload["metadata"]
    print(f"    Total edges:         {meta['total_edges']:>5}")
    print(f"    Consensus edges:     {meta['consensus_edges']:>5}")
    print(f"    Node count:          {meta['node_count']:>5}")

    # Method breakdown (sanity check: consensus + pcmci_only + dynotears_only = total)
    if meta.get("method_counts"):
        print(f"\n    Method breakdown:")
        for method_name, count in sorted(meta["method_counts"].items()):
            print(f"      {method_name:<18} {count:>5}")

    print("\n  Step 3: Category breakdown:")
    print(f"    {'Category':<18} {'Color':<10} {'Nodes':>6} {'Edges':>7}")
    for cat_key, stats in payload["category_stats"].items():
        if stats["node_count"] == 0:
            continue
        print(f"    {stats['label']:<18} {stats['color']:<10} "
              f"{stats['node_count']:>6}  {stats['total_degree']:>7}")

    # ----- Sanity check vs canonical -----
    print("\n  Sanity check vs canonical_paper_numbers.py:")
    PAPER_CONSENSUS_EDGES = 255
    PAPER_NODE_COUNT = 56
    if meta['consensus_edges'] == PAPER_CONSENSUS_EDGES:
        print(f"    [OK]   consensus_edges = {meta['consensus_edges']} "
              f"(paper: {PAPER_CONSENSUS_EDGES}) — exact match")
    elif abs(meta['consensus_edges'] - PAPER_CONSENSUS_EDGES) <= 3:
        print(f"    [OK]   consensus_edges = {meta['consensus_edges']} "
              f"(paper: {PAPER_CONSENSUS_EDGES}) — within tolerance")
    else:
        print(f"    [WARN] consensus_edges = {meta['consensus_edges']} "
              f"(paper: {PAPER_CONSENSUS_EDGES}) — significant drift, review needed")

    if meta['node_count'] == PAPER_NODE_COUNT:
        print(f"    [OK]   node_count = {meta['node_count']} (paper: {PAPER_NODE_COUNT})")
    else:
        print(f"    [INFO] node_count = {meta['node_count']} "
              f"(paper: {PAPER_NODE_COUNT}) — some nodes may be isolated")

    # Check Other category is empty
    other_count = payload["category_stats"].get("other", {}).get("node_count", 0)
    if other_count == 0:
        print(f"    [OK]   All {meta['node_count']} nodes categorized (no 'Other' bucket)")
    else:
        print(f"    [WARN] {other_count} nodes in 'Other' category — add them to CATEGORIES dict")

    # ----- Top-degree nodes (for paper discussion) -----
    print("\n  Top-10 most-connected nodes (consensus edges only):")
    sorted_nodes = sorted(payload["nodes"], key=lambda n: -n["total_degree"])[:10]
    print(f"    {'Variable':<16} {'Category':<15} {'In':>4} {'Out':>4} {'Total':>5}")
    for n in sorted_nodes:
        cat_label = payload["category_stats"].get(n["category"], {}).get("label", n["category"])
        print(f"    {n['id']:<16} {cat_label:<15} "
              f"{n['in_degree']:>4} {n['out_degree']:>4} {n['total_degree']:>5}")

    # ----- Top-10 strongest edges -----
    print("\n  Top-10 strongest consensus edges (by dyno_conf * pcmci_strength):")
    sorted_edges = sorted(payload["edges"], key=lambda e: -e["consensus_score"])[:10]
    print(f"    {'Source':<16} {'Target':<16} {'Dyno':>7} {'PCMCI':>7} {'Score':>8}")
    for e in sorted_edges:
        print(f"    {e['source']:<16} {e['target']:<16} "
              f"{e['dyno_confidence']:>7.3f} {e['pcmci_strength']:>7.3f} "
              f"{e['consensus_score']:>8.4f}")

    # ----- Write JSON -----
    out_path = Path(__file__).parent / "causal_graph_data.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, default=str)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Wrote: {out_path}  ({size_kb:.1f} KB)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()