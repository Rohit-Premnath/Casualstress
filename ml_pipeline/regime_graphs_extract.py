"""
Regime-Conditional Graph Data Extract (for Figure 4)
=====================================================
Pulls the two regime-specific causal graphs from models.causal_graphs:
  - method='regime_calm'      (or fallback: 'regime_conditional_calm')
  - method='regime_stressed'  (or fallback: 'regime_conditional_stressed')

Computes edge set diff:
  - stress_only_edges: in stressed graph but NOT calm graph (contagion paths)
  - calm_only_edges:   in calm graph but NOT stressed graph (broken relationships)
  - shared_edges:      in both graphs

Output:
  - regime_graphs_data.json
    {
      "metadata": { ... calm_method, stressed_method, counts ... },
      "category_stats": { ... },
      "nodes": [ ... 56 categorized nodes ... ],
      "calm_edges":     [ ... ],
      "stressed_edges": [ ... ],
      "shared_edges":   [ ... ],
      "stress_only_edges": [ ... ],
      "calm_only_edges":   [ ... ]
    }

Paper claims this should produce:
  - ~211 stress_only edges (contagion paths activate during stress)
  - ~97  calm_only edges (normal relationships break)
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Reuse categorization from Figure 3's extract
from causal_graph_extract import CATEGORIES, categorize

warnings.filterwarnings("ignore")
load_dotenv()


def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def fetch_regime_graph(regime_name: str):
    """
    Fetch a regime-conditional graph. Tries 'regime_<name>' first, then
    falls back to 'regime_conditional_<name>' naming.
    Schema-adaptive: SELECTs only columns that exist.
    """
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='models' AND table_name='causal_graphs'
    """)
    available = {row["column_name"] for row in cursor.fetchall()}
    wanted = ["id", "method", "adjacency_matrix", "created_at",
              "confidence_scores", "node_count", "model_version"]
    select_cols = [c for c in wanted if c in available]
    select_clause = ", ".join(select_cols)

    for method_name in [f"regime_{regime_name}", f"regime_conditional_{regime_name}"]:
        cursor.execute(f"""
            SELECT {select_clause}
            FROM models.causal_graphs
            WHERE method = %s
            ORDER BY created_at DESC LIMIT 1
        """, (method_name,))
        row = cursor.fetchone()
        if row:
            cursor.close(); conn.close()
            return row, method_name

    cursor.close(); conn.close()
    return None, None


def parse_edges(adj):
    """
    Parse an adjacency matrix and return list of edge dicts.
    Edge keys are 'src->tgt'. Values may be dicts with various fields,
    or scalars (weight). We normalize to a consistent shape.
    """
    if not adj:
        return [], set()

    edges = []
    nodes = set()
    for key, val in adj.items():
        if "->" not in key:
            continue
        src, tgt = key.split("->", 1)
        nodes.add(src)
        nodes.add(tgt)

        if isinstance(val, dict):
            weight = float(val.get("weight") or 0.0)
            dyno_conf = float(val.get("dyno_confidence") or 0.0)
            pcmci_strength = float(val.get("pcmci_strength") or 0.0)
            method = str(val.get("method", ""))
            edge_type = str(val.get("edge_type", ""))
            lag = int(val.get("lag") or 0)
        elif isinstance(val, (int, float)):
            weight = float(val)
            dyno_conf = 0.0
            pcmci_strength = 0.0
            method = ""
            edge_type = ""
            lag = 0
        else:
            continue

        # Skip near-zero edges
        if abs(weight) < 1e-6 and dyno_conf == 0 and pcmci_strength == 0:
            continue

        edges.append({
            "source": src,
            "target": tgt,
            "key": f"{src}->{tgt}",
            "weight": weight,
            "dyno_confidence": dyno_conf,
            "pcmci_strength": pcmci_strength,
            "method": method,
            "edge_type": edge_type,
            "lag": lag,
        })
    return edges, nodes


def main():
    print("=" * 80)
    print("  REGIME-CONDITIONAL GRAPH DATA EXTRACT (for Figure 4)")
    print("=" * 80)

    # Fetch both regime graphs
    print("\n  Step 1: Fetch regime-conditional graphs from models.causal_graphs")

    calm_row, calm_method_used = fetch_regime_graph("calm")
    if calm_row:
        print(f"    [OK] calm graph: method='{calm_method_used}'")
        print(f"         ID: {calm_row['id']}  created: {calm_row['created_at']}")
    else:
        print(f"    [MISSING] calm graph — tried 'regime_calm' and 'regime_conditional_calm'")
        print(f"    Try running regime_causal_graphs.py first to generate them")
        return

    stress_row, stress_method_used = fetch_regime_graph("stressed")
    if stress_row:
        print(f"    [OK] stressed graph: method='{stress_method_used}'")
        print(f"         ID: {stress_row['id']}  created: {stress_row['created_at']}")
    else:
        print(f"    [MISSING] stressed graph")
        return

    # Parse edges
    print("\n  Step 2: Parse edges and compute diff")
    calm_edges, calm_nodes = parse_edges(calm_row["adjacency_matrix"])
    stress_edges, stress_nodes = parse_edges(stress_row["adjacency_matrix"])

    all_nodes = calm_nodes | stress_nodes

    calm_keys = {e["key"] for e in calm_edges}
    stress_keys = {e["key"] for e in stress_edges}

    shared_keys = calm_keys & stress_keys
    stress_only_keys = stress_keys - calm_keys
    calm_only_keys = calm_keys - stress_keys

    print(f"    Calm graph:        {len(calm_edges):>5} edges, {len(calm_nodes)} nodes")
    print(f"    Stressed graph:    {len(stress_edges):>5} edges, {len(stress_nodes)} nodes")
    print(f"    Shared:            {len(shared_keys):>5} edges (in both)")
    print(f"    Stress-ONLY:       {len(stress_only_keys):>5} edges (contagion paths)")
    print(f"    Calm-ONLY:         {len(calm_only_keys):>5} edges (broken relationships)")

    # Build edge lists for output, tagged with which regime they belong to
    calm_by_key = {e["key"]: e for e in calm_edges}
    stress_by_key = {e["key"]: e for e in stress_edges}

    shared_edges = []
    for key in shared_keys:
        # Merge: prefer stress edge data for weights (more relevant for stress-test)
        e = dict(stress_by_key[key])
        e["calm_weight"] = calm_by_key[key]["weight"]
        e["stressed_weight"] = stress_by_key[key]["weight"]
        # Amplification factor: how much does the edge strengthen under stress?
        if abs(calm_by_key[key]["weight"]) > 1e-6:
            e["amplification"] = stress_by_key[key]["weight"] / calm_by_key[key]["weight"]
        else:
            e["amplification"] = None
        e["regime_tag"] = "shared"
        shared_edges.append(e)

    stress_only_edges = []
    for key in stress_only_keys:
        e = dict(stress_by_key[key])
        e["calm_weight"] = 0.0
        e["stressed_weight"] = e["weight"]
        e["amplification"] = None  # undefined (was zero in calm)
        e["regime_tag"] = "stress_only"
        stress_only_edges.append(e)

    calm_only_edges = []
    for key in calm_only_keys:
        e = dict(calm_by_key[key])
        e["calm_weight"] = e["weight"]
        e["stressed_weight"] = 0.0
        e["amplification"] = None
        e["regime_tag"] = "calm_only"
        calm_only_edges.append(e)

    # Build node list with categorization + per-regime degrees
    print("\n  Step 3: Categorize nodes and compute per-regime degrees")
    calm_deg = defaultdict(int)
    stress_deg = defaultdict(int)
    for e in calm_edges:
        calm_deg[e["source"]] += 1
        calm_deg[e["target"]] += 1
    for e in stress_edges:
        stress_deg[e["source"]] += 1
        stress_deg[e["target"]] += 1

    nodes_out = []
    for var in sorted(all_nodes):
        cat = categorize(var)
        nodes_out.append({
            "id": var,
            "label": var,
            "category": cat,
            "calm_degree": calm_deg[var],
            "stressed_degree": stress_deg[var],
        })

    # Category stats
    cat_stats = {}
    for cat_key, cat_data in CATEGORIES.items():
        members = [n for n in nodes_out if n["category"] == cat_key]
        if not members:
            cat_stats[cat_key] = {
                "label": cat_data["label"], "color": cat_data["color"],
                "node_count": 0, "calm_degree": 0, "stressed_degree": 0,
            }
            continue
        cat_stats[cat_key] = {
            "label": cat_data["label"],
            "color": cat_data["color"],
            "node_count": len(members),
            "calm_degree": sum(n["calm_degree"] for n in members),
            "stressed_degree": sum(n["stressed_degree"] for n in members),
        }
    # Other category (should be empty now)
    other_members = [n for n in nodes_out if n["category"] == "other"]
    if other_members:
        cat_stats["other"] = {
            "label": "Other", "color": "#999999",
            "node_count": len(other_members),
            "calm_degree": sum(n["calm_degree"] for n in other_members),
            "stressed_degree": sum(n["stressed_degree"] for n in other_members),
        }

    # ------- Sanity checks vs paper -------
    print("\n  Step 4: Sanity checks vs paper claims")
    # Paper: "211 stress-only edges, 97 calm-only edges"
    PAPER_STRESS_ONLY = 211
    PAPER_CALM_ONLY = 97
    for label, actual, expected in [
        ("stress_only", len(stress_only_keys), PAPER_STRESS_ONLY),
        ("calm_only",   len(calm_only_keys),   PAPER_CALM_ONLY),
    ]:
        delta = actual - expected
        if delta == 0:
            print(f"    [OK]   {label} = {actual} (paper: {expected}) — exact match")
        elif abs(delta) <= 10:
            print(f"    [OK]   {label} = {actual} (paper: {expected}) — within tolerance ({delta:+})")
        else:
            print(f"    [WARN] {label} = {actual} (paper: {expected}) — significant drift ({delta:+})")

    # Top-10 strongest stress-only edges (the key contagion paths)
    print("\n  Top-10 strongest stress-only contagion edges (by |weight|):")
    top_stress_only = sorted(stress_only_edges, key=lambda e: -abs(e["weight"]))[:10]
    print(f"    {'Source':<16} {'Target':<16} {'Weight':>9}")
    for e in top_stress_only:
        print(f"    {e['source']:<16} {e['target']:<16} {e['weight']:>+9.4f}")

    # Top-10 most-amplified shared edges (credit cascade, inflation spiral)
    amplified = [e for e in shared_edges if e["amplification"] is not None]
    amplified.sort(key=lambda e: -abs(e["amplification"]))
    print("\n  Top-10 most-amplified shared edges (stressed/calm ratio):")
    print(f"    {'Source':<16} {'Target':<16} {'Calm':>8} {'Stress':>8} {'Amp':>7}")
    for e in amplified[:10]:
        print(f"    {e['source']:<16} {e['target']:<16} "
              f"{e['calm_weight']:>+8.4f} {e['stressed_weight']:>+8.4f} "
              f"{e['amplification']:>+7.2f}x")

    payload = {
        "metadata": {
            "calm_graph_id": str(calm_row["id"]),
            "calm_method": calm_method_used,
            "calm_edge_count": len(calm_edges),
            "stress_graph_id": str(stress_row["id"]),
            "stress_method": stress_method_used,
            "stress_edge_count": len(stress_edges),
            "shared_count": len(shared_edges),
            "stress_only_count": len(stress_only_edges),
            "calm_only_count": len(calm_only_edges),
            "node_count": len(all_nodes),
        },
        "category_stats": cat_stats,
        "nodes": nodes_out,
        "shared_edges": shared_edges,
        "stress_only_edges": stress_only_edges,
        "calm_only_edges": calm_only_edges,
    }

    out_path = Path(__file__).parent / "regime_graphs_data.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, default=str)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Wrote: {out_path}  ({size_kb:.1f} KB)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()