"""
load_regime_graphs.py
======================
Loads regime-specific causal graphs from regime_causal_graphs.json
into models.causal_graphs so the frontend can show different edges
per regime (calm / normal / elevated / stressed / crisis).

Run from repo root:
    python load_regime_graphs.py
"""

import json
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))
import psycopg2
from psycopg2.extras import Json


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def build_adjacency_matrix(edges):
    """Convert edge list to { 'cause->effect': { weight, confidence } } dict."""
    adj = {}
    for e in edges:
        if e["source"] == e["target"]:
            continue  # skip self-loops
        key = f"{e['source']}->{e['target']}"
        adj[key] = {
            "weight":     round(abs(e["weight"]), 6),
            "confidence": round(min(abs(e["weight"]), 1.0), 6),
            "lag":        e.get("lag", 0),
            "method":     e.get("edge_type", "lagged"),
        }
    return adj


def main():
    graph_path = os.path.join(os.path.dirname(__file__), "regime_causal_graphs.json")
    print(f"Loading {graph_path} ...")

    with open(graph_path) as f:
        data = json.load(f)

    created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
    regimes = data["regimes"]

    conn = get_conn()
    cursor = conn.cursor()

    for regime_name, regime_data in regimes.items():
        method = f"regime_{regime_name}"   # e.g. regime_stressed — matches backend query

        # Remove existing entry for this regime
        cursor.execute("DELETE FROM models.causal_graphs WHERE method = %s", (method,))

        nodes = [n["id"] if isinstance(n, dict) else n for n in regime_data.get("nodes", [])]
        edges = regime_data.get("edges", [])
        adj = build_adjacency_matrix(edges)

        graph_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO models.causal_graphs
                (id, created_at, method, variables, adjacency_matrix,
                 confidence_scores, regime_id)
            VALUES (%s, %s, %s, %s, %s, %s, NULL)
        """, (
            graph_id,
            created_at,
            method,
            Json(nodes),
            Json(adj),
            Json({"n_edges": len(adj), "n_nodes": len(nodes), "regime": regime_name}),
        ))

        print(f"  {regime_name:<12} method='{method}'  {len(adj)} edges  id={graph_id}")

    conn.commit()
    cursor.close()
    conn.close()

    print("\nDone. Regime-specific graphs loaded.")
    print("Refresh the causal graph page — each regime tab now shows different edges.")


if __name__ == "__main__":
    main()
