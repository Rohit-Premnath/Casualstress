"""
load_causal_graph.py
=====================
Loads the pre-computed ensemble_causal_graph.json into models.causal_graphs
so the dashboard top-causal-links panel works.

Run from repo root:
    python load_causal_graph.py
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

def main():
    graph_path = os.path.join(os.path.dirname(__file__), "ensemble_causal_graph.json")
    print(f"Loading {graph_path} ...")

    with open(graph_path) as f:
        graph = json.load(f)

    edges = graph["edges"]
    nodes = [n["id"] if isinstance(n, dict) else n for n in graph["nodes"]]
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")

    # Convert to adjacency_matrix format the dashboard expects:
    # { "cause->effect": { "weight": float, "confidence": float } }
    adjacency_matrix = {}
    for e in edges:
        key = f"{e['source']}->{e['target']}"
        confidence = e.get("dyno_confidence", 0.0)
        if e.get("pcmci_pvalue") is not None and e["pcmci_pvalue"] < 0.05:
            confidence = max(confidence, 1 - e["pcmci_pvalue"])
        adjacency_matrix[key] = {
            "weight":     round(abs(e["weight"]), 6),
            "confidence": round(min(confidence, 1.0), 6),
            "lag":        e.get("lag", 0),
            "method":     e.get("method", "ensemble"),
        }

    conn = get_conn()
    cursor = conn.cursor()

    # Clear existing global graphs (not regime-specific ones)
    cursor.execute("""
        DELETE FROM models.causal_graphs
        WHERE method = 'ensemble' AND regime_id IS NULL
    """)

    graph_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO models.causal_graphs
            (id, created_at, method, variables, adjacency_matrix,
             confidence_scores, regime_id)
        VALUES (%s, %s, %s, %s, %s, %s, NULL)
    """, (
        graph_id,
        datetime.fromisoformat(graph["metadata"].get("created_at", datetime.now().isoformat())),
        "ensemble",
        Json(nodes),
        Json(adjacency_matrix),
        Json({"n_edges": len(edges), "n_nodes": len(nodes)}),
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Inserted graph {graph_id}")
    print(f"  {len(adjacency_matrix)} edges loaded into models.causal_graphs")
    print("Done — refresh the dashboard to see Top 10 Causal Links.")

if __name__ == "__main__":
    main()
