"""
Causal Graph API Router
Serves the causal graph data for visualization.
"""

from fastapi import APIRouter, Query
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from app.config import settings

router = APIRouter(prefix="/api/v1/causal", tags=["causal"])


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def _load_preferred_global_graph(cursor):
    cursor.execute(
        """
        SELECT adjacency_matrix, variables, method FROM models.causal_graphs
        ORDER BY
            CASE
                WHEN method = 'ensemble_dynotears_pcmci' THEN 0
                WHEN method LIKE '%%ensemble%%' THEN 1
                WHEN method LIKE '%%dynotears%%' THEN 2
                ELSE 3
            END,
            created_at DESC
        LIMIT 1
        """
    )
    return cursor.fetchone()


def _load_preferred_regime_graph(cursor, regime_lower: str):
    cursor.execute(
        """
        SELECT adjacency_matrix, variables, method
        FROM models.causal_graphs
        WHERE method LIKE %s
        ORDER BY
            CASE
                WHEN lower(method) = %s THEN 0
                WHEN lower(method) LIKE %s THEN 1
                ELSE 2
            END,
            created_at DESC
        LIMIT 1
        """,
        (f"%regime%{regime_lower}%", f"regime_{regime_lower}", f"%regime%{regime_lower}%"),
    )
    return cursor.fetchone()


# Variable metadata for display
VARIABLE_META = {
    "^GSPC": {"label": "S&P 500", "category": "equity"},
    "^NDX": {"label": "Nasdaq 100", "category": "equity"},
    "^RUT": {"label": "Russell 2000", "category": "equity"},
    "XLK": {"label": "Tech ETF", "category": "equity"},
    "XLF": {"label": "Financial ETF", "category": "equity"},
    "XLE": {"label": "Energy ETF", "category": "equity"},
    "XLV": {"label": "Health ETF", "category": "equity"},
    "XLY": {"label": "Consumer Disc", "category": "equity"},
    "XLRE": {"label": "Real Estate ETF", "category": "equity"},
    "XLU": {"label": "Utilities ETF", "category": "equity"},
    "EEM": {"label": "Emerging Mkts", "category": "equity"},
    "A191RL1Q225SBEA": {"label": "Real GDP", "category": "macro"},
    "CPIAUCSL": {"label": "CPI", "category": "macro"},
    "PCEPILFE": {"label": "Core PCE", "category": "macro"},
    "UNRATE": {"label": "Unemployment", "category": "macro"},
    "PAYEMS": {"label": "Nonfarm Payrolls", "category": "macro"},
    "INDPRO": {"label": "Industrial Prod", "category": "macro"},
    "ICSA": {"label": "Initial Claims", "category": "macro"},
    "M2SL": {"label": "M2 Money Supply", "category": "macro"},
    "HOUST": {"label": "Housing Starts", "category": "macro"},
    "UMCSENT": {"label": "Consumer Sent", "category": "macro"},
    "RSXFS": {"label": "Retail Sales", "category": "macro"},
    "DGS10": {"label": "10Y Treasury", "category": "rates"},
    "DGS2": {"label": "2Y Treasury", "category": "rates"},
    "FEDFUNDS": {"label": "Fed Funds Rate", "category": "rates"},
    "T10Y2Y": {"label": "10Y-2Y Spread", "category": "rates"},
    "SOFR": {"label": "SOFR", "category": "rates"},
    "SOFR90DAYAVG": {"label": "SOFR 90D Avg", "category": "rates"},
    "TEDRATE": {"label": "TED Spread", "category": "rates"},
    "DCPF3M": {"label": "Fin CP 3M", "category": "rates"},
    "DCPN3M": {"label": "Non-Fin CP 3M", "category": "rates"},
    "^VIX": {"label": "VIX", "category": "volatility"},
    "^VVIX": {"label": "VVIX", "category": "volatility"},
    "^MOVE": {"label": "MOVE Index", "category": "volatility"},
    "STLFSI4": {"label": "St. Louis FSI", "category": "volatility"},
    "CL=F": {"label": "Crude Oil", "category": "commodities"},
    "GC=F": {"label": "Gold", "category": "commodities"},
    "TLT": {"label": "20Y Treasury Bond", "category": "fixed-income"},
    "LQD": {"label": "IG Corp Bond", "category": "fixed-income"},
    "HYG": {"label": "High Yield Bond", "category": "fixed-income"},
    "BAMLH0A0HYM2": {"label": "HY OAS Spread", "category": "fixed-income"},
    "BAMLH0A1HYBB": {"label": "BB Spread", "category": "fixed-income"},
    "BAMLH0A2HYB": {"label": "B Spread", "category": "fixed-income"},
    "BAMLH0A3HYC": {"label": "CCC Spread", "category": "fixed-income"},
    "BAMLC0A0CM": {"label": "IG Spread", "category": "fixed-income"},
    "BAMLC0A4CBBB": {"label": "BBB Spread", "category": "fixed-income"},
    "BAMLC0A3CA": {"label": "A Spread", "category": "fixed-income"},
    "BAMLC0A2CAA": {"label": "AA Spread", "category": "fixed-income"},
    "BAMLC0A1CAAA": {"label": "AAA Spread", "category": "fixed-income"},
    "BAMLEMCBPIOAS": {"label": "EM Corp Spread", "category": "fixed-income"},
    "DX-Y.NYB": {"label": "US Dollar Index", "category": "currency"},
    "EURUSD=X": {"label": "EUR/USD", "category": "currency"},
    "DRTSCIS": {"label": "Small Biz Lending", "category": "macro"},
    "DRTSCILM": {"label": "Large Biz Lending", "category": "macro"},
    "DRTSSP": {"label": "Subprime Auto", "category": "macro"},
    "DRSDCILM": {"label": "Demand Change", "category": "macro"},
}

MAJOR_NODES = {"^GSPC", "^VIX", "DGS10", "BAMLH0A0HYM2", "CL=F", "FEDFUNDS", "UNRATE"}


@router.get("/graph")
async def get_causal_graph(
    regime: Optional[str] = Query(None, description="Filter edges by regime: Calm, Normal, Elevated, Stressed, High Stress, Crisis, or ALL"),
    min_weight: float = Query(0.0, description="Minimum absolute edge weight to include"),
    limit: int = Query(200, description="Maximum number of edges to return"),
):
    """Full causal graph with nodes and edges."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Try regime-specific graph first
    graph_method = None
    if regime and regime.lower() not in ("all", "none"):
        regime_lower = regime.lower().replace(" ", "_")
        row = _load_preferred_regime_graph(cursor, regime_lower)
        if row:
            graph_method = row["method"]
    
    if not graph_method:
        row = _load_preferred_global_graph(cursor)
        graph_method = row["method"] if row else "ensemble"

    cursor.close()
    conn.close()

    if not row:
        return {"nodes": [], "edges": [], "stats": {}}

    adj = row["adjacency_matrix"]

    # Build nodes from edges
    node_ids = set()
    edges = []
    for edge_key, edge_data in adj.items():
        cause, effect = edge_key.split("->")
        weight = abs(edge_data.get("weight", 0))
        if weight < min_weight:
            continue
        node_ids.add(cause)
        node_ids.add(effect)
        edges.append({
            "source": cause,
            "target": effect,
            "weight": round(weight, 3),
            "regime": regime or "ALL",
        })

    edges.sort(key=lambda x: x["weight"], reverse=True)
    edges = edges[:limit]

    # Rebuild node set from filtered edges
    node_ids = set()
    for e in edges:
        node_ids.add(e["source"])
        node_ids.add(e["target"])

    nodes = []
    for nid in node_ids:
        meta = VARIABLE_META.get(nid, {"label": nid, "category": "macro"})
        nodes.append({
            "id": nid,
            "label": meta["label"],
            "category": meta["category"],
            "major": nid in MAJOR_NODES,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "totalNodes": len(nodes),
            "totalEdges": len(edges),
            "method": graph_method,
        },
    }


@router.get("/regime-comparison")
async def get_regime_comparison():
    """Compare causal graph structure across regimes."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    regimes = ["calm", "normal", "elevated", "stressed", "high_stress", "crisis"]
    comparison = {}

    for regime in regimes:
        row = _load_preferred_regime_graph(cursor, regime)
        if row:
            n_edges = len(row["adjacency_matrix"])
            comparison[regime] = {"edges": n_edges}

    cursor.close()
    conn.close()

    return comparison
