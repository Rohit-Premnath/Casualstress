"""
Dashboard API Router
Serves real-time data for the main dashboard page.
Reads from the same PostgreSQL database that ml_pipeline writes to.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

from app.config import settings

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


@router.get("/summary")
async def get_dashboard_summary():
    """Main dashboard summary: current regime, key metrics, system health."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Current regime
    cursor.execute("""
        SELECT regime_name, probability, date
        FROM models.regimes
        ORDER BY date DESC
        LIMIT 1
    """)
    current = cursor.fetchone()

    # Regime streak (how many consecutive days of current regime)
    streak = 0
    if current:
        cursor.execute("""
            SELECT COUNT(*) as streak FROM (
                SELECT date, regime_name,
                       ROW_NUMBER() OVER (ORDER BY date DESC) -
                       ROW_NUMBER() OVER (PARTITION BY regime_name ORDER BY date DESC) as grp
                FROM models.regimes
                ORDER BY date DESC
            ) sub
            WHERE regime_name = %s AND grp = 0
        """, (current["regime_name"],))
        row = cursor.fetchone()
        streak = row["streak"] if row else 0

    # Regime probabilities (latest)
    cursor.execute("""
        SELECT regime_name, probability
        FROM models.regimes
        WHERE date = (SELECT MAX(date) FROM models.regimes)
    """)
    probs_rows = cursor.fetchall()
    probabilities = {}
    for r in probs_rows:
        probabilities[r["regime_name"].capitalize()] = round(r["probability"] * 100, 1) if r["probability"] else 0

    # Latest S&P 500 value
    cursor.execute("""
        SELECT raw_value, date FROM processed.time_series_data
        WHERE variable_code = '^GSPC' AND source = 'yahoo'
        ORDER BY date DESC LIMIT 1
    """)
    spx = cursor.fetchone()

    # Data pipeline stats
    cursor.execute("SELECT COUNT(DISTINCT variable_code) as n_vars FROM processed.time_series_data")
    n_vars = cursor.fetchone()["n_vars"]

    cursor.execute("SELECT COUNT(DISTINCT date) as n_days FROM processed.time_series_data")
    n_days = cursor.fetchone()["n_days"]

    cursor.execute("SELECT COUNT(*) as n_edges FROM (SELECT 1 FROM models.causal_graphs ORDER BY created_at DESC LIMIT 1) g")

    # Causal graph edges
    cursor.execute("""
        SELECT adjacency_matrix FROM models.causal_graphs
        WHERE method LIKE '%%ensemble%%' OR method LIKE '%%dynotears%%'
        ORDER BY created_at DESC LIMIT 1
    """)
    graph_row = cursor.fetchone()
    n_edges = len(graph_row["adjacency_matrix"]) if graph_row else 0

    # Scenarios count
    cursor.execute("SELECT COUNT(*) as n FROM models.scenarios")
    n_scenarios = cursor.fetchone()["n"]

    cursor.close()
    conn.close()

    return {
        "currentRegime": {
            "name": current["regime_name"].capitalize() if current else "Unknown",
            "confidence": round(current["probability"] * 100, 1) if current and current["probability"] else 0,
            "streak": streak,
            "date": str(current["date"]) if current else None,
            "probabilities": probabilities,
        },
        "spx": {
            "value": round(spx["raw_value"], 2) if spx else None,
            "date": str(spx["date"]) if spx else None,
        },
        "system": {
            "variables": n_vars,
            "tradingDays": n_days,
            "causalEdges": n_edges,
            "scenarios": n_scenarios,
        },
    }


@router.get("/spx-history")
async def get_spx_history(days: int = 180):
    """S&P 500 price history for the dashboard chart."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT date, raw_value as value
        FROM processed.time_series_data
        WHERE variable_code = '^GSPC' AND source = 'yahoo'
        ORDER BY date DESC
        LIMIT %s
    """, (days,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{
        "date": r["date"].strftime("%b %d") if r["date"] else "",
        "value": round(r["value"], 2) if r["value"] else 0,
    } for r in reversed(rows)]


@router.get("/regime-chart")
async def get_regime_chart(months: int = 27):
    """Regime classification timeline for the dashboard chart."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT date, regime_name
        FROM models.regimes
        ORDER BY date DESC
        LIMIT %s
    """, (months * 22,))  # ~22 trading days per month
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return []

    # Aggregate to monthly
    from collections import defaultdict
    monthly = defaultdict(list)
    for r in rows:
        key = r["date"].strftime("%b %Y")
        monthly[key].append(r["regime_name"])

    regime_map = {"calm": 1, "normal": 2, "elevated": 3, "stressed": 4, "high_stress": 4, "crisis": 5}
    name_map = {"calm": "Calm", "normal": "Normal", "elevated": "Elevated",
                "stressed": "Stressed", "high_stress": "Stressed", "crisis": "Crisis"}

    result = []
    for month_key in sorted(monthly.keys(), key=lambda x: datetime.strptime(x, "%b %Y")):
        regimes = monthly[month_key]
        from collections import Counter
        most_common = Counter(regimes).most_common(1)[0][0]
        regime_name = name_map.get(most_common, "Normal")
        val = regime_map.get(most_common, 2)

        result.append({
            "month": month_key,
            "monthShort": month_key[:3] + " '" + month_key[-2:],
            "regime": regime_name,
            "value": val,
            "Calm": val if regime_name == "Calm" else 0,
            "Normal": val if regime_name == "Normal" else 0,
            "Elevated": val if regime_name == "Elevated" else 0,
            "Stressed": val if regime_name == "Stressed" else 0,
            "Crisis": val if regime_name == "Crisis" else 0,
        })

    return result[-months:]


@router.get("/top-causal-links")
async def get_top_causal_links(limit: int = 10):
    """Top causal links for the dashboard."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT adjacency_matrix FROM models.causal_graphs
        WHERE method LIKE '%%ensemble%%' OR method LIKE '%%dynotears%%'
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return []

    edges = []
    for edge_key, edge_data in row["adjacency_matrix"].items():
        cause, effect = edge_key.split("->")
        weight = abs(edge_data.get("weight", 0))
        confidence = edge_data.get("confidence", 1.0)
        edges.append({
            "cause": cause,
            "effect": effect,
            "weight": round(weight, 2),
            "confidence": round(confidence * 100, 0),
        })

    edges.sort(key=lambda x: x["weight"], reverse=True)
    return edges[:limit]