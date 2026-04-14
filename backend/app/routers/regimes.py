"""
Regime Detection API Router
Serves regime timeline, current regime, transition matrix, and characteristics.
"""

from fastapi import APIRouter, Query
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import Counter, defaultdict
from datetime import datetime

from app.config import settings

router = APIRouter(prefix="/api/v1/regimes", tags=["regimes"])


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


NAME_MAP = {
    "calm": "Calm", "normal": "Normal", "elevated": "Elevated",
    "stressed": "Stressed", "high_stress": "High Stress", "crisis": "Crisis",
}
REGIME_LABELS = ["Calm", "Normal", "Elevated", "Stressed", "High Stress", "Crisis"]


@router.get("/current")
async def get_current_regime():
    """Current regime with confidence and streak."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT regime_name, probability, date
        FROM models.regimes ORDER BY date DESC LIMIT 1
    """)
    current = cursor.fetchone()

    if not current:
        cursor.close()
        conn.close()
        return {"name": "Unknown", "confidence": 0, "streak": 0}

    # Streak
    cursor.execute("""
        SELECT date, regime_name FROM models.regimes
        ORDER BY date DESC LIMIT 500
    """)
    rows = cursor.fetchall()
    streak = 0
    current_name = current["regime_name"]
    for r in rows:
        if r["regime_name"] == current_name:
            streak += 1
        else:
            break

    cursor.close()
    conn.close()

    return {
        "name": NAME_MAP.get(current_name, current_name.capitalize()),
        "confidence": round(current["probability"] * 100, 1) if current["probability"] else 0,
        "streak": streak,
        "date": str(current["date"]),
    }


@router.get("/timeline")
async def get_regime_timeline():
    """Regime segments for the timeline chart."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT date, regime_name FROM models.regimes ORDER BY date
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return []

    # Build segments
    segments = []
    current_regime = NAME_MAP.get(rows[0]["regime_name"], rows[0]["regime_name"].capitalize())
    segment_start = rows[0]["date"]

    for i in range(1, len(rows)):
        regime = NAME_MAP.get(rows[i]["regime_name"], rows[i]["regime_name"].capitalize())
        if regime != current_regime:
            months = max(1, (rows[i]["date"] - segment_start).days // 30)
            segments.append({
                "regime": current_regime,
                "start": segment_start.strftime("%Y-%m"),
                "end": rows[i - 1]["date"].strftime("%Y-%m"),
                "months": months,
            })
            current_regime = regime
            segment_start = rows[i]["date"]

    # Final segment
    months = max(1, (rows[-1]["date"] - segment_start).days // 30)
    segments.append({
        "regime": current_regime,
        "start": segment_start.strftime("%Y-%m"),
        "end": rows[-1]["date"].strftime("%Y-%m"),
        "months": months,
    })

    return segments


@router.get("/characteristics")
async def get_regime_characteristics():
    """Statistics for each regime: days, VIX mean, SPX return, spreads."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Get regime labels joined with key variables
    cursor.execute("""
        SELECT r.regime_name,
               COUNT(DISTINCT r.date) as days,
               AVG(vix.raw_value) as vix_mean,
               AVG(spx.transformed_value) as spx_return,
               AVG(hy.raw_value) as hy_spread,
               AVG(yc.raw_value) as yield_curve
        FROM models.regimes r
        LEFT JOIN processed.time_series_data vix
            ON r.date = vix.date AND vix.variable_code = '^VIX' AND vix.source = 'yahoo'
        LEFT JOIN processed.time_series_data spx
            ON r.date = spx.date AND spx.variable_code = '^GSPC' AND spx.source = 'yahoo'
        LEFT JOIN processed.time_series_data hy
            ON r.date = hy.date AND hy.variable_code = 'BAMLH0A0HYM2' AND hy.source = 'fred'
        LEFT JOIN processed.time_series_data yc
            ON r.date = yc.date AND yc.variable_code = 'T10Y2Y' AND yc.source = 'fred'
        GROUP BY r.regime_name
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    total_days = sum(r["days"] for r in rows)
    ordered_rows = sorted(
        rows,
        key=lambda r: REGIME_LABELS.index(NAME_MAP.get(r["regime_name"], r["regime_name"].capitalize()))
        if NAME_MAP.get(r["regime_name"], r["regime_name"].capitalize()) in REGIME_LABELS
        else len(REGIME_LABELS),
    )
    return [{
        "regime": NAME_MAP.get(r["regime_name"], r["regime_name"].capitalize()),
        "days": r["days"],
        "pct": round(r["days"] / total_days * 100, 1) if total_days > 0 else 0,
        "vixMean": round(r["vix_mean"], 2) if r["vix_mean"] else 0,
        "spxReturn": round(r["spx_return"], 4) if r["spx_return"] else 0,
        "hySpread": round(r["hy_spread"], 2) if r["hy_spread"] else 0,
        "yieldCurve": round(r["yield_curve"], 2) if r["yield_curve"] else 0,
    } for r in ordered_rows]


@router.get("/transition-matrix")
async def get_transition_matrix():
    """Regime transition probability matrix."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT date, regime_name FROM models.regimes ORDER BY date
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    labels = REGIME_LABELS
    transitions = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for i in range(1, len(rows)):
        from_r = NAME_MAP.get(rows[i-1]["regime_name"], rows[i-1]["regime_name"].capitalize())
        to_r = NAME_MAP.get(rows[i]["regime_name"], rows[i]["regime_name"].capitalize())
        if from_r in labels and to_r in labels:
            transitions[from_r][to_r] += 1
            totals[from_r] += 1

    matrix = []
    for from_r in labels:
        row = []
        for to_r in labels:
            if totals[from_r] > 0:
                row.append(round(transitions[from_r][to_r] / totals[from_r] * 100, 1))
            else:
                row.append(0)
        matrix.append(row)

    return {"labels": labels, "data": matrix}
