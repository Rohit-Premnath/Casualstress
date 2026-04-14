"""
Stress Test API Router
Runs portfolio stress tests against generated scenarios.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import uuid

from app.config import settings

router = APIRouter(prefix="/api/v1/stress-test", tags=["stress-test"])


EVENT_TYPE_META = {
    "market_crash": {"familyId": "market-crash", "family": "Market Crash"},
    "credit_crisis": {"familyId": "credit-crisis", "family": "Credit Crisis"},
    "rate_shock": {"familyId": "rate-shock", "family": "Rate Shock"},
    "global_shock": {"familyId": "global-shock", "family": "Global Shock"},
    "volatility_shock": {"familyId": "vol-shock", "family": "Volatility Shock"},
    "pandemic_exogenous": {"familyId": "pandemic", "family": "Pandemic / Exogenous"},
}
SEVERITY_DEFAULT_MAGNITUDES = {
    "market_crash": 3.0,
    "credit_crisis": 3.0,
    "rate_shock": 2.0,
    "global_shock": 3.0,
    "volatility_shock": 4.0,
    "pandemic_exogenous": 4.0,
}


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def _scenarios_has_event_type_column() -> bool:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'models'
          AND table_name = 'scenarios'
          AND column_name = 'event_type'
        LIMIT 1
        """
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists


# Asset-to-variable mapping
ASSET_VAR_MAP = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "20Y Treasury Bonds": "TLT",
    "Investment Grade Bonds": "LQD",
    "High Yield Bonds": "HYG",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Emerging Markets": "EEM",
    "Financials": "XLF",
    "Tech": "XLK",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Disc": "XLY",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}


class Holding(BaseModel):
    asset: str
    weight: float
    amount: float
    category: str


class StressTestRequest(BaseModel):
    holdings: List[Holding]
    scenario_id: Optional[str] = None


LOG_RETURN_VARS = {
    "^GSPC", "^NDX", "^RUT", "^VIX", "XLF", "XLK", "XLE", "XLV",
    "XLY", "XLU", "TLT", "LQD", "HYG", "EEM", "CL=F", "GC=F",
}


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    if weights.size == 0:
        return weights
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.ones_like(weights, dtype=float) / len(weights)
    return weights / weights.sum()


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    weights = _normalize_weights(np.asarray(weights, dtype=float))
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    return float(np.interp(q, cumulative, sorted_values))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    weights = _normalize_weights(np.asarray(weights, dtype=float))
    return float(np.average(values, weights=weights))


def _build_distribution(values: np.ndarray, total_notional: float, weights: np.ndarray) -> List[dict]:
    if values.size == 0 or total_notional <= 0:
        return []

    pct_values = (values / total_notional) * 100.0
    lo = float(pct_values.min())
    hi = float(pct_values.max())
    if lo == hi:
        hi = lo + 1.0

    bins = np.linspace(lo, hi, 14)
    hist, edges = np.histogram(pct_values, bins=bins, weights=weights)
    return [
        {
            "bucket": f"{round(float(edges[i]), 1)}%",
            "freq": round(float(hist[i]), 4),
        }
        for i in range(len(hist))
    ]


def _infer_event_type(row: dict) -> str:
    raw_value = row.get("event_type") or row.get("shock_variable") or "market_crash"
    return raw_value if raw_value in EVENT_TYPE_META else "market_crash"


def _infer_severity(row: dict) -> str:
    event_type = _infer_event_type(row)
    default_magnitude = SEVERITY_DEFAULT_MAGNITUDES.get(event_type, 3.0)
    stored_magnitude = abs(float(row.get("shock_magnitude") or default_magnitude))
    ratio = stored_magnitude / default_magnitude if default_magnitude else 1.0
    if ratio < 0.75:
        return "Mild"
    if ratio > 1.3:
        return "Extreme"
    return "Severe"


@router.post("/run")
async def run_stress_test(request: StressTestRequest):
    """Run portfolio stress test against latest or specified scenarios."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    has_event_type = _scenarios_has_event_type_column()

    # Load scenarios
    if request.scenario_id:
        if has_event_type:
            cursor.execute("""
                SELECT id, scenario_paths, event_type, shock_variable, shock_magnitude, created_at, n_scenarios
                FROM models.scenarios WHERE id = %s
            """, (request.scenario_id,))
        else:
            cursor.execute("""
                SELECT id, scenario_paths, shock_variable, shock_magnitude, created_at, n_scenarios
                FROM models.scenarios WHERE id = %s
            """, (request.scenario_id,))
    else:
        if has_event_type:
            cursor.execute("""
                SELECT id, scenario_paths, event_type, shock_variable, shock_magnitude, created_at, n_scenarios
                FROM models.scenarios
                ORDER BY created_at DESC LIMIT 1
            """)
        else:
            cursor.execute("""
                SELECT id, scenario_paths, shock_variable, shock_magnitude, created_at, n_scenarios
                FROM models.scenarios
                ORDER BY created_at DESC LIMIT 1
            """)

    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": "No scenarios found"}

    paths = row["scenario_paths"]
    total_notional = sum(h.amount for h in request.holdings)
    event_type = _infer_event_type(row)
    scenario_meta = EVENT_TYPE_META.get(event_type, {"familyId": "market-crash", "family": "Market Crash"})

    horizon = 0
    if paths:
        first_valid_path = next(
            (
                path for path in paths
                if isinstance(path, dict) and isinstance(path.get("data"), dict) and path["data"]
            ),
            None,
        )
        if first_valid_path:
            first_series = next(iter(first_valid_path["data"].values()), [])
            horizon = len(first_series) if isinstance(first_series, list) else 0

    # Compute portfolio P&L for each scenario
    portfolio_pnls = []
    per_holding_pnls: Dict[str, List[float]] = {holding.asset: [] for holding in request.holdings}
    per_holding_weights: Dict[str, List[float]] = {holding.asset: [] for holding in request.holdings}
    scenario_weights = []

    for path in paths:
        if not isinstance(path, dict) or "data" not in path:
            continue

        scenario_weight = float(path.get("soft_filter_weight", 1.0) or 1.0)
        scenario_pnl = 0
        for holding in request.holdings:
            var_code = ASSET_VAR_MAP.get(holding.asset, holding.asset)
            if var_code not in path["data"]:
                continue

            cum_return = sum(path["data"][var_code])
            if var_code in LOG_RETURN_VARS:
                pct_return = np.exp(cum_return) - 1
            else:
                pct_return = cum_return

            holding_pnl = holding.amount * pct_return
            scenario_pnl += holding_pnl
            per_holding_pnls[holding.asset].append(float(holding_pnl))
            per_holding_weights[holding.asset].append(scenario_weight)

        portfolio_pnls.append(scenario_pnl)
        scenario_weights.append(scenario_weight)

    if not portfolio_pnls:
        cursor.close()
        conn.close()
        return {"error": "Could not compute P&L"}

    pnls = np.array(portfolio_pnls)
    weights = _normalize_weights(np.asarray(scenario_weights, dtype=float))

    # Risk metrics
    var_95 = _weighted_quantile(pnls, 0.05, weights)
    var_99 = _weighted_quantile(pnls, 0.01, weights)
    tail_mask = pnls <= var_95
    cvar_95 = _weighted_mean(pnls[tail_mask], weights[tail_mask]) if np.any(tail_mask) else var_95
    max_drawdown = float(np.min(pnls))

    # Loss probabilities
    loss_any = float(np.average((pnls < 0).astype(float), weights=weights) * 100)
    loss_5 = float(np.average((pnls < -total_notional * 0.05).astype(float), weights=weights) * 100)
    loss_10 = float(np.average((pnls < -total_notional * 0.10).astype(float), weights=weights) * 100)

    # Sector risk decomposition
    sector_risk = {}
    for holding in request.holdings:
        cat = holding.category
        var_code = ASSET_VAR_MAP.get(holding.asset, holding.asset)
        asset_pnls = []
        for path in paths:
            if isinstance(path, dict) and "data" in path and var_code in path["data"]:
                cum = sum(path["data"][var_code])
                if var_code in LOG_RETURN_VARS:
                    asset_pnls.append(holding.amount * (np.exp(cum) - 1))
                else:
                    asset_pnls.append(holding.amount * cum)

        if asset_pnls:
            asset_weights = np.asarray(per_holding_weights.get(holding.asset, []), dtype=float)
            risk_contribution = abs(_weighted_quantile(np.asarray(asset_pnls, dtype=float), 0.05, asset_weights))
        else:
            risk_contribution = 0

        if cat not in sector_risk:
            sector_risk[cat] = 0
        sector_risk[cat] += risk_contribution

    total_risk = sum(sector_risk.values()) or 1
    sector_colors = {
        "equity": "#3b82f6", "fixed-income": "#a855f7",
        "commodities": "#f59e0b", "currency": "#06b6d4",
    }
    sector_data = [{
        "sector": k.replace("-", " ").title(),
        "value": round(v / total_risk * 100),
        "color": sector_colors.get(k, "#6b7280"),
    } for k, v in sector_risk.items()]

    # Per-holding risk
    holding_risk = []
    contributor_data = []
    absorber_data = []
    for holding in request.holdings:
        asset_pnls = per_holding_pnls.get(holding.asset, [])
        if asset_pnls:
            asset_weights = np.asarray(per_holding_weights.get(holding.asset, []), dtype=float)
            asset_array = np.asarray(asset_pnls, dtype=float)
            p5_risk = _weighted_quantile(asset_array, 0.05, asset_weights)
            mean_pnl = _weighted_mean(asset_array, asset_weights)
            holding_risk.append({
                "holding": holding.asset,
                "risk": round(p5_risk),
            })
            contributor_data.append({
                "asset": holding.asset,
                "contribution": round(p5_risk),
                "absRisk": abs(p5_risk),
            })
            if mean_pnl > 0:
                absorber_data.append({
                    "asset": holding.asset,
                    "contribution": round(mean_pnl),
                })

    holding_risk.sort(key=lambda x: x["risk"])
    contributor_total = sum(item["absRisk"] for item in contributor_data if item["contribution"] < 0) or 1.0
    top_contributors = [
        {
            "asset": item["asset"],
            "contribution": item["contribution"],
            "pct": round((item["absRisk"] / contributor_total) * 100, 1),
        }
        for item in sorted(
            (item for item in contributor_data if item["contribution"] < 0),
            key=lambda x: x["contribution"],
        )[:3]
    ]
    absorber_total = sum(item["contribution"] for item in absorber_data) or 1.0
    top_absorbers = [
        {
            "asset": item["asset"],
            "contribution": item["contribution"],
            "pct": round((item["contribution"] / absorber_total) * 100, 1),
        }
        for item in sorted(absorber_data, key=lambda x: x["contribution"], reverse=True)[:3]
    ]
    pnl_distribution = _build_distribution(pnls, total_notional, weights)

    # Store result
    result_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO app.stress_test_results
            (id, portfolio, var_95, var_99, cvar_95, max_drawdown, sector_decomposition, marginal_contributions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        result_id,
        psycopg2.extras.Json([h.dict() for h in request.holdings]),
        var_95, var_99, cvar_95, max_drawdown,
        psycopg2.extras.Json(sector_data),
        psycopg2.extras.Json(holding_risk),
    ))
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "id": result_id,
        "portfolioValue": round(total_notional),
        "scenario": {
            "id": str(row["id"]),
            "familyId": scenario_meta["familyId"],
            "family": scenario_meta["family"],
            "eventType": event_type,
            "severity": _infer_severity(row),
            "model": "Full Model (Soft Filtered)",
            "generatedAt": str(row["created_at"]) if row.get("created_at") else None,
            "pathsUsed": int(row["n_scenarios"] or len(paths) or 0),
            "horizon": horizon,
        },
        "var95": round(var_95),
        "var99": round(var_99),
        "cvar95": round(cvar_95),
        "maxDrawdown": round(max_drawdown),
        "lossProbabilities": {
            "any": round(loss_any),
            "over5": round(loss_5),
            "over10": round(loss_10),
        },
        "sectorRisk": sector_data,
        "holdingRisk": holding_risk[:5],
        "pnlDistribution": pnl_distribution,
        "topContributors": top_contributors,
        "topAbsorbers": top_absorbers,
    }
