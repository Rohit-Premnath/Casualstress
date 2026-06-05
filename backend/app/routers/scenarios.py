"""
Scenario Lab API Router
Serves canonical scenario-generation results and supports on-demand generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from psycopg.rows import dict_row

from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml_pipeline.canonical_best_model import (  # noqa: E402
    CANONICAL_PAPER_NAME,
    get_canonical_candidate_count,
    get_canonical_signature,
    get_canonical_target_scenarios,
    weighted_quantile,
)
from ml_pipeline.generative_engine.scenario_generator import (  # noqa: E402
    DEFAULT_TRAINING_REGIMES,
    LOG_RETURN_VARS,
    apply_canonical_soft_filter,
    fit_regime_var,
    generate_scenarios,
    get_shock_template,
    load_processed_data_with_regimes,
    load_regime_causal_graph,
    score_plausibility,
    store_scenarios,
)

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])


def get_conn():
    return psycopg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def _scenarios_has_event_type_column() -> bool:
    return _scenarios_has_column("event_type")


def _scenarios_has_anchor_variable_column() -> bool:
    return _scenarios_has_column("anchor_variable")


def _scenarios_has_column(column_name: str) -> bool:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'models'
          AND table_name = 'scenarios'
          AND column_name = %s
        LIMIT 1
        """,
        (column_name,),
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists


SCENARIO_FAMILY_META = {
    "market-crash": {
        "label": "Market Crash",
        "event_type": "market_crash",
        "default_anchor": "^GSPC",
        "default_magnitude": -3.0,
    },
    "credit-crisis": {
        "label": "Credit Crisis",
        "event_type": "credit_crisis",
        "default_anchor": "BAMLH0A0HYM2",
        "default_magnitude": 3.0,
    },
    "rate-shock": {
        "label": "Rate Shock",
        "event_type": "rate_shock",
        "default_anchor": "DGS10",
        "default_magnitude": 2.0,
    },
    "global-shock": {
        "label": "Global Shock",
        "event_type": "global_shock",
        "default_anchor": "CL=F",
        "default_magnitude": 3.0,
    },
    "vol-shock": {
        "label": "Volatility Shock",
        "event_type": "volatility_shock",
        "default_anchor": "^VIX",
        "default_magnitude": 4.0,
    },
    "pandemic": {
        "label": "Pandemic / Exogenous",
        "event_type": "pandemic_exogenous",
        "default_anchor": "^GSPC",
        "default_magnitude": -4.0,
    },
}

EVENT_TYPE_TO_FAMILY_ID = {
    meta["event_type"]: family_id for family_id, meta in SCENARIO_FAMILY_META.items()
}
EVENT_TYPE_TO_DEFAULT_ANCHOR = {
    meta["event_type"]: meta["default_anchor"] for meta in SCENARIO_FAMILY_META.values()
}
LEGACY_SHOCK_VALUE_TO_EVENT_TYPE = {
    "^GSPC": "market_crash",
    "BAMLH0A0HYM2": "credit_crisis",
    "DGS10": "rate_shock",
    "CL=F": "global_shock",
    "^VIX": "volatility_shock",
}
SEVERITY_MULTIPLIER = {"Mild": 0.5, "Severe": 1.0, "Extreme": 1.6}

FOCUS_VARIABLES = [
    {"id": "spx", "label": "S&P 500", "ticker": "^GSPC"},
    {"id": "vix", "label": "VIX", "ticker": "^VIX"},
    {"id": "yield", "label": "10Y Yield", "ticker": "DGS10"},
    {"id": "oil", "label": "Crude Oil", "ticker": "CL=F"},
    {"id": "xlf", "label": "Financials", "ticker": "XLF"},
    {"id": "hy", "label": "HY Spread", "ticker": "BAMLH0A0HYM2"},
]
FOCUS_VAR_META = {item["ticker"]: item for item in FOCUS_VARIABLES}
LEVEL_VALUE_VARS = {"DGS10", "BAMLH0A0HYM2"}


class GenerateScenarioRequest(BaseModel):
    family_id: str
    severity: str = "Severe"
    horizon: int = 60
    displayed_paths: int = 200
    anchor_variable_override: Optional[str] = None
    anchor_magnitude_override: Optional[float] = None
    random_seed: Optional[int] = None


@router.get("/metadata")
async def get_scenario_metadata():
    return {
        "families": [
            {
                "id": family_id,
                "label": meta["label"],
                "eventType": meta["event_type"],
                "defaultAnchor": meta["default_anchor"],
                "defaultMagnitude": meta["default_magnitude"],
            }
            for family_id, meta in SCENARIO_FAMILY_META.items()
        ],
        "severityLevels": list(SEVERITY_MULTIPLIER.keys()),
        "severityMultipliers": SEVERITY_MULTIPLIER,
        "horizonOptions": [10, 30, 60],
        "displayedPaths": get_canonical_target_scenarios(),
        "candidateCount": get_canonical_candidate_count(get_canonical_target_scenarios()),
        "focusVariables": FOCUS_VARIABLES,
    }


def _infer_event_type(row: dict) -> str:
    raw_event_type = row.get("event_type")
    if raw_event_type in EVENT_TYPE_TO_FAMILY_ID:
        return raw_event_type

    raw_anchor = row.get("anchor_variable") or row.get("shock_variable")
    if raw_anchor in LEGACY_SHOCK_VALUE_TO_EVENT_TYPE:
        return LEGACY_SHOCK_VALUE_TO_EVENT_TYPE[raw_anchor]

    family_meta = SCENARIO_FAMILY_META.get(raw_anchor)
    if family_meta:
        return family_meta["event_type"]
    return "market_crash"


def _infer_anchor_variable(row: dict) -> str:
    anchor_variable = row.get("anchor_variable")
    if anchor_variable:
        return anchor_variable

    shock_variable = row.get("shock_variable")
    if shock_variable and shock_variable not in EVENT_TYPE_TO_FAMILY_ID:
        return shock_variable

    return EVENT_TYPE_TO_DEFAULT_ANCHOR.get(_infer_event_type(row), "^GSPC")


def _infer_severity_from_row(row: dict) -> str:
    event_type = _infer_event_type(row)
    family_id = EVENT_TYPE_TO_FAMILY_ID.get(event_type, "market-crash")
    default_magnitude = abs(float(SCENARIO_FAMILY_META[family_id]["default_magnitude"]))
    stored_magnitude = abs(float(row.get("shock_magnitude") or default_magnitude))
    if default_magnitude <= 0:
        return "Severe"
    ratio = stored_magnitude / default_magnitude
    if ratio < 0.75:
        return "Mild"
    if ratio > 1.3:
        return "Extreme"
    return "Severe"


def _normalize_weights(paths: List[dict]) -> np.ndarray:
    weights = np.array(
        [
            float(path.get("soft_filter_weight", 1.0) or 1.0)
            if isinstance(path, dict) else 1.0
            for path in paths
        ],
        dtype=float,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.ones(len(paths), dtype=float) / max(len(paths), 1)
    return weights / weights.sum()


def _current_values_for_variables(tickers: List[str]) -> Dict[str, dict]:
    if not tickers:
        return {}

    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    current_values: Dict[str, dict] = {}

    for ticker in tickers:
        cursor.execute(
            """
            SELECT date, COALESCE(raw_value, transformed_value) AS value
            FROM processed.time_series_data
            WHERE variable_code = %s
            ORDER BY date DESC
            LIMIT 1
            """,
            (ticker,),
        )
        row = cursor.fetchone()
        if row and row["value"] is not None:
            current_values[ticker] = {
                "value": round(float(row["value"]), 4),
                "date": str(row["date"]),
                "type": "level" if ticker in LEVEL_VALUE_VARS else "return",
            }

    cursor.close()
    conn.close()
    return current_values


def _convert_cumulative_moves(cumulative: np.ndarray, ticker: str) -> np.ndarray:
    if ticker in LOG_RETURN_VARS:
        return (np.exp(cumulative) - 1.0) * 100.0
    return cumulative


def _round_number(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _build_distribution(values: np.ndarray, weights: np.ndarray) -> List[dict]:
    if values.size == 0:
        return []

    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 16)
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    return [
        {
            "bucket": round(float(edges[i]), 2),
            "freq": round(float(hist[i]), 4),
        }
        for i in range(len(hist))
    ]


def _implied_level(current: Optional[float], move_value: float, ticker: str) -> Optional[float]:
    if current is None:
        return None
    if ticker in LOG_RETURN_VARS:
        return current * (1.0 + move_value / 100.0)
    return current + move_value


def _build_scenario_response(
    scenario_id: str,
    family_id: str,
    severity: str,
    horizon: int,
    paths: List[dict],
    scores: List[float],
    displayed_paths: int,
    event_type: Optional[str] = None,
    anchor_variable: Optional[str] = None,
    shock_magnitude: Optional[float] = None,
    created_at: Optional[str] = None,
) -> dict:
    family_meta = SCENARIO_FAMILY_META[family_id]
    actual_event_type = event_type or family_meta["event_type"]
    actual_anchor = anchor_variable or family_meta["default_anchor"]
    actual_shock = (
        float(shock_magnitude)
        if shock_magnitude is not None
        else family_meta["default_magnitude"] * SEVERITY_MULTIPLIER[severity]
    )
    weights = _normalize_weights(paths)
    current_values = _current_values_for_variables([item["ticker"] for item in FOCUS_VARIABLES])
    score_arr = np.asarray(scores, dtype=float) if scores else np.asarray([], dtype=float)
    weighted_mean_score = float(np.average(score_arr, weights=weights)) if score_arr.size else 0.0
    raw_mean_score = float(score_arr.mean()) if score_arr.size else 0.0

    variables_payload: Dict[str, dict] = {}
    stress_rows: List[dict] = []

    for item in FOCUS_VARIABLES:
        ticker = item["ticker"]
        cumulative_paths = []
        cumulative_by_day = []

        for path in paths:
            if not isinstance(path, dict):
                continue
            path_data = path.get("data", {})
            series = path_data.get(ticker)
            if not series:
                continue
            arr = np.asarray(series[:horizon], dtype=float)
            cumulative = np.cumsum(arr)
            cumulative_paths.append(cumulative[-1])
            cumulative_by_day.append(cumulative)

        if not cumulative_paths:
            continue

        cumulative_paths_arr = np.asarray(cumulative_paths, dtype=float)
        move_values = _convert_cumulative_moves(cumulative_paths_arr, ticker)

        p5 = weighted_quantile(move_values, 0.05, weights)
        median = weighted_quantile(move_values, 0.50, weights)
        p95 = weighted_quantile(move_values, 0.95, weights)
        mean = float(np.average(move_values, weights=weights))

        cumulative_matrix = np.vstack(cumulative_by_day)
        fan_points = []
        for day_idx in range(cumulative_matrix.shape[1]):
            day_moves = _convert_cumulative_moves(cumulative_matrix[:, day_idx], ticker)
            fan_points.append(
                {
                    "day": day_idx,
                    "median": _round_number(weighted_quantile(day_moves, 0.50, weights)),
                    "p5": _round_number(weighted_quantile(day_moves, 0.05, weights)),
                    "p95": _round_number(weighted_quantile(day_moves, 0.95, weights)),
                }
            )

        current_info = current_values.get(ticker, {})
        current_value = current_info.get("value")
        implied_median = _implied_level(current_value, median, ticker)
        implied_low = _implied_level(current_value, p5, ticker)
        implied_high = _implied_level(current_value, p95, ticker)

        variables_payload[ticker] = {
            "label": item["label"],
            "ticker": ticker,
            "valueType": current_info.get("type", "return"),
            "distribution": _build_distribution(move_values, weights),
            "fanChart": fan_points,
        }
        stress_rows.append(
            {
                "variable": item["label"],
                "ticker": ticker,
                "current": _round_number(current_value, 4),
                "currentDate": current_info.get("date"),
                "valueType": current_info.get("type", "return"),
                "p5Move": _round_number(p5),
                "medianMove": _round_number(median),
                "p95Move": _round_number(p95),
                "meanMove": _round_number(mean),
                "impliedMedian": _round_number(implied_median, 4),
                "impliedLow": _round_number(implied_low, 4),
                "impliedHigh": _round_number(implied_high, 4),
            }
        )

    template = get_shock_template(
        actual_event_type,
        actual_anchor,
        actual_shock,
        list(FOCUS_VAR_META.keys()) + [item["ticker"] for item in FOCUS_VARIABLES],
    )

    shock_template = [
        {
            "label": FOCUS_VAR_META.get(ticker, {"label": ticker})["label"],
            "ticker": ticker,
            "shock": round(float(template[ticker]), 2),
        }
        for ticker in template
        if ticker in FOCUS_VAR_META
    ]

    return {
        "id": scenario_id,
        "model": CANONICAL_PAPER_NAME,
        "modelSignature": get_canonical_signature(),
        "family": {
            "id": family_id,
            "label": family_meta["label"],
            "eventType": actual_event_type,
        },
        "severity": severity,
        "graph": "Stressed Causal Graph",
        "filter": "Soft",
        "candidateCount": len(paths),
        "scenarioCount": len(paths),
        "displayTargetCount": displayed_paths,
        "anchorVariable": actual_anchor,
        "shockMagnitude": _round_number(actual_shock, 2),
        "horizon": horizon,
        "createdAt": created_at,
        "avgPlausibility": _round_number(weighted_mean_score, 3),
        "plausibility": {
            "mean": _round_number(weighted_mean_score, 3),
            "weightedMean": _round_number(weighted_mean_score, 3),
            "rawMean": _round_number(raw_mean_score, 3),
            "min": _round_number(float(score_arr.min()), 3) if score_arr.size else 0.0,
            "max": _round_number(float(score_arr.max()), 3) if score_arr.size else 0.0,
        },
        "focusVariables": FOCUS_VARIABLES,
        "shockTemplate": shock_template,
        "variables": variables_payload,
        "keyVariableStressRange": stress_rows,
    }


def _load_latest_row(event_type: Optional[str] = None) -> Optional[dict]:
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    has_event_type = _scenarios_has_event_type_column()
    has_anchor_variable = _scenarios_has_anchor_variable_column()
    selected_columns = [
        "id",
        "shock_variable",
        "shock_magnitude",
        "scenario_paths",
        "plausibility_scores",
        "n_scenarios",
        "created_at",
    ]
    if has_event_type:
        selected_columns.insert(3, "event_type")
    if has_anchor_variable:
        selected_columns.insert(3, "anchor_variable")

    query = f"""
        SELECT {", ".join(selected_columns)}
        FROM models.scenarios
    """
    params: List[object] = []

    if event_type:
        filters = []
        if has_event_type:
            filters.append("event_type = %s")
            params.append(event_type)

        legacy_matches = [event_type]
        default_anchor = EVENT_TYPE_TO_DEFAULT_ANCHOR.get(event_type)
        if default_anchor:
            legacy_matches.append(default_anchor)
            if has_anchor_variable:
                filters.append("anchor_variable = %s")
                params.append(default_anchor)

        filters.append("shock_variable = ANY(%s)")
        params.append(legacy_matches)
        query += " WHERE " + " OR ".join(f"({flt})" for flt in filters)

    query += " ORDER BY created_at DESC LIMIT 1"
    cursor.execute(query, params)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row


@router.get("/latest")
async def get_latest_scenarios(event_type: Optional[str] = None):
    row = _load_latest_row(event_type)
    if not row:
        raise HTTPException(status_code=404, detail="No scenarios found")

    inferred_event_type = _infer_event_type(row)
    family_id = EVENT_TYPE_TO_FAMILY_ID.get(inferred_event_type, "market-crash")
    horizon = len(next(iter(row["scenario_paths"][0]["data"].values()), [])) if row.get("scenario_paths") else 0
    return _build_scenario_response(
        scenario_id=str(row["id"]),
        family_id=family_id,
        severity=_infer_severity_from_row(row),
        horizon=horizon,
        paths=row["scenario_paths"],
        scores=row["plausibility_scores"] or [],
        displayed_paths=int(row.get("n_scenarios") or len(row["scenario_paths"]) or 0),
        event_type=inferred_event_type,
        anchor_variable=_infer_anchor_variable(row),
        shock_magnitude=row.get("shock_magnitude"),
        created_at=str(row["created_at"]),
    )


@router.post("/generate")
async def generate_scenario(request: GenerateScenarioRequest):
    if request.family_id not in SCENARIO_FAMILY_META:
        raise HTTPException(status_code=400, detail=f"Unknown scenario family: {request.family_id}")

    severity = request.severity if request.severity in SEVERITY_MULTIPLIER else "Severe"
    family_meta = SCENARIO_FAMILY_META[request.family_id]
    anchor_variable = request.anchor_variable_override or family_meta["default_anchor"]
    base_magnitude = request.anchor_magnitude_override
    if base_magnitude is None:
        base_magnitude = family_meta["default_magnitude"] * SEVERITY_MULTIPLIER[severity]

    if request.random_seed is not None:
        np.random.seed(request.random_seed)

    data = load_processed_data_with_regimes()
    target_regime = "stressed"
    graph_id, causal_adj, _ = load_regime_causal_graph(target_regime)

    var_model = fit_regime_var(
        data,
        target_regime,
        shock_variable=anchor_variable,
        train_regimes=DEFAULT_TRAINING_REGIMES,
    )

    target_scenarios = max(1, int(request.displayed_paths or get_canonical_target_scenarios()))
    candidate_scenarios = get_canonical_candidate_count(target_scenarios)
    scenarios = generate_scenarios(
        var_model=var_model,
        shock_variable=anchor_variable,
        shock_magnitude=float(base_magnitude),
        n_scenarios=candidate_scenarios,
        horizon=max(1, int(request.horizon or 60)),
        causal_adjacency=causal_adj,
        event_type=family_meta["event_type"],
    )
    scores = score_plausibility(
        scenarios,
        var_model,
        causal_adj,
        event_type=family_meta["event_type"],
    )
    scenarios, scores, scenario_weights = apply_canonical_soft_filter(scenarios, scores)

    scenario_id = store_scenarios(
        scenarios=scenarios,
        scores=scores,
        shock_variable=anchor_variable,
        shock_magnitude=float(base_magnitude),
        regime_name=target_regime,
        graph_id=str(graph_id) if graph_id else None,
        regime_condition_label=target_regime,
        event_type=family_meta["event_type"],
        scenario_weights=scenario_weights,
    )

    paths = []
    for idx, scenario in enumerate(scenarios):
        paths.append(
            {
                "scenario_idx": idx,
                "plausibility_score": float(scores[idx]),
                "soft_filter_weight": float(scenario_weights[idx]),
                "model_signature": get_canonical_signature(),
                "data": {col: scenario[col].tolist() for col in scenario.columns},
            }
        )

    return _build_scenario_response(
        scenario_id=scenario_id,
        family_id=request.family_id,
        severity=severity,
        horizon=max(1, int(request.horizon or 60)),
        paths=paths,
        scores=[float(score) for score in scores],
        displayed_paths=target_scenarios,
        event_type=family_meta["event_type"],
        anchor_variable=anchor_variable,
        shock_magnitude=float(base_magnitude),
    )


@router.get("/list")
async def list_scenarios():
    """List recent scenario sets."""
    conn = get_conn()
    cursor = conn.cursor(row_factory=dict_row)
    has_event_type = _scenarios_has_event_type_column()
    if has_event_type:
        cursor.execute(
            """
            SELECT id, shock_variable, shock_magnitude, event_type, scenario_paths, n_scenarios,
                   plausibility_scores, created_at
            FROM models.scenarios
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
    else:
        cursor.execute(
            """
            SELECT id, shock_variable, shock_magnitude, scenario_paths, n_scenarios,
                   plausibility_scores, created_at
            FROM models.scenarios
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [
        {
            "id": str(r["id"]),
            "familyId": EVENT_TYPE_TO_FAMILY_ID.get(_infer_event_type(r), "market-crash"),
            "family": SCENARIO_FAMILY_META[
                EVENT_TYPE_TO_FAMILY_ID.get(_infer_event_type(r), "market-crash")
            ]["label"],
            "eventType": _infer_event_type(r),
            "shockMagnitude": r["shock_magnitude"],
            "nScenarios": r["n_scenarios"],
            "meanPlausibility": round(float(np.mean(r["plausibility_scores"])), 3)
            if r["plausibility_scores"] else 0,
            "createdAt": str(r["created_at"]),
            "severity": _infer_severity_from_row(r),
            "horizon": len(next(iter(r["scenario_paths"][0]["data"].values()), []))
            if r.get("scenario_paths") and isinstance(r["scenario_paths"][0], dict)
            and isinstance(r["scenario_paths"][0].get("data"), dict) and r["scenario_paths"][0]["data"]
            else 0,
        }
        for r in rows
    ]
