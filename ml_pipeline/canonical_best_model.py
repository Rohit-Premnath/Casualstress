"""
Canonical best-model definition shared across backtest, generator, and paper code.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import numpy as np


CANONICAL_MODEL_NAME = "causal_regime_multi_root_soft_filtered_ttails_datafit"
CANONICAL_PAPER_NAME = "Canonical Soft Filtered (Student-t, data-fit df)"
CANONICAL_GRAPH_FILE = "regime_causal_graphs.json"
CANONICAL_GRAPH_REGIME = "stressed"
CANONICAL_GRAPH_MODE = "full"
CANONICAL_SCENARIO_FILTER = "soft_plausibility"
CANONICAL_TRAIN_REGIMES = ["elevated", "stressed", "high_stress", "crisis"]
CANONICAL_TARGET_SCENARIOS = 200
CANONICAL_CANDIDATE_MULTIPLIER = 2
CANONICAL_SOFT_FILTER_POWER = 6.0
CANONICAL_SOFT_FILTER_MIN_WEIGHT = 1e-6
CANONICAL_INNOVATION_MODE = "student_t_data_fit"

# Fitted via Student-t MLE on VAR residuals from the pre-2020 calibration window.
# These are the locked paper-winning tail parameters used by all_paper_experiments.py.
CANONICAL_DF_NORMAL = 5.97
CANONICAL_DF_CRISIS = 3.84
CANONICAL_DF_MID = 4.79
CANONICAL_EXTREME_NOISE_SCALE = 1.2
CANONICAL_MID_NOISE_SCALE = 1.1

CANONICAL_PAIRWISE_RULES = {
    "credit_crisis": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("XLF", "down"),
        ("BAMLH0A0HYM2", "up"),
        ("DGS10", "down"),
    ],
    "sovereign_crisis": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("BAMLH0A0HYM2", "up"),
        ("DGS10", "down"),
    ],
    "global_shock": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("CL=F", "down"),
        ("XLF", "down"),
    ],
    "volatility_shock": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("XLF", "down"),
    ],
    "rate_shock": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("DGS10", "up"),
        ("XLF", "down"),
    ],
    "pandemic_exogenous": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("CL=F", "down"),
        ("XLF", "down"),
        ("BAMLH0A0HYM2", "up"),
        ("DGS10", "down"),
    ],
    "pandemic": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("CL=F", "down"),
        ("XLF", "down"),
        ("BAMLH0A0HYM2", "up"),
        ("DGS10", "down"),
    ],
    "market_crash": [
        ("^GSPC", "down"),
        ("^VIX", "up"),
        ("XLF", "down"),
        ("BAMLH0A0HYM2", "up"),
    ],
}

CANONICAL_SIGNATURE = (
    f"{CANONICAL_MODEL_NAME} | graph={CANONICAL_GRAPH_REGIME}_{CANONICAL_GRAPH_MODE} "
    f"| filter=soft | multi_root=yes | train_regimes={','.join(CANONICAL_TRAIN_REGIMES)} "
    f"| innov=student_t_data_fit(df_n={CANONICAL_DF_NORMAL},df_c={CANONICAL_DF_CRISIS},df_mid={CANONICAL_DF_MID})"
)


def get_canonical_signature() -> str:
    return CANONICAL_SIGNATURE


def get_canonical_target_scenarios() -> int:
    return CANONICAL_TARGET_SCENARIOS


def get_canonical_candidate_count(target_scenarios: int | None = None) -> int:
    base = target_scenarios or CANONICAL_TARGET_SCENARIOS
    return base * CANONICAL_CANDIDATE_MULTIPLIER


def canonical_best_model_config() -> Dict[str, object]:
    return {
        "name": CANONICAL_MODEL_NAME,
        "paper_name": CANONICAL_PAPER_NAME,
        "graph_file": CANONICAL_GRAPH_FILE,
        "graph_regime": CANONICAL_GRAPH_REGIME,
        "graph_mode": CANONICAL_GRAPH_MODE,
        "scenario_filter": CANONICAL_SCENARIO_FILTER,
        "train_regimes": list(CANONICAL_TRAIN_REGIMES),
        "target_scenarios": CANONICAL_TARGET_SCENARIOS,
        "candidate_multiplier": CANONICAL_CANDIDATE_MULTIPLIER,
        "soft_filter_power": CANONICAL_SOFT_FILTER_POWER,
        "soft_filter_min_weight": CANONICAL_SOFT_FILTER_MIN_WEIGHT,
        "innovation_mode": CANONICAL_INNOVATION_MODE,
        "df_normal": CANONICAL_DF_NORMAL,
        "df_crisis": CANONICAL_DF_CRISIS,
        "df_mid": CANONICAL_DF_MID,
        "extreme_noise_scale": CANONICAL_EXTREME_NOISE_SCALE,
        "mid_noise_scale": CANONICAL_MID_NOISE_SCALE,
        "signature": CANONICAL_SIGNATURE,
    }


def expected_direction_ok(value: float, direction: str) -> bool:
    if direction == "up":
        return value >= 0
    if direction == "down":
        return value < 0
    return False


def weighted_quantile(values: Iterable[float], quantile: float, weights: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    weights = np.maximum(weights, CANONICAL_SOFT_FILTER_MIN_WEIGHT)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    target = float(np.clip(quantile, 0.0, 1.0)) * total
    idx = np.searchsorted(cumulative, target, side="left")
    idx = int(np.clip(idx, 0, len(values) - 1))
    return float(values[idx])


def soft_filter_weights(scores: Iterable[float]) -> np.ndarray:
    weights = np.asarray(list(scores), dtype=float)
    if weights.size == 0:
        return weights
    weights = np.maximum(weights, CANONICAL_SOFT_FILTER_MIN_WEIGHT)
    weights = np.power(weights, CANONICAL_SOFT_FILTER_POWER)
    total = weights.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def score_canonical_plausibility(
    scenarios,
    available_vars: List[str],
    stds: Iterable[float],
    event_type: str,
    causal_adj: Dict[str, dict] | None = None,
) -> List[float]:
    scores: List[float] = []
    expected_rules = CANONICAL_PAIRWISE_RULES.get(event_type, [])
    safe_stds = np.where(np.asarray(list(stds), dtype=float) > 0, np.asarray(list(stds), dtype=float), 1.0)

    for scenario in scenarios:
        score = 1.0
        daily_sigma = np.abs(scenario[available_vars].values) / safe_stds
        extreme_ratio = float((daily_sigma > 3).sum()) / daily_sigma.size
        if extreme_ratio > 0.15:
            score *= 0.80
        elif extreme_ratio > 0.08:
            score *= 0.90

        for var, direction in expected_rules:
            if var not in scenario.columns:
                continue
            cum_move = float(scenario[var].sum())
            if not expected_direction_ok(cum_move, direction):
                score *= 0.92

        if "^GSPC" in scenario.columns and "^VIX" in scenario.columns:
            spx = float(scenario["^GSPC"].sum())
            vix = float(scenario["^VIX"].sum())
            if spx < 0 and vix < 0:
                score *= 0.85
            elif spx > 0 and vix > 0:
                score *= 0.90

        if "^GSPC" in scenario.columns and "BAMLH0A0HYM2" in scenario.columns:
            spx = float(scenario["^GSPC"].sum())
            hy = float(scenario["BAMLH0A0HYM2"].sum())
            if spx < 0 and hy < 0:
                score *= 0.85

        if causal_adj is not None:
            violations = 0
            total_checks = 0
            for edge_key, edge_data in causal_adj.items():
                cause, effect = edge_key.split("->")
                if cause not in scenario.columns or effect not in scenario.columns:
                    continue
                total_checks += 1
                c_chg = float(scenario[cause].sum())
                e_chg = float(scenario[effect].sum())
                weight = edge_data.get("weight", 0.0)
                if weight > 0 and c_chg * e_chg < 0:
                    violations += 1
                elif weight < 0 and c_chg * e_chg > 0:
                    violations += 1
            if total_checks > 0:
                consistency = 1 - (violations / total_checks)
                score *= (0.7 + 0.3 * consistency)

        scores.append(round(min(score, 1.0), 4))

    return scores


def build_edge_map(edges: List[dict]) -> Dict[str, dict]:
    edge_map: Dict[str, dict] = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target or source == target:
            continue
        key = f"{source}->{target}"
        score = abs(edge.get("weight", 0.0))
        existing = edge_map.get(key)
        if existing is None or score > abs(existing.get("weight", 0.0)):
            edge_map[key] = {
                "weight": float(edge.get("weight", 0.0)),
                "lag": int(edge.get("lag", 0)),
                "edge_type": edge.get("edge_type", "unknown"),
                "confidence": float(edge.get("confidence", edge.get("dyno_confidence", 1.0) or 0.0)),
                "method": edge.get("method", "unknown"),
            }
    return edge_map


def prune_causal_graph(edge_map: Dict[str, dict], mode: str = "full") -> Dict[str, dict]:
    if not edge_map or mode == "full":
        return edge_map

    min_abs_weight = 0.12
    min_confidence = 0.55
    max_children_per_source = 4
    contemporaneous_bonus = 0.05

    source_buckets: Dict[str, list] = {}
    for edge_key, edge_data in edge_map.items():
        weight = abs(edge_data.get("weight", 0.0))
        confidence = edge_data.get("confidence", 1.0)
        lag = edge_data.get("lag", 0)
        if weight < min_abs_weight or confidence < min_confidence:
            continue
        score = weight * (1.0 + min(confidence, 1.0))
        if lag == 0:
            score += contemporaneous_bonus
        source = edge_key.split("->", 1)[0]
        source_buckets.setdefault(source, []).append((score, edge_key, edge_data))

    pruned: Dict[str, dict] = {}
    for source, rows in source_buckets.items():
        rows.sort(key=lambda item: item[0], reverse=True)
        for _, edge_key, edge_data in rows[:max_children_per_source]:
            pruned[edge_key] = edge_data
    return pruned


def load_canonical_graph(base_dir: str) -> Dict[str, dict] | None:
    search_paths = [
        os.path.join(base_dir, CANONICAL_GRAPH_FILE),
        os.path.join(os.getcwd(), "ml_pipeline", CANONICAL_GRAPH_FILE),
        os.path.join(os.getcwd(), CANONICAL_GRAPH_FILE),
    ]
    filepath = next((p for p in search_paths if os.path.exists(p)), None)
    if filepath is None:
        return None
    with open(filepath, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    edges = payload.get("regimes", {}).get(CANONICAL_GRAPH_REGIME, {}).get("edges", [])
    edge_map = build_edge_map(edges)
    return prune_causal_graph(edge_map, mode=CANONICAL_GRAPH_MODE)
