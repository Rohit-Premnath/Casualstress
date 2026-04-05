"""
Multi-Event Backtest v2
========================
Tests CausalStress scenario generator against multiple historical crises.
Uses all 5 improvements: multi-shock, causal propagation, crisis covariance,
higher clipping, regime-specific dynamics.

Also includes COVID test with proper framing.
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import uuid
import warnings
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


EVENTS = [
    {
        "name": "2008 Global Financial Crisis",
        "cutoff": "2007-06-01",
        "event_start": "2007-10-09",
        "event_end": "2009-03-09",
        "compare_window": 60,
        "type": "credit_crisis",
    },
    {
        "name": "2010 Flash Crash / Euro Stress",
        "cutoff": "2010-03-01",
        "event_start": "2010-04-26",
        "event_end": "2010-06-04",
        "compare_window": 30,
        "type": "sovereign_crisis",
    },
    {
        "name": "2011 US Debt Downgrade / Euro Crisis",
        "cutoff": "2011-06-01",
        "event_start": "2011-07-07",
        "event_end": "2011-10-03",
        "compare_window": 60,
        "type": "sovereign_crisis",
    },
    {
        "name": "2015 China Devaluation / Oil Crash",
        "cutoff": "2015-07-01",
        "event_start": "2015-08-10",
        "event_end": "2016-02-11",
        "compare_window": 60,
        "type": "global_shock",
    },
    {
        "name": "2016 Brexit Shock",
        "cutoff": "2016-05-15",
        "event_start": "2016-06-23",
        "event_end": "2016-07-15",
        "compare_window": 20,
        "type": "sovereign_crisis",
    },
    {
        "name": "2018 Volmageddon",
        "cutoff": "2018-01-01",
        "event_start": "2018-01-26",
        "event_end": "2018-04-02",
        "compare_window": 45,
        "type": "volatility_shock",
    },
    {
        "name": "2018 Q4 Fed Tightening Selloff",
        "cutoff": "2018-09-01",
        "event_start": "2018-09-20",
        "event_end": "2018-12-24",
        "compare_window": 60,
        "type": "rate_shock",
    },
    {
        "name": "2020 COVID Crash",
        "cutoff": "2020-02-01",
        "event_start": "2020-02-19",
        "event_end": "2020-03-23",
        "compare_window": 24,
        "type": "pandemic_exogenous",
    },
    {
        "name": "2020 September Tech Selloff",
        "cutoff": "2020-08-15",
        "event_start": "2020-09-02",
        "event_end": "2020-09-23",
        "compare_window": 15,
        "type": "volatility_shock",
    },
    {
        "name": "2022 Rate Hike Selloff",
        "cutoff": "2021-12-01",
        "event_start": "2022-01-03",
        "event_end": "2022-06-16",
        "compare_window": 60,
        "type": "rate_shock",
    },
    {
        "name": "2023 SVB Banking Crisis",
        "cutoff": "2023-02-15",
        "event_start": "2023-03-08",
        "event_end": "2023-03-20",
        "compare_window": 10,
        "type": "credit_crisis",
    },
]

CORE_VARS = [
    "^GSPC", "^VIX", "^NDX", "^RUT", "DGS10", "DGS2", "T10Y2Y",
    "CL=F", "GC=F", "BAMLH0A0HYM2",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLU",
    "TLT", "LQD", "HYG", "EEM",
]

LOG_RETURN_VARS = {
    "^GSPC", "^NDX", "^RUT", "^VIX", "XLF", "XLK", "XLE", "XLV",
    "XLY", "XLU", "TLT", "LQD", "HYG", "EEM", "CL=F", "GC=F",
}

PAIRWISE_RULES = {
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
}

BACKTEST_VARIANTS = [
    {
        "name": "baseline_no_causal",
        "label": "Baseline VAR only",
        "graph_file": None,
        "regime": None,
        "train_regimes": None,
    },
    {
        "name": "causal_global",
        "label": "VAR + global causal graph",
        "graph_file": "causal_graph.json",
        "regime": None,
        "train_regimes": None,
    },
    {
        "name": "causal_stressed",
        "label": "VAR + stressed regime graph",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": None,
    },
    {
        "name": "baseline_regime_stress",
        "label": "Regime-conditioned VAR",
        "graph_file": None,
        "regime": None,
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
    },
    {
        "name": "causal_regime_stress",
        "label": "Regime-conditioned VAR + stressed graph",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": False,
    },
    {
        "name": "baseline_regime_multi_root",
        "label": "Regime-conditioned VAR + multi-root shocks",
        "graph_file": None,
        "regime": None,
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": True,
    },
    {
        "name": "causal_regime_multi_root",
        "label": "Regime-conditioned VAR + stressed graph + multi-root shocks",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": True,
        "graph_mode": "full",
    },
    {
        "name": "causal_regime_multi_root_pruned",
        "label": "Regime-conditioned VAR + pruned stressed graph + multi-root shocks",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": True,
        "graph_mode": "pruned",
    },
    {
        "name": "causal_regime_multi_root_filtered",
        "label": "Regime-conditioned VAR + stressed graph + multi-root shocks + filtering",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": True,
        "graph_mode": "full",
        "scenario_filter": "plausibility",
    },
    {
        "name": "causal_regime_multi_root_pruned_filtered",
        "label": "Regime-conditioned VAR + pruned stressed graph + multi-root shocks + filtering",
        "graph_file": "regime_causal_graphs.json",
        "regime": "stressed",
        "train_regimes": ["elevated", "stressed", "high_stress", "crisis"],
        "multi_root": True,
        "graph_mode": "pruned",
        "scenario_filter": "plausibility",
    },
]

for _variant in BACKTEST_VARIANTS:
    _variant.setdefault("multi_root", False)
    _variant.setdefault("graph_mode", "full")
    _variant.setdefault("scenario_filter", None)


EVENT_SHOCK_TEMPLATES = {
    "credit_crisis": {
        "^GSPC": -3.0,
        "^VIX": 3.5,
        "XLF": -3.5,
        "BAMLH0A0HYM2": 3.0,
        "DGS10": -1.5,
    },
    "sovereign_crisis": {
        "^GSPC": -3.0,
        "^VIX": 3.0,
        "BAMLH0A0HYM2": 2.5,
        "DGS10": -2.0,
        "XLF": -2.5,
    },
    "global_shock": {
        "^GSPC": -2.0,
        "^VIX": 2.5,
        "CL=F": -3.0,
        "XLF": -1.5,
        "BAMLH0A0HYM2": 1.5,
    },
    "volatility_shock": {
        "^GSPC": -2.0,
        "^VIX": 4.0,
        "XLF": -2.0,
        "BAMLH0A0HYM2": 1.0,
    },
    "rate_shock": {
        "^GSPC": -1.5,
        "^VIX": 2.0,
        "DGS10": 3.0,
        "TLT": -2.5,
        "XLF": -1.5,
    },
    "pandemic_exogenous": {
        "^GSPC": -4.0,
        "^VIX": 5.0,
        "CL=F": -5.0,
        "XLF": -4.0,
        "BAMLH0A0HYM2": 4.0,
        "DGS10": -2.5,
    },
}


def ensure_positive_definite(cov, d):
    """Ensure covariance matrix is positive definite for Cholesky."""
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 0:
        cov = cov + np.eye(d) * (abs(eigvals.min()) + 0.01)
    # Double check
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(d) * 0.1
    return cov


def load_all_data():
    conn = get_db()
    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)
    conn.close()

    pivoted = df.pivot_table(index="date", columns="variable_code", values="transformed_value")
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.7))
    pivoted = pivoted.dropna()
    return pivoted


def load_regime_series():
    conn = get_db()
    df = pd.read_sql("""
        SELECT date, regime_name
        FROM models.regimes
        ORDER BY date
    """, conn)
    conn.close()

    if df.empty:
        return pd.Series(dtype="object", name="regime_name")

    df["date"] = pd.to_datetime(df["date"])
    regime_series = df.drop_duplicates(subset=["date"]).set_index("date")["regime_name"]
    regime_series.name = "regime_name"
    return regime_series


def build_edge_map(edges):
    """Convert graph JSON edge list into the adjacency format used by the generator."""
    edge_map = {}
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


def prune_causal_graph(edge_map, mode="full"):
    """Keep only stronger, more credible edges for propagation."""
    if not edge_map or mode == "full":
        return edge_map

    min_abs_weight = 0.12
    min_confidence = 0.55
    max_children_per_source = 4
    contemporaneous_bonus = 0.05

    source_buckets = {}
    for edge_key, edge_data in edge_map.items():
        weight = abs(edge_data.get("weight", 0.0))
        confidence = edge_data.get("confidence", 1.0)
        lag = edge_data.get("lag", 0)
        if weight < min_abs_weight:
            continue
        if confidence < min_confidence:
            continue
        score = weight * (1.0 + min(confidence, 1.0))
        if lag == 0:
            score += contemporaneous_bonus
        source = edge_key.split("->", 1)[0]
        source_buckets.setdefault(source, []).append((score, edge_key, edge_data))

    pruned = {}
    for source, rows in source_buckets.items():
        rows.sort(key=lambda item: item[0], reverse=True)
        for _, edge_key, edge_data in rows[:max_children_per_source]:
            pruned[edge_key] = edge_data

    return pruned


def load_causal_graph(graph_file=None, regime=None, mode="full"):
    """Load a causal graph from local JSON exports."""
    if not graph_file:
        return None

    search_paths = [
        os.path.join(os.path.dirname(__file__), graph_file),
        os.path.join(os.getcwd(), graph_file),
    ]

    filepath = next((p for p in search_paths if os.path.exists(p)), None)
    if filepath is None:
        print(f"  WARNING: Graph file not found: {graph_file}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if regime:
        edges = payload.get("regimes", {}).get(regime, {}).get("edges", [])
    else:
        edges = payload.get("edges", [])

    edge_map = build_edge_map(edges)
    edge_map = prune_causal_graph(edge_map, mode=mode)
    if not edge_map:
        print(f"  WARNING: No usable edges found in {graph_file}")
        return None

    return edge_map


def get_event_shock_template(event, available_vars):
    """Return the event-family shock template filtered to available variables."""
    template = EVENT_SHOCK_TEMPLATES.get(event.get("type"), {"^GSPC": -3.0, "^VIX": 2.5})
    filtered = {var: sigma for var, sigma in template.items() if var in available_vars}
    if "^GSPC" not in filtered and "^GSPC" in available_vars:
        filtered["^GSPC"] = -3.0
    return filtered


def convert_to_display_units(value, var):
    if var in LOG_RETURN_VARS:
        return (np.exp(value) - 1) * 100, "%"
    return value * 100, "bps"


def expected_direction_ok(value, direction):
    if direction == "up":
        return value >= 0
    if direction == "down":
        return value < 0
    return False


def summarize_variant_metrics(rows):
    metrics = {
        "coverage": np.mean([s["coverage"] for s in rows]) if rows else 0.0,
        "direction": np.mean([s["direction"] for s in rows]) if rows else 0.0,
        "median_abs_error": np.mean([s["median_abs_error"] for s in rows]) if rows else 0.0,
        "tail_miss_penalty": np.mean([s["tail_miss_penalty"] for s in rows]) if rows else 0.0,
        "pairwise_consistency": np.mean([s["pairwise_consistency"] for s in rows]) if rows else 0.0,
    }
    metrics["benchmark_score"] = (
        metrics["coverage"] * 0.35
        + metrics["direction"] * 0.25
        + metrics["pairwise_consistency"] * 0.20
        + max(0.0, 100.0 - metrics["median_abs_error"]) * 0.10
        + max(0.0, 100.0 - metrics["tail_miss_penalty"]) * 0.10
    )
    return metrics


def score_scenario_plausibility(scenarios, available_vars, stds, event_type, causal_adj=None):
    scores = []
    expected_rules = PAIRWISE_RULES.get(event_type, [])

    for scenario in scenarios:
        score = 1.0
        safe_stds = np.where(stds > 0, stds, 1.0)

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


def generate_scenarios(
    train_data,
    available_vars,
    n_scenarios=200,
    shock_sigma=-3.0,
    causal_adj=None,
    shock_template=None,
):
    data = train_data[available_vars]
    values = data.values
    d = len(available_vars)
    T = len(values)

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    lag = 5
    if T - lag < d * lag + 10:
        lag = max(1, (T - 10) // d)

    Y = standardized[lag:]
    X_parts = [np.ones((T - lag, 1))]
    for l in range(1, lag + 1):
        X_parts.append(standardized[lag - l:T - l])
    X = np.hstack(X_parts)

    ridge = 0.01
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    B = np.linalg.solve(XtX, XtY)

    residuals = Y - X @ B
    cov_normal = np.cov(residuals.T)
    if cov_normal.shape[0] != d:
        cov_normal = np.eye(d) * 0.01
    cov_normal = ensure_positive_definite(cov_normal, d)

    # Crisis covariance from worst 10% of days
    spx_idx = available_vars.index("^GSPC") if "^GSPC" in available_vars else 0
    spx_abs = np.abs(standardized[:, spx_idx])
    threshold = np.percentile(spx_abs, 90)
    crisis_mask = spx_abs >= threshold

    if crisis_mask.sum() > d + 5:
        crisis_data = standardized[crisis_mask]
        if crisis_data.shape[0] > d:
            cov_crisis = np.cov(crisis_data.T)
            if cov_crisis.shape[0] != d:
                cov_crisis = cov_normal * 2.0
            else:
                cov_crisis = ensure_positive_definite(cov_crisis, d)
        else:
            cov_crisis = cov_normal * 2.0
            cov_crisis = ensure_positive_definite(cov_crisis, d)
    else:
        cov_crisis = cov_normal * 2.0
        cov_crisis = ensure_positive_definite(cov_crisis, d)

    L_normal = np.linalg.cholesky(cov_normal)
    L_crisis = np.linalg.cholesky(cov_crisis)

    horizon = 60

    # Multi-shock distribution for fat tails
    template = shock_template or {"^GSPC": shock_sigma}
    anchor_var = "^GSPC" if "^GSPC" in template else next(iter(template))
    anchor_sigma = template.get(anchor_var, shock_sigma)
    sign = 1.0 if anchor_sigma >= 0 else -1.0
    shock_levels = []
    for sigma, pct in [(3.0, 0.35), (4.0, 0.25), (5.0, 0.20), (6.0, 0.12), (7.0, 0.08)]:
        count = max(1, int(n_scenarios * pct))
        shock_levels.extend([sign * sigma] * count)
    while len(shock_levels) < n_scenarios:
        shock_levels.append(shock_sigma)
    shock_levels = shock_levels[:n_scenarios]
    np.random.shuffle(shock_levels)

    # Build causal adjacency for propagation
    adj = {}
    if causal_adj:
        for edge_key, edge_data in causal_adj.items():
            cause, effect = edge_key.split("->")
            if cause not in adj:
                adj[cause] = []
            adj[cause].append((effect, edge_data.get("weight", 0)))

    scenarios = []
    for s in range(n_scenarios):
        current_shock = shock_levels[s]
        scale = current_shock / anchor_sigma if anchor_sigma not in (0, None) else 1.0

        initial = np.zeros(d)
        for var, sigma in template.items():
            if var in available_vars:
                initial[available_vars.index(var)] += sigma * scale

        # Propagate through causal graph (3 hops with decay)
        if adj:
            visited = {var for var in template if var in available_vars}
            current_layer = [(var, template[var] * scale) for var in template if var in available_vars]
            for depth in range(3):
                next_layer = []
                decay = 0.4 ** (depth + 1)
                for source, src_shock in current_layer:
                    for target, weight in adj.get(source, []):
                        if target in available_vars and target not in visited:
                            prop = src_shock * weight * decay
                            prop = np.clip(prop, -2.5, 2.5)
                            if abs(prop) > 0.12:
                                t_idx = available_vars.index(target)
                                initial[t_idx] += prop
                                visited.add(target)
                                next_layer.append((target, prop))
                current_layer = next_layer

        path = np.zeros((horizon + lag, d))
        path[lag, :] = initial

        # Choose covariance based on shock severity
        if abs(current_shock) >= 5.0:
            L = L_crisis
            noise_scale = 1.2
        elif abs(current_shock) >= 4.0:
            blend_cov = 0.5 * cov_crisis + 0.5 * cov_normal
            blend_cov = ensure_positive_definite(blend_cov, d)
            L = np.linalg.cholesky(blend_cov)
            noise_scale = 1.1
        else:
            L = L_normal
            noise_scale = 1.0

        for t in range(lag + 1, horizon + lag):
            x = [1.0]
            for l_idx in range(1, lag + 1):
                x.extend(path[t - l_idx])
            x = np.array(x)
            predicted = x @ B
            noise = L @ np.random.randn(d) * noise_scale
            path[t] = np.clip(predicted + noise, -6.0, 6.0)

        real_path = path[lag:] * stds + means
        scenario_df = pd.DataFrame(real_path, columns=available_vars, index=range(horizon))
        scenarios.append(scenario_df)

    return scenarios


def select_training_window(train_full, event, train_regimes=None, min_rows=500):
    """Optionally restrict fitting data to selected regimes with safe fallbacks."""
    if not train_regimes or "regime_name" not in train_full.columns:
        return train_full.drop(columns=["regime_name"], errors="ignore"), {
            "mode": "full_history",
            "n_rows": int(len(train_full)),
        }

    regime_filtered = train_full[train_full["regime_name"].isin(train_regimes)]
    if len(regime_filtered) >= min_rows:
        return regime_filtered.drop(columns=["regime_name"], errors="ignore"), {
            "mode": "regime_filtered",
            "regimes": list(train_regimes),
            "n_rows": int(len(regime_filtered)),
        }

    # For early events, widen to include calm/normal data rather than fail outright.
    fallback = train_full.drop(columns=["regime_name"], errors="ignore")
    return fallback, {
        "mode": "fallback_full_history",
        "regimes": list(train_regimes),
        "n_rows": int(len(fallback)),
        "filtered_rows": int(len(regime_filtered)),
    }


def run_backtest(
    all_data,
    event,
    causal_adj=None,
    variant_name="baseline_no_causal",
    train_regimes=None,
    multi_root=False,
    scenario_filter=None,
):
    cutoff = pd.to_datetime(event["cutoff"])
    event_start = pd.to_datetime(event["event_start"])
    event_end = pd.to_datetime(event["event_end"])

    train_full = all_data[all_data.index < cutoff]
    actual = all_data[(all_data.index >= event_start) & (all_data.index <= event_end)]
    train, train_meta = select_training_window(train_full, event, train_regimes=train_regimes)

    available_vars = [v for v in CORE_VARS if v in train.columns]
    compare_days = min(event["compare_window"], len(actual), 60)

    if len(train) < 500:
        print(f"  SKIP: Not enough training data ({len(train)} days)")
        return None

    shock_template = get_event_shock_template(event, available_vars) if multi_root else None
    requested_scenarios = 200
    generated_scenarios = 400 if scenario_filter == "plausibility" else requested_scenarios
    scenarios = generate_scenarios(
        train,
        available_vars,
        generated_scenarios,
        causal_adj=causal_adj,
        shock_template=shock_template,
    )
    scenario_scores = None
    if scenario_filter == "plausibility":
        stds = train[available_vars].std().to_numpy()
        scenario_scores = score_scenario_plausibility(
            scenarios,
            available_vars,
            stds,
            event["type"],
            causal_adj=causal_adj,
        )
        ranked = sorted(zip(scenario_scores, scenarios), key=lambda item: item[0], reverse=True)
        top_ranked = ranked[:requested_scenarios]
        scenario_scores = [score for score, _ in top_ranked]
        scenarios = [scenario for _, scenario in top_ranked]

    key_vars = ["^GSPC", "^VIX", "DGS10", "CL=F", "XLF", "BAMLH0A0HYM2"]
    key_vars = [v for v in key_vars if v in available_vars and v in actual.columns]

    results = {}
    matches = 0
    direction_matches = 0
    total = 0
    median_abs_errors = []
    tail_miss_penalties = []

    for var in key_vars:
        actual_cum = actual[var].iloc[:compare_days].sum()
        pred_cums = np.array([s[var].iloc[:compare_days].sum() for s in scenarios])

        p5 = np.percentile(pred_cums, 5)
        p25 = np.percentile(pred_cums, 25)
        p50 = np.median(pred_cums)
        p75 = np.percentile(pred_cums, 75)
        p95 = np.percentile(pred_cums, 95)

        in_range = p5 <= actual_cum <= p95
        same_direction = (actual_cum >= 0 and p50 >= 0) or (actual_cum < 0 and p50 < 0)
        abs_error = abs(actual_cum - p50)
        denom = max(abs(actual_cum), 1e-6)
        normalized_abs_error = min((abs_error / denom) * 100, 500)
        if actual_cum < p5:
            tail_gap = p5 - actual_cum
        elif actual_cum > p95:
            tail_gap = actual_cum - p95
        else:
            tail_gap = 0.0
        normalized_tail_penalty = min((tail_gap / denom) * 100, 500)

        total += 1
        if in_range:
            matches += 1
        if same_direction:
            direction_matches += 1
        median_abs_errors.append(normalized_abs_error)
        tail_miss_penalties.append(normalized_tail_penalty)

        a_disp, unit = convert_to_display_units(actual_cum, var)
        p5_d, _ = convert_to_display_units(p5, var)
        p50_d, _ = convert_to_display_units(p50, var)
        p95_d, _ = convert_to_display_units(p95, var)

        range_str = "IN RANGE" if in_range else "MISSED"
        dir_str = "DIR OK" if same_direction else "DIR WRONG"
        print(f"    {var:<18} actual={a_disp:>+8.1f}{unit}  pred=[{p5_d:>+7.1f}, {p50_d:>+7.1f}, {p95_d:>+7.1f}]{unit}  {range_str:>8}  {dir_str}")

        results[var] = {
            "actual": round(float(a_disp), 1),
            "p5": round(float(p5_d), 1),
            "median": round(float(p50_d), 1),
            "p95": round(float(p95_d), 1),
            "in_range": bool(in_range),
            "direction_match": bool(same_direction),
            "median_abs_error_pct": round(float(normalized_abs_error), 1),
            "tail_miss_penalty_pct": round(float(normalized_tail_penalty), 1),
        }

    coverage = matches / total * 100 if total > 0 else 0
    dir_coverage = direction_matches / total * 100 if total > 0 else 0
    pairwise_checks = []
    for var, direction in PAIRWISE_RULES.get(event["type"], []):
        if var not in results:
            continue
        pairwise_checks.append(100.0 if expected_direction_ok(results[var]["median"], direction) else 0.0)
    pairwise_consistency = np.mean(pairwise_checks) if pairwise_checks else 0.0
    median_abs_error = np.mean(median_abs_errors) if median_abs_errors else 0.0
    tail_miss_penalty = np.mean(tail_miss_penalties) if tail_miss_penalties else 0.0

    return {
        "variant": variant_name,
        "train_meta": train_meta,
        "shock_template": shock_template,
        "scenario_filter": scenario_filter,
        "filter_stats": {
            "generated": generated_scenarios,
            "kept": len(scenarios),
            "score_mean": round(float(np.mean(scenario_scores)), 4) if scenario_scores else None,
            "score_min": round(float(np.min(scenario_scores)), 4) if scenario_scores else None,
            "score_max": round(float(np.max(scenario_scores)), 4) if scenario_scores else None,
        },
        "coverage": coverage,
        "direction": dir_coverage,
        "median_abs_error": median_abs_error,
        "tail_miss_penalty": tail_miss_penalty,
        "pairwise_consistency": pairwise_consistency,
        "results": results,
    }


# ============ MAIN ============
print("=" * 90)
print("  CAUSALSTRESS - MULTI-EVENT BACKTEST v2")
print("  (Ablation: baseline vs causal graph variants)")
print("=" * 90)

all_data = load_all_data()
regime_series = load_regime_series()
all_data = all_data.join(regime_series, how="left")
print(f"  Total data: {len(all_data)} days")
if regime_series.empty:
    print("  Regime labels: not found\n")
else:
    labeled_pct = all_data["regime_name"].notna().mean() * 100
    print(f"  Regime labels joined: {labeled_pct:.1f}% coverage\n")

graphs = {}
for variant in BACKTEST_VARIANTS:
    graphs[variant["name"]] = load_causal_graph(
        graph_file=variant["graph_file"],
        regime=variant["regime"],
        mode=variant["graph_mode"],
    )
    if variant["graph_file"]:
        n_edges = len(graphs[variant["name"]]) if graphs[variant["name"]] else 0
        print(f"  Loaded {variant['name']}: {n_edges} edges")

summary = []

for variant in BACKTEST_VARIANTS:
    print(f"\n\n{'='*90}")
    print(f"  VARIANT: {variant['label']} [{variant['name']}]")
    print(f"{'='*90}")

    variant_results = []
    for event in EVENTS:
        print(f"\n{'-'*90}")
        print(f"  {event['name']} [{event['type']}]")
        print(f"  Train: before {event['cutoff']} | Event: {event['event_start']} to {event['event_end']}")
        print(f"{'-'*90}")

        result = run_backtest(
            all_data,
            event,
            causal_adj=graphs[variant["name"]],
            variant_name=variant["name"],
            train_regimes=variant["train_regimes"],
            multi_root=variant["multi_root"],
            scenario_filter=variant["scenario_filter"],
        )
        if result:
            row = {"event": event["name"], "type": event["type"], **result}
            variant_results.append(row)
            summary.append(row)
            train_meta = result.get("train_meta", {})
            shock_template = result.get("shock_template") or {"^GSPC": -3.0}
            filter_stats = result.get("filter_stats", {})
            print(f"\n  Shock Template: {shock_template}")
            print(
                f"\n  Train Mode: {train_meta.get('mode', 'unknown')} "
                f"({train_meta.get('n_rows', 0)} rows)"
            )
            if result.get("scenario_filter"):
                print(
                    f"  Filter: {result['scenario_filter']} "
                    f"(generated {filter_stats.get('generated')}, kept {filter_stats.get('kept')}, "
                    f"mean score {filter_stats.get('score_mean')})"
                )
            print(f"\n  Range Coverage: {result['coverage']:.0f}% | Direction Accuracy: {result['direction']:.0f}%")

    if not variant_results:
        continue

    print(
        f"\n  {'Event':<40} {'Coverage':>10} {'Direction':>10} "
        f"{'MedAE':>10} {'TailMiss':>10} {'Pairwise':>10}"
    )
    print(f"  {'-'*100}")
    for s in variant_results:
        print(
            f"  {s['event']:<40} {s['coverage']:>8.0f}% {s['direction']:>8.0f}% "
            f"{s['median_abs_error']:>9.1f} {s['tail_miss_penalty']:>9.1f} "
            f"{s['pairwise_consistency']:>9.1f}%"
        )

    financial = [s for s in variant_results if s["type"] != "pandemic_exogenous"]
    exogenous = [s for s in variant_results if s["type"] == "pandemic_exogenous"]
    overall_metrics = summarize_variant_metrics(variant_results)

    if financial:
        fin_metrics = summarize_variant_metrics(financial)
        print(
            f"\n  {'FINANCIAL CRISES AVG':<40} "
            f"{fin_metrics['coverage']:>8.1f}% {fin_metrics['direction']:>8.1f}% "
            f"{fin_metrics['median_abs_error']:>9.1f} {fin_metrics['tail_miss_penalty']:>9.1f} "
            f"{fin_metrics['pairwise_consistency']:>9.1f}%"
        )

    if exogenous:
        exo_metrics = summarize_variant_metrics(exogenous)
        print(
            f"  {'EXOGENOUS SHOCKS AVG':<40} "
            f"{exo_metrics['coverage']:>8.1f}% {exo_metrics['direction']:>8.1f}% "
            f"{exo_metrics['median_abs_error']:>9.1f} {exo_metrics['tail_miss_penalty']:>9.1f} "
            f"{exo_metrics['pairwise_consistency']:>9.1f}%"
        )

    print(
        f"  {'OVERALL AVERAGE':<40} "
        f"{overall_metrics['coverage']:>8.1f}% {overall_metrics['direction']:>8.1f}% "
        f"{overall_metrics['median_abs_error']:>9.1f} {overall_metrics['tail_miss_penalty']:>9.1f} "
        f"{overall_metrics['pairwise_consistency']:>9.1f}%"
    )
    print(f"  {'BENCHMARK SCORE':<40} {overall_metrics['benchmark_score']:>8.1f}")

print(f"\n\n{'='*90}")
print("  VARIANT COMPARISON")
print(f"{'='*90}")
print(
    f"\n  {'Variant':<32} {'Coverage':>10} {'Direction':>10} "
    f"{'MedAE':>10} {'TailMiss':>10} {'Pairwise':>10} {'Score':>10}"
)
print(f"  {'-'*100}")
for variant in BACKTEST_VARIANTS:
    rows = [s for s in summary if s["variant"] == variant["name"]]
    if not rows:
        continue
    metrics = summarize_variant_metrics(rows)
    print(
        f"  {variant['name']:<32} "
        f"{metrics['coverage']:>8.1f}% "
        f"{metrics['direction']:>8.1f}% "
        f"{metrics['median_abs_error']:>9.1f} "
        f"{metrics['tail_miss_penalty']:>9.1f} "
        f"{metrics['pairwise_consistency']:>9.1f}% "
        f"{metrics['benchmark_score']:>9.1f}"
    )

# Store results
try:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models.backtest_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            experiment_name VARCHAR(200),
            cutoff_date DATE,
            target_event VARCHAR(200),
            n_scenarios INTEGER,
            coverage_pct DOUBLE PRECISION,
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    for s in summary:
        result_payload = {
            "event_type": s["type"],
            "coverage": s["coverage"],
            "direction": s["direction"],
            "median_abs_error": s["median_abs_error"],
            "tail_miss_penalty": s["tail_miss_penalty"],
            "pairwise_consistency": s["pairwise_consistency"],
            "train_meta": s.get("train_meta", {}),
            "shock_template": s.get("shock_template", {}),
            "variables": s["results"],
        }
        cursor.execute("""
            INSERT INTO models.backtest_results
                (id, experiment_name, cutoff_date, target_event, n_scenarios, coverage_pct, results)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()),
            f"Multi-Event Backtest v2::{s['variant']}",
            None,
            s["event"],
            200,
            s["coverage"],
            Json(result_payload),
        ))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n  Results stored in database!")
except Exception as e:
    print(f"\n  Warning: Could not store results: {e}")

print(f"\n{'='*90}")
