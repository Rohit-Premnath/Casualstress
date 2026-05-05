"""
Scenario Generator v2
========================
Generates novel financial crisis scenarios using a VAR-based
generative model with causal constraints and regime conditioning.

v2 Improvements:
1. Stressed + Crisis regime VAR blending for realistic crisis dynamics
2. Multi-layer causal shock propagation (2-3 hops through causal graph)
3. Higher clipping (±6σ) to capture real tail events (COVID, 2008)
4. Crisis-period covariance for extreme correlation clustering
5. Multi-shock-level generation for fat-tailed distributions
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as student_t
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
try:
    from ml_pipeline.canonical_best_model import (
        CANONICAL_GRAPH_FILE,
        CANONICAL_GRAPH_MODE,
        CANONICAL_GRAPH_REGIME,
        CANONICAL_DF_CRISIS,
        CANONICAL_DF_MID,
        CANONICAL_DF_NORMAL,
        CANONICAL_EXTREME_NOISE_SCALE,
        CANONICAL_INNOVATION_MODE,
        CANONICAL_MID_NOISE_SCALE,
        CANONICAL_MODEL_NAME,
        CANONICAL_PAPER_NAME,
        CANONICAL_SCENARIO_FILTER,
        CANONICAL_TRAIN_REGIMES,
        get_canonical_candidate_count,
        get_canonical_signature,
        get_canonical_target_scenarios,
        load_canonical_graph,
        score_canonical_plausibility,
        soft_filter_weights,
        weighted_quantile,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from canonical_best_model import (
        CANONICAL_GRAPH_FILE,
        CANONICAL_GRAPH_MODE,
        CANONICAL_GRAPH_REGIME,
        CANONICAL_DF_CRISIS,
        CANONICAL_DF_MID,
        CANONICAL_DF_NORMAL,
        CANONICAL_EXTREME_NOISE_SCALE,
        CANONICAL_INNOVATION_MODE,
        CANONICAL_MID_NOISE_SCALE,
        CANONICAL_MODEL_NAME,
        CANONICAL_PAPER_NAME,
        CANONICAL_SCENARIO_FILTER,
        CANONICAL_TRAIN_REGIMES,
        get_canonical_candidate_count,
        get_canonical_signature,
        get_canonical_target_scenarios,
        load_canonical_graph,
        score_canonical_plausibility,
        soft_filter_weights,
        weighted_quantile,
    )
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

SCENARIO_HORIZON = 60
N_SCENARIOS_DEFAULT = get_canonical_target_scenarios()
MAX_VAR_VARIABLES = 25
FILTERED_SELECTION_ENABLED = CANONICAL_SCENARIO_FILTER == "soft_plausibility"

# Multi-shock distribution: generates scenarios at different severity levels
# This creates fat tails naturally
SHOCK_DISTRIBUTION = [
    {"sigma": 3.0, "count_pct": 0.40},   # 40% of scenarios at 3σ
    {"sigma": 4.0, "count_pct": 0.25},   # 25% at 4σ
    {"sigma": 5.0, "count_pct": 0.20},   # 20% at 5σ
    {"sigma": 6.0, "count_pct": 0.10},   # 10% at 6σ
    {"sigma": 7.0, "count_pct": 0.05},   # 5% at 7σ (extreme tail)
]

# Clipping: allow up to ±6σ per daily step (real crises hit 5-6σ regularly)
DAILY_CLIP_SIGMA = 6.0

# Causal propagation: how many hops through the graph
CAUSAL_PROPAGATION_DEPTH = 3
CAUSAL_PROPAGATION_DECAY = 0.4  # Each hop multiplies weight by this factor
CAUSAL_PROPAGATION_CLIP = 2.5
CAUSAL_PROPAGATION_MIN = 0.12
SHOCK_PERSISTENCE_DAYS = 5
SHOCK_PERSISTENCE_DECAY = 0.72
VIX_TEMPLATE_CAP = 3.5
VIX_PROPAGATION_SCALE = 0.65
DEFAULT_TRAINING_REGIMES = CANONICAL_TRAIN_REGIMES

CORE_VARIABLES = [
    "^GSPC", "^VIX", "^NDX", "^RUT", "DGS10", "DGS2", "T10Y2Y",
    "FEDFUNDS", "CL=F", "GC=F", "BAMLH0A0HYM2", "BAMLH0A3HYC",
    "BAMLC0A0CM", "XLF", "XLK", "XLE", "XLV", "XLY", "XLU",
    "TLT", "LQD", "HYG", "EEM", "CPIAUCSL", "UNRATE",
]

LOG_RETURN_VARS = {
    "^GSPC", "^NDX", "^RUT", "^VIX", "^VVIX", "^MOVE",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLU", "XLRE",
    "TLT", "LQD", "HYG", "EEM",
    "CL=F", "GC=F", "DX-Y.NYB", "EURUSD=X",
    "CPIAUCSL", "PCEPILFE", "PAYEMS", "INDPRO", "M2SL",
    "HOUST", "ICSA", "RSXFS", "UMCSENT",
}

FIRST_DIFF_VARS = {
    "BAMLH0A0HYM2", "BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC",
    "BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLC0A3CA", "BAMLC0A2CAA", "BAMLC0A1CAAA",
    "BAMLEMCBPIOAS",
    "DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "TEDRATE",
    "SOFR", "SOFR90DAYAVG", "DCPF3M", "DCPN3M",
    "DRTSCILM", "DRTSCIS", "DRTSSP", "DRSDCILM",
    "UNRATE",
}

EVENT_SHOCK_TEMPLATES = {
    "market_crash": {
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
    "credit_crisis": {
        "^GSPC": -3.0,
        "^VIX": 3.0,
        "XLF": -3.0,
        "BAMLH0A0HYM2": 3.0,
        "DGS10": -1.0,
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

EVENT_VARIABLE_GUARDRAILS = {
    "market_crash": {
        "^VIX": {"cum_min": 0.50, "cum_max": 4.50, "daily_abs_cap": 0.18},
        "CL=F": {"cum_abs_cap": 0.45, "daily_abs_cap": 0.08},
        "DGS10": {"cum_abs_cap": 1.20, "daily_abs_cap": 0.03},
    },
    "credit_crisis": {
        "^VIX": {"cum_min": 0.40, "cum_max": 4.00, "daily_abs_cap": 0.16},
        "CL=F": {"cum_abs_cap": 0.40, "daily_abs_cap": 0.08},
        "BAMLH0A0HYM2": {"cum_min": 0.05, "daily_abs_cap": 0.20},
    },
    "rate_shock": {
        "^VIX": {"cum_min": 0.20, "cum_max": 3.20, "daily_abs_cap": 0.12},
        "CL=F": {"cum_abs_cap": 0.55, "daily_abs_cap": 0.10},
        "DGS10": {"cum_min": 0.15, "daily_abs_cap": 0.05},
    },
    "global_shock": {
        "^VIX": {"cum_min": 0.15, "cum_max": 3.00, "daily_abs_cap": 0.12},
        "CL=F": {"cum_min": -0.35, "daily_abs_cap": 0.12},
    },
}


# ============================================
# DATABASE
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# LOAD CAUSAL GRAPH
# ============================================

def load_regime_causal_graph(regime_name):
    """Load the causal graph for a specific regime."""
    if regime_name == CANONICAL_GRAPH_REGIME:
        canonical_graph = load_canonical_graph(os.path.dirname(os.path.dirname(__file__)))
        if canonical_graph:
            return f"local::{CANONICAL_GRAPH_FILE}::{CANONICAL_GRAPH_REGIME}::{CANONICAL_GRAPH_MODE}", canonical_graph, None

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, adjacency_matrix, variables
        FROM models.causal_graphs
        WHERE method = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (f"regime_{regime_name}",))
    row = cursor.fetchone()

    if row is None:
        cursor.execute("""
            SELECT id, adjacency_matrix, variables
            FROM models.causal_graphs
            WHERE method = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (f"regime_conditional_{regime_name}",))
        row = cursor.fetchone()

    # Fallback to ensemble graph
    if row is None:
        cursor.execute("""
            SELECT id, adjacency_matrix, variables
            FROM models.causal_graphs
            WHERE method LIKE '%ensemble%' OR method LIKE '%dynotears%'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row is None:
        return None, None, None
    return row[0], row[1], row[2]


def load_processed_data_with_regimes():
    """Load processed data joined with regime labels."""
    conn = get_db_connection()

    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)

    regimes = pd.read_sql("""
        SELECT date, regime_name
        FROM models.regimes
        ORDER BY date
    """, conn)

    conn.close()

    pivoted = df.pivot_table(
        index="date", columns="variable_code", values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.7))
    pivoted = pivoted.dropna()

    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.set_index("date")
    pivoted = pivoted.join(regimes, how="inner")

    return pivoted


# ============================================
# VARIABLE SELECTION
# ============================================

def select_var_variables(regime_data, shock_variable=None):
    """Select top variables for VAR fitting."""
    available_cols = list(regime_data.columns)
    selected = [v for v in CORE_VARIABLES if v in available_cols]

    if shock_variable and shock_variable not in selected and shock_variable in available_cols:
        selected.append(shock_variable)

    if len(selected) < MAX_VAR_VARIABLES:
        remaining = [c for c in available_cols if c not in selected]
        if remaining:
            variances = regime_data[remaining].var().sort_values(ascending=False)
            extra = list(variances.index[:MAX_VAR_VARIABLES - len(selected)])
            selected.extend(extra)

    return selected[:MAX_VAR_VARIABLES]


# ============================================
# FIT VAR WITH CRISIS COVARIANCE
# ============================================

def fit_regime_var(data, regime_name, max_lag=5, shock_variable=None, train_regimes=None):
    """
    Fit VAR with two covariance matrices:
    1. Regime covariance (for normal scenario noise)
    2. Crisis covariance (from worst 10% of days - for extreme scenarios)
    """
    if train_regimes:
        regime_data = data[data["regime_name"].isin(train_regimes)].drop(columns=["regime_name"])
    else:
        regime_data = data[data["regime_name"] == regime_name].drop(columns=["regime_name"])

    if len(regime_data.columns) > MAX_VAR_VARIABLES:
        selected_vars = select_var_variables(regime_data, shock_variable)
        regime_data = regime_data[selected_vars]
        print(f"    Selected {len(selected_vars)} variables for VAR")

    if len(regime_data) < 50:
        max_lag = 2

    variables = list(regime_data.columns)
    values = regime_data.values
    d = len(variables)
    T = len(values)

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    effective_lag = min(max_lag, (T - 10) // d)
    effective_lag = max(effective_lag, 1)

    Y = standardized[effective_lag:]
    X_parts = [np.ones((T - effective_lag, 1))]
    for lag in range(1, effective_lag + 1):
        X_parts.append(standardized[effective_lag - lag:T - lag])
    X = np.hstack(X_parts)

    try:
        ridge_lambda = 0.01
        XtX = X.T @ X + ridge_lambda * np.eye(X.shape[1])
        XtY = X.T @ Y
        B = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        B = np.zeros((X.shape[1], d))

    residuals = Y - X @ B

    # Standard covariance
    cov_normal = np.cov(residuals.T) if residuals.shape[0] > d else np.eye(d) * 0.01
    eigvals = np.linalg.eigvalsh(cov_normal)
    if eigvals.min() < 0:
        cov_normal += np.eye(d) * (abs(eigvals.min()) + 0.001)

    # IMPROVEMENT 4: Crisis covariance from worst 10% of days
    # Identify days with largest portfolio moves (proxy: S&P absolute return)
    all_data_no_regime = data.drop(columns=["regime_name"])
    if "^GSPC" in all_data_no_regime.columns:
        spx_col = all_data_no_regime["^GSPC"]
        threshold = spx_col.abs().quantile(0.90)
        crisis_mask = spx_col.abs() >= threshold
        crisis_days = all_data_no_regime[crisis_mask]

        if len(crisis_days) > d + 5:
            crisis_subset = crisis_days[variables].dropna()
            if len(crisis_subset) > d:
                crisis_vals = crisis_subset.values
                crisis_standardized = (crisis_vals - means) / stds
                cov_crisis = np.cov(crisis_standardized.T)
                eigvals_c = np.linalg.eigvalsh(cov_crisis)
                if eigvals_c.min() < 0:
                    cov_crisis += np.eye(d) * (abs(eigvals_c.min()) + 0.001)
            else:
                cov_crisis = cov_normal * 2.0
        else:
            cov_crisis = cov_normal * 2.0
    else:
        cov_crisis = cov_normal * 2.0

    return {
        "coefficients": B,
        "covariance_normal": cov_normal,
        "covariance_crisis": cov_crisis,
        "means": means,
        "stds": stds,
        "variables": variables,
        "lag": effective_lag,
        "n_obs": len(regime_data),
    }


# ============================================
# MULTI-LAYER CAUSAL PROPAGATION
# ============================================

def get_shock_template(event_type, shock_variable, shock_magnitude, variables):
    """Return a multi-root shock template filtered to model variables."""
    template = dict(EVENT_SHOCK_TEMPLATES.get(event_type, {shock_variable: shock_magnitude}))
    if shock_variable not in template:
        template[shock_variable] = shock_magnitude
    filtered = {var: sigma for var, sigma in template.items() if var in variables}
    if shock_variable in variables and shock_variable not in filtered:
        filtered[shock_variable] = shock_magnitude
    if "^VIX" in filtered:
        filtered["^VIX"] = float(np.clip(filtered["^VIX"], -VIX_TEMPLATE_CAP, VIX_TEMPLATE_CAP))
    return filtered


def apply_event_guardrails(scenario_df, event_type):
    """Apply light event-family guardrails in real transformed-value space."""
    guardrails = EVENT_VARIABLE_GUARDRAILS.get(event_type)
    if not guardrails:
        return scenario_df

    for var, rules in guardrails.items():
        if var not in scenario_df.columns:
            continue
        series = scenario_df[var].to_numpy(copy=True)

        daily_cap = rules.get("daily_abs_cap")
        if daily_cap is not None:
            series = np.clip(series, -daily_cap, daily_cap)

        cum = float(series.sum())
        cum_abs_cap = rules.get("cum_abs_cap")
        if cum_abs_cap is not None and abs(cum) > cum_abs_cap:
            scale = cum_abs_cap / abs(cum) if cum != 0 else 1.0
            series *= scale

        cum_min = rules.get("cum_min")
        if cum_min is not None and cum < cum_min:
            adjustment = (cum_min - cum) / len(series)
            series += adjustment

        cum_max = rules.get("cum_max")
        if cum_max is not None and cum > cum_max:
            adjustment = (cum_max - cum) / len(series)
            series += adjustment

        scenario_df[var] = series

    return scenario_df


def draw_canonical_innovation(cholesky, df=None):
    """Draw one canonical innovation vector using the locked paper-winning tail model."""
    d = cholesky.shape[0]
    if CANONICAL_INNOVATION_MODE == "student_t_data_fit":
        tail_df = float(df or CANONICAL_DF_NORMAL)
        draws = student_t.rvs(tail_df, size=d)
        if tail_df > 2:
            draws = draws * np.sqrt((tail_df - 2.0) / tail_df)
        return cholesky @ draws
    return cholesky @ np.random.randn(d)


def compute_causal_initial_shock(shock_template, variables, causal_adjacency):
    """
    IMPROVEMENT 2: Propagate initial shock through multiple layers of causal graph.
    Instead of just shocking direct children, propagate 2-3 hops deep.

    Example: Oil shock → CPI (hop 1) → Fed Funds (hop 2) → DGS10 (hop 3)
    """
    initial_shocks = np.zeros(len(variables))
    for var, sigma in shock_template.items():
        if var in variables:
            initial_shocks[variables.index(var)] += sigma

    if causal_adjacency is None:
        return initial_shocks

    # Build adjacency dict: cause -> [(effect, weight), ...]
    adj = {}
    for edge_key, edge_data in causal_adjacency.items():
        cause, effect = edge_key.split("->")
        if cause not in adj:
            adj[cause] = []
        adj[cause].append((effect, edge_data.get("weight", 0)))

    # BFS propagation with decay
    visited = {var: sigma for var, sigma in shock_template.items() if var in variables}
    current_layer = list(visited.items())

    for depth in range(CAUSAL_PROPAGATION_DEPTH):
        next_layer = []
        decay = CAUSAL_PROPAGATION_DECAY ** (depth + 1)

        for source, source_shock in current_layer:
            for target, weight in adj.get(source, []):
                if target in variables and target not in visited:
                    propagated_shock = source_shock * weight * decay
                    if target == "^VIX":
                        propagated_shock *= VIX_PROPAGATION_SCALE
                    propagated_shock = float(np.clip(
                        propagated_shock,
                        -CAUSAL_PROPAGATION_CLIP,
                        CAUSAL_PROPAGATION_CLIP,
                    ))
                    # Only propagate if the shock is meaningful
                    if abs(propagated_shock) > CAUSAL_PROPAGATION_MIN:
                        target_idx = variables.index(target)
                        initial_shocks[target_idx] += propagated_shock
                        visited[target] = propagated_shock
                        next_layer.append((target, propagated_shock))

        current_layer = next_layer
        if not current_layer:
            break

    n_shocked = np.sum(np.abs(initial_shocks) > 0.01)
    return initial_shocks


# ============================================
# GENERATE SCENARIOS
# ============================================

def generate_scenarios(
    var_model, shock_variable, shock_magnitude,
    n_scenarios=N_SCENARIOS_DEFAULT, horizon=SCENARIO_HORIZON,
    causal_adjacency=None, use_multi_shock=True, event_type="market_crash",
):
    """
    Generate scenarios with all 5 improvements:
    1. Uses stressed-regime fitted VAR coefficients
    2. Multi-layer causal shock propagation
    3. ±6σ daily clipping (captures real tail events)
    4. Crisis covariance for extreme scenarios
    5. Multi-shock-level generation for fat tails
    """
    B = var_model["coefficients"]
    cov_normal = var_model["covariance_normal"]
    cov_crisis = var_model["covariance_crisis"]
    means = var_model["means"]
    stds = var_model["stds"]
    variables = var_model["variables"]
    lag = var_model["lag"]
    d = len(variables)

    if shock_variable not in variables:
        print(f"  WARNING: {shock_variable} not in variables, using ^GSPC")
        shock_variable = "^GSPC"

    base_template = get_shock_template(event_type, shock_variable, shock_magnitude, variables)
    anchor_var = shock_variable if shock_variable in base_template else next(iter(base_template))
    anchor_sigma = base_template.get(anchor_var, shock_magnitude)

    # Cholesky for both covariance matrices
    try:
        L_normal = np.linalg.cholesky(cov_normal)
    except np.linalg.LinAlgError:
        cov_normal += np.eye(d) * 0.001
        L_normal = np.linalg.cholesky(cov_normal)

    try:
        L_crisis = np.linalg.cholesky(cov_crisis)
    except np.linalg.LinAlgError:
        cov_crisis += np.eye(d) * 0.001
        L_crisis = np.linalg.cholesky(cov_crisis)

    all_scenarios = []

    # IMPROVEMENT 5: Multi-shock-level generation
    if use_multi_shock:
        shock_schedule = []
        for level in SHOCK_DISTRIBUTION:
            n_at_level = max(1, int(n_scenarios * level["count_pct"]))
            for _ in range(n_at_level):
                # Determine shock direction (same sign as requested)
                sign = 1.0 if anchor_sigma >= 0 else -1.0
                shock_schedule.append(sign * level["sigma"])

        # Trim or pad to exact n_scenarios
        while len(shock_schedule) < n_scenarios:
            shock_schedule.append(shock_magnitude)
        shock_schedule = shock_schedule[:n_scenarios]
        np.random.shuffle(shock_schedule)
    else:
        shock_schedule = [shock_magnitude] * n_scenarios

    for s_idx in range(n_scenarios):
        current_shock = shock_schedule[s_idx]
        scale = current_shock / anchor_sigma if anchor_sigma not in (0, None) else 1.0
        scaled_template = {var: sigma * scale for var, sigma in base_template.items()}

        # IMPROVEMENT 2: Multi-layer causal propagation
        initial_shocks = compute_causal_initial_shock(
            scaled_template, variables, causal_adjacency
        )

        path = np.zeros((horizon + lag, d))
        path[lag, :] = initial_shocks

        # IMPROVEMENT 4: Use crisis covariance for extreme shocks
        if abs(current_shock) >= 5.0:
            L = L_crisis
            noise_scale = CANONICAL_EXTREME_NOISE_SCALE
            noise_df = CANONICAL_DF_CRISIS
        elif abs(current_shock) >= 4.0:
            # Blend normal and crisis covariance
            blend = 0.5
            L_blend_cov = blend * cov_crisis + (1 - blend) * cov_normal
            eigvals = np.linalg.eigvalsh(L_blend_cov)
            if eigvals.min() < 0:
                L_blend_cov += np.eye(d) * (abs(eigvals.min()) + 0.001)
            L = np.linalg.cholesky(L_blend_cov)
            noise_scale = CANONICAL_MID_NOISE_SCALE
            noise_df = CANONICAL_DF_MID
        else:
            L = L_normal
            noise_scale = 1.0
            noise_df = CANONICAL_DF_NORMAL

        # Simulate forward
        for t in range(lag + 1, horizon + lag):
            x = [1.0]
            for l_idx in range(1, lag + 1):
                x.extend(path[t - l_idx])
            x = np.array(x)

            predicted = x @ B
            noise = draw_canonical_innovation(L, noise_df) * noise_scale

            path[t] = predicted + noise

            # Keep the crisis impulse alive for a few days so the template
            # does not disappear after the first step.
            elapsed = t - lag
            if elapsed < SHOCK_PERSISTENCE_DAYS:
                overlay = initial_shocks * (SHOCK_PERSISTENCE_DECAY ** elapsed)
                path[t] += overlay

            # IMPROVEMENT 3: Higher clipping (±6σ) for realistic tails
            path[t] = np.clip(path[t], -DAILY_CLIP_SIGMA, DAILY_CLIP_SIGMA)

        real_path = path[lag:] * stds + means
        scenario_df = pd.DataFrame(real_path, columns=variables, index=range(horizon))
        scenario_df = apply_event_guardrails(scenario_df, event_type)
        all_scenarios.append(scenario_df)

    return all_scenarios


# ============================================
# SCORE PLAUSIBILITY
# ============================================

def score_plausibility(scenarios, var_model, causal_adjacency=None, event_type="market_crash"):
    """Score plausibility with financial consistency checks."""
    return score_canonical_plausibility(
        scenarios,
        var_model["variables"],
        var_model["stds"],
        event_type,
        causal_adj=causal_adjacency,
    )


def apply_canonical_soft_filter(scenarios, scores):
    """Canonical soft-filtered selection used by the shared best model."""
    weights = soft_filter_weights(scores)
    return scenarios, scores, list(weights)


def select_top_scenarios(scenarios, scores, keep_n):
    """Legacy hard-filter helper kept for backwards compatibility."""
    ranked = sorted(zip(scores, scenarios), key=lambda item: item[0], reverse=True)
    selected = ranked[:keep_n]
    kept_scores = [score for score, _ in selected]
    kept_scenarios = [scenario for _, scenario in selected]
    return kept_scenarios, kept_scores


# ============================================
# STORE SCENARIOS
# ============================================

def store_scenarios(scenarios, scores, shock_variable, shock_magnitude,
                    regime_name, graph_id, regime_condition_label=None,
                    event_type=None, scenario_weights=None):
    """Store generated scenarios in the database."""
    print("\nStoring scenarios in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    scenario_id = str(uuid.uuid4())

    paths = []
    for i, scenario in enumerate(scenarios):
        paths.append({
            "scenario_idx": i,
            "plausibility_score": scores[i],
            "soft_filter_weight": None if scenario_weights is None else float(scenario_weights[i]),
            "model_signature": get_canonical_signature(),
            "data": {col: scenario[col].tolist() for col in scenario.columns},
        })

    graph_uuid = None
    if graph_id:
        try:
            graph_uuid = str(uuid.UUID(str(graph_id)))
        except (ValueError, TypeError, AttributeError):
            graph_uuid = None

    cursor.execute("""
        INSERT INTO models.scenarios
            (id, shock_variable, shock_magnitude, regime_condition,
             causal_graph_id, scenario_paths, plausibility_scores, n_scenarios)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        scenario_id,
        event_type or shock_variable,
        shock_magnitude,
        regime_condition_label if regime_condition_label is not None else 0,
        graph_uuid,
        Json(paths),
        Json(scores),
        len(scenarios),
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"  Stored {len(scenarios)} scenarios with ID: {scenario_id}")
    return scenario_id


# ============================================
# SUMMARIZE
# ============================================

def format_cumulative_impact(var_name, cum_values):
    """Format cumulative impact correctly based on variable type."""
    if var_name in LOG_RETURN_VARS:
        pct = [(np.exp(v) - 1) * 100 for v in cum_values]
        return pct, "%"
    elif var_name in FIRST_DIFF_VARS:
        if "BAML" in var_name or "BAMLEM" in var_name:
            return [v * 100 for v in cum_values], "bps"
        elif var_name in ("DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "TEDRATE", "SOFR"):
            return [v * 100 for v in cum_values], "bps"
        else:
            return list(cum_values), "pts"
    else:
        return list(cum_values), ""


def summarize_scenarios(scenarios, scores, shock_variable, variables, scenario_weights=None):
    """Print summary with proper units."""
    print(f"\n{'='*70}")
    print("  SCENARIO GENERATION SUMMARY")
    print(f"{'='*70}")

    n = len(scenarios)
    horizon = len(scenarios[0])

    print(f"\n  Shock: {shock_variable}")
    print(f"  Scenarios: {n}")
    print(f"  Horizon: {horizon} days")
    print(f"  Plausibility: min={min(scores):.2f}, mean={np.mean(scores):.2f}, max={max(scores):.2f}")
    if scenario_weights is not None:
        print(f"  Soft filter: weighted quantiles from canonical best model")

    key_vars = ["^GSPC", "^VIX", "DGS10", "CL=F", "XLF", "BAMLH0A0HYM2"]
    key_vars = [v for v in key_vars if v in variables]

    print(f"\n  Cumulative impact over {horizon} days (across {n} scenarios):")
    print(f"  {'Variable':<20} {'5th pctl':>12} {'Median':>12} {'95th pctl':>12} {'Mean':>12}")
    print("  " + "-" * 72)

    for var in key_vars:
        cum_raw = np.array([scenario[var].sum() for scenario in scenarios])
        display_vals, unit = format_cumulative_impact(var, cum_raw)
        display_vals = np.array(display_vals)

        if scenario_weights is not None:
            p5 = weighted_quantile(display_vals, 0.05, scenario_weights)
            p50 = weighted_quantile(display_vals, 0.50, scenario_weights)
            p95 = weighted_quantile(display_vals, 0.95, scenario_weights)
        else:
            p5 = np.percentile(display_vals, 5)
            p50 = np.percentile(display_vals, 50)
            p95 = np.percentile(display_vals, 95)
        mean = np.mean(display_vals)

        if unit == "%":
            print(f"  {var:<20} {p5:>+10.1f}% {p50:>+10.1f}% {p95:>+10.1f}% {mean:>+10.1f}%")
        elif unit == "bps":
            print(f"  {var:<20} {p5:>+9.0f}bps {p50:>+9.0f}bps {p95:>+9.0f}bps {mean:>+9.0f}bps")
        elif unit == "pts":
            print(f"  {var:<20} {p5:>+10.2f}pt {p50:>+10.2f}pt {p95:>+10.2f}pt {mean:>+10.2f}pt")
        else:
            print(f"  {var:<20} {p5:>+10.2f}   {p50:>+10.2f}   {p95:>+10.2f}   {mean:>+10.2f}")

    print(f"\n{'='*70}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("CAUSALSTRESS - SCENARIO GENERATOR v2")
    print("  (Multi-shock, causal propagation, crisis covariance)")
    print("=" * 70)
    print(f"  Canonical winner: {CANONICAL_PAPER_NAME}")
    print(f"  Model signature: {get_canonical_signature()}")

    print("\nLoading data and regime labels...")
    data = load_processed_data_with_regimes()
    print(f"  Loaded {len(data)} days x {len(data.columns)} columns")

    target_regime = CANONICAL_GRAPH_REGIME
    print(f"\n  Target graph regime: {target_regime}")
    print(f"  Training regimes: {', '.join(DEFAULT_TRAINING_REGIMES)}")
    print(f"  Graph mode: {CANONICAL_GRAPH_MODE}")
    print(f"  Filter mode: {CANONICAL_SCENARIO_FILTER}")

    graph_id, causal_adj, graph_vars = load_regime_causal_graph(target_regime)
    if graph_id:
        print(f"  Loaded causal graph: {graph_id} ({len(causal_adj)} edges)")
    else:
        print(f"  WARNING: No causal graph found for {target_regime}")
        causal_adj = None

    shocks = [
        ("global_shock", "CL=F", 3.0, "Global shock / oil crash template"),
        ("market_crash", "^GSPC", -3.0, "Market crash template"),
        ("rate_shock", "DGS10", 2.0, "Interest rate shock template"),
        ("credit_crisis", "BAMLH0A0HYM2", 3.0, "Credit crisis template"),
    ]

    for event_type, shock_var, shock_mag, shock_desc in shocks:
        print(f"\n{'-'*70}")
        print(f"  Generating scenarios: {shock_desc}")
        print(f"{'-'*70}")

        var_model = fit_regime_var(
            data,
            target_regime,
            shock_variable=shock_var,
            train_regimes=DEFAULT_TRAINING_REGIMES,
        )
        print(f"  Variables: {len(var_model['variables'])}")
        print(f"  Observations: {var_model['n_obs']}")
        print(f"  Lag order: {var_model['lag']}")
        print(f"  Event family: {event_type}")
        print(f"  Multi-root template: {get_shock_template(event_type, shock_var, shock_mag, var_model['variables'])}")

        target_scenarios = get_canonical_target_scenarios()
        candidate_scenarios = (
            get_canonical_candidate_count(target_scenarios)
            if FILTERED_SELECTION_ENABLED else target_scenarios
        )

        scenarios = generate_scenarios(
            var_model=var_model,
            shock_variable=shock_var,
            shock_magnitude=shock_mag,
            n_scenarios=candidate_scenarios,
            horizon=60,
            causal_adjacency=causal_adj,
            event_type=event_type,
        )

        scores = score_plausibility(scenarios, var_model, causal_adj, event_type=event_type)
        scenario_weights = None
        if FILTERED_SELECTION_ENABLED:
            scenarios, scores, scenario_weights = apply_canonical_soft_filter(scenarios, scores)
            print(
                f"  Soft-filtered weighting: generated {candidate_scenarios}, "
                f"weighted over {len(scenarios)} scenarios, mean score {np.mean(scores):.2f}"
            )

        summarize_scenarios(scenarios, scores, shock_var, var_model["variables"], scenario_weights=scenario_weights)

        store_scenarios(
            scenarios, scores, shock_var, shock_mag,
            target_regime,
            str(graph_id) if graph_id else None,
            regime_condition_label=target_regime,
            event_type=event_type,
            scenario_weights=scenario_weights,
        )

    print("\nScenario generation complete!")
    print(f"  Generated {len(shocks)} shock scenarios x {get_canonical_candidate_count(get_canonical_target_scenarios())} weighted paths each = {len(shocks)*get_canonical_candidate_count(get_canonical_target_scenarios())} total scenarios")
    print("=" * 70)


if __name__ == "__main__":
    main()
