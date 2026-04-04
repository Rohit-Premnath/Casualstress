"""
Scenario Generator
====================
Generates novel financial crisis scenarios using a VAR-based
generative model with causal constraints and regime conditioning.

Approach:
We use a Regime-Conditional Causal VAR model instead of a full
diffusion model for practical reasons:
1. It respects the causal DAG structure directly
2. It conditions on the current regime
3. It can generate from any initial shock
4. It's fast enough to generate 100s of scenarios in seconds

The model:
1. Loads the regime-conditional causal graph
2. Fits a structural VAR within each regime
3. Generates forward paths by propagating shocks through the causal graph
4. Adds calibrated noise based on regime-specific volatility
5. Scores each scenario for plausibility

This is MORE interpretable than a black-box diffusion model while
achieving the same goal: generating novel, causally-consistent,
regime-appropriate crisis scenarios.
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

SCENARIO_HORIZON = 60
N_SCENARIOS_DEFAULT = 100
NOISE_SCALE = 1.0
MAX_VAR_VARIABLES = 25

# Core variables to always include in VAR model
CORE_VARIABLES = [
    "^GSPC", "^VIX", "^NDX", "^RUT", "DGS10", "DGS2", "T10Y2Y",
    "FEDFUNDS", "CL=F", "GC=F", "BAMLH0A0HYM2", "BAMLH0A3HYC",
    "BAMLC0A0CM", "XLF", "XLK", "XLE", "XLV", "XLY", "XLU",
    "TLT", "LQD", "HYG", "EEM", "CPIAUCSL", "UNRATE",
]

# Variables that are LOG RETURNS (Yahoo prices + some FRED)
# cumulative impact = exp(sum of log returns) - 1 = simple return
LOG_RETURN_VARS = {
    "^GSPC", "^NDX", "^RUT", "^VIX", "^VVIX", "^MOVE",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLU", "XLRE",
    "TLT", "LQD", "HYG", "EEM",
    "CL=F", "GC=F", "DX-Y.NYB", "EURUSD=X",
    "CPIAUCSL", "PCEPILFE", "PAYEMS", "INDPRO", "M2SL",
    "HOUST", "ICSA", "RSXFS", "UMCSENT",
}

# Variables that are FIRST DIFFERENCES (credit spreads, rates)
# cumulative impact = sum of daily changes (in original units)
FIRST_DIFF_VARS = {
    "BAMLH0A0HYM2", "BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC",
    "BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLC0A3CA", "BAMLC0A2CAA", "BAMLC0A1CAAA",
    "BAMLEMCBPIOAS",
    "DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "TEDRATE",
    "SOFR", "SOFR90DAYAVG", "DCPF3M", "DCPN3M",
    "DRTSCILM", "DRTSCIS", "DRTSSP", "DRSDCILM",
    "UNRATE",
}

# Level variables (no transform)
LEVEL_VARS = {
    "STLFSI4", "A191RL1Q225SBEA",
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
# STEP 1: LOAD CAUSAL GRAPH AND DATA
# ============================================

def load_regime_causal_graph(regime_name):
    """Load the causal graph for a specific regime from database."""
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
# STEP 2: FIT REGIME-CONDITIONAL VAR
# ============================================

def select_var_variables(regime_data, shock_variable=None):
    """Select top variables for VAR fitting to prevent overfitting."""
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


def fit_regime_var(data, regime_name, max_lag=5, shock_variable=None):
    """Fit a VAR model within a specific regime with ridge regularization."""
    regime_data = data[data["regime_name"] == regime_name].drop(columns=["regime_name"])

    if len(regime_data.columns) > MAX_VAR_VARIABLES:
        selected_vars = select_var_variables(regime_data, shock_variable)
        regime_data = regime_data[selected_vars]
        print(f"    Selected {len(selected_vars)} variables for VAR (reduced from {len(data.columns) - 1})")

    if len(regime_data) < 50:
        print(f"    WARNING: Only {len(regime_data)} days in {regime_name}, using lag=2")
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
    cov = np.cov(residuals.T) if residuals.shape[0] > d else np.eye(d) * 0.01

    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += np.eye(d) * (abs(eigvals.min()) + 0.001)

    return {
        "coefficients": B,
        "covariance": cov,
        "means": means,
        "stds": stds,
        "variables": variables,
        "lag": effective_lag,
        "n_obs": len(regime_data),
    }


# ============================================
# STEP 3: GENERATE SCENARIOS
# ============================================

def generate_scenarios(
    var_model, shock_variable, shock_magnitude,
    n_scenarios=N_SCENARIOS_DEFAULT, horizon=SCENARIO_HORIZON,
    causal_adjacency=None,
):
    """
    Generate forward-looking crisis scenarios.
    Simulation in standardized space, clipped to ±4 sigma per step.
    """
    B = var_model["coefficients"]
    cov = var_model["covariance"]
    means = var_model["means"]
    stds = var_model["stds"]
    variables = var_model["variables"]
    lag = var_model["lag"]
    d = len(variables)

    if shock_variable not in variables:
        print(f"  WARNING: {shock_variable} not in variables, using ^GSPC")
        shock_variable = "^GSPC"

    shock_idx = variables.index(shock_variable)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov_reg = cov + np.eye(d) * 0.001
        L = np.linalg.cholesky(cov_reg)

    all_scenarios = []

    for s in range(n_scenarios):
        path = np.zeros((horizon + lag, d))
        path[lag, shock_idx] = shock_magnitude

        if causal_adjacency is not None:
            for edge_key, edge_data in causal_adjacency.items():
                cause, effect = edge_key.split("->")
                if cause == shock_variable and effect in variables:
                    effect_idx = variables.index(effect)
                    weight = edge_data.get("weight", 0)
                    path[lag, effect_idx] += shock_magnitude * weight * 0.3

        for t in range(lag + 1, horizon + lag):
            x = [1.0]
            for l_idx in range(1, lag + 1):
                x.extend(path[t - l_idx])
            x = np.array(x)

            predicted = x @ B
            noise = L @ np.random.randn(d) * NOISE_SCALE
            path[t] = predicted + noise
            # Clip per step: daily moves beyond 4 sigma are unrealistic
            path[t] = np.clip(path[t], -4.0, 4.0)

        real_path = path[lag:] * stds + means

        scenario_df = pd.DataFrame(real_path, columns=variables, index=range(horizon))
        all_scenarios.append(scenario_df)

    return all_scenarios


# ============================================
# STEP 4: SCORE SCENARIO PLAUSIBILITY
# ============================================

def score_plausibility(scenarios, var_model, causal_adjacency=None):
    """
    Score plausibility using:
    1. Daily move realism (extreme day ratio)
    2. Cumulative move bounds
    3. Financial consistency (S&P/VIX, credit/equity)
    4. Causal graph consistency
    """
    variables = var_model["variables"]
    stds = var_model["stds"]
    scores = []

    for scenario in scenarios:
        score = 1.0

        # Check 1: Daily extreme ratio
        daily_sigma = np.abs(scenario.values) / stds
        extreme_ratio = (daily_sigma > 3).sum() / daily_sigma.size
        if extreme_ratio > 0.10:
            score *= 0.80
        elif extreme_ratio > 0.05:
            score *= 0.90

        # Check 2: Cumulative bounds per variable
        for j, var in enumerate(variables):
            cum_move = abs(scenario[var].sum())
            daily_std = stds[j] if stds[j] > 0 else 1
            cum_sigma = cum_move / (daily_std * np.sqrt(len(scenario)))
            if cum_sigma > 6:
                score *= 0.97

        # Check 3: S&P / VIX
        if "^GSPC" in variables and "^VIX" in variables:
            spx = scenario["^GSPC"].sum()
            vix = scenario["^VIX"].sum()
            if spx < 0 and vix < 0:
                score *= 0.85
            elif spx > 0 and vix > 0:
                score *= 0.90

        # Check 4: Credit / Equity
        if "^GSPC" in variables and "BAMLH0A0HYM2" in variables:
            spx = scenario["^GSPC"].sum()
            hy = scenario["BAMLH0A0HYM2"].sum()
            if spx < -0.5 and hy < -0.5:
                score *= 0.85

        # Check 5: Causal consistency
        if causal_adjacency is not None:
            violations = 0
            total_checks = 0
            for edge_key, edge_data in causal_adjacency.items():
                cause, effect = edge_key.split("->")
                if cause in variables and effect in variables:
                    total_checks += 1
                    c_chg = scenario[cause].sum()
                    e_chg = scenario[effect].sum()
                    w = edge_data.get("raw_weight", edge_data.get("weight", 0))
                    if w > 0 and c_chg * e_chg < 0:
                        violations += 1
                    elif w < 0 and c_chg * e_chg > 0:
                        violations += 1
            if total_checks > 0:
                consistency = 1 - (violations / total_checks)
                score *= (0.7 + 0.3 * consistency)

        scores.append(round(min(score, 1.0), 4))

    return scores


# ============================================
# STEP 5: STORE SCENARIOS
# ============================================

def store_scenarios(scenarios, scores, shock_variable, shock_magnitude,
                    regime_name, graph_id):
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
            "data": {col: scenario[col].tolist() for col in scenario.columns},
        })

    cursor.execute("""
        INSERT INTO models.scenarios
            (id, shock_variable, shock_magnitude, regime_condition,
             causal_graph_id, scenario_paths, plausibility_scores, n_scenarios)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        scenario_id,
        shock_variable,
        shock_magnitude,
        0,
        graph_id if graph_id else None,
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
# STEP 6: SUMMARIZE SCENARIOS
# ============================================

def format_cumulative_impact(var_name, cum_values):
    """
    Format cumulative impact correctly based on variable type.
    - Log-return vars: exp(sum) - 1 as percentage
    - First-diff vars: sum as basis points or points
    - Level vars: raw cumulative change
    """
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


def summarize_scenarios(scenarios, scores, shock_variable, variables):
    """Print summary with proper units per variable type."""
    print(f"\n{'='*70}")
    print("  SCENARIO GENERATION SUMMARY")
    print(f"{'='*70}")

    n = len(scenarios)
    horizon = len(scenarios[0])

    print(f"\n  Shock: {shock_variable}")
    print(f"  Scenarios: {n}")
    print(f"  Horizon: {horizon} days")
    print(f"  Plausibility: min={min(scores):.2f}, mean={np.mean(scores):.2f}, max={max(scores):.2f}")

    key_vars = ["^GSPC", "^VIX", "DGS10", "CL=F", "XLF", "BAMLH0A0HYM2"]
    key_vars = [v for v in key_vars if v in variables]

    print(f"\n  Cumulative impact over {horizon} days (across {n} scenarios):")
    print(f"  {'Variable':<20} {'5th pctl':>12} {'Median':>12} {'95th pctl':>12} {'Mean':>12}")
    print("  " + "-" * 72)

    for var in key_vars:
        cum_raw = np.array([scenario[var].sum() for scenario in scenarios])
        display_vals, unit = format_cumulative_impact(var, cum_raw)
        display_vals = np.array(display_vals)

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
    print("CAUSALSTRESS - SCENARIO GENERATOR")
    print("=" * 70)

    print("\nLoading data and regime labels...")
    data = load_processed_data_with_regimes()
    print(f"  Loaded {len(data)} days x {len(data.columns)} columns")

    target_regime = "stressed"
    print(f"\n  Target regime for generation: {target_regime}")

    graph_id, causal_adj, graph_vars = load_regime_causal_graph(target_regime)
    if graph_id:
        print(f"  Loaded causal graph: {graph_id} ({len(causal_adj)} edges)")
    else:
        print(f"  WARNING: No causal graph found for {target_regime}")
        causal_adj = None

    shocks = [
        ("CL=F", 3.0, "Oil price spike (+3 sigma)"),
        ("^GSPC", -3.0, "Market crash (-3 sigma)"),
        ("DGS10", 2.0, "Interest rate shock (+2 sigma)"),
        ("BAMLH0A0HYM2", 3.0, "Credit spread blowout (+3 sigma)"),
    ]

    for shock_var, shock_mag, shock_desc in shocks:
        print(f"\n{'─'*70}")
        print(f"  Generating scenarios: {shock_desc}")
        print(f"{'─'*70}")

        var_model = fit_regime_var(data, target_regime, shock_variable=shock_var)
        print(f"  Variables: {len(var_model['variables'])}")
        print(f"  Observations: {var_model['n_obs']}")
        print(f"  Lag order: {var_model['lag']}")

        scenarios = generate_scenarios(
            var_model=var_model,
            shock_variable=shock_var,
            shock_magnitude=shock_mag,
            n_scenarios=100,
            horizon=60,
            causal_adjacency=causal_adj,
        )

        scores = score_plausibility(scenarios, var_model, causal_adj)

        summarize_scenarios(scenarios, scores, shock_var, var_model["variables"])

        store_scenarios(
            scenarios, scores, shock_var, shock_mag,
            target_regime, str(graph_id) if graph_id else None,
        )

    print("\n" + chr(10003) + " Scenario generation complete!")
    print(f"  Generated {len(shocks)} shock scenarios x 100 paths each = {len(shocks)*100} total scenarios")
    print("=" * 70)


if __name__ == "__main__":
    main()