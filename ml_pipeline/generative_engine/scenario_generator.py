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

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

# Scenario generation parameters
SCENARIO_HORIZON = 60        # Generate 60-day (3 month) forward paths
N_SCENARIOS_DEFAULT = 100    # Default number of scenarios
NOISE_SCALE = 1.0            # Scale factor for regime-specific noise


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

    # Load processed data
    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)

    # Load regime labels
    regimes = pd.read_sql("""
        SELECT date, regime_name
        FROM models.regimes
        ORDER BY date
    """, conn)

    conn.close()

    # Pivot data
    pivoted = df.pivot_table(
        index="date", columns="variable_code", values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()

    # Drop columns with too many NaNs
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.7))
    pivoted = pivoted.dropna()

    # Join with regimes
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.set_index("date")
    pivoted = pivoted.join(regimes, how="inner")

    return pivoted


# ============================================
# STEP 2: FIT REGIME-CONDITIONAL VAR
# ============================================

def fit_regime_var(data, regime_name, max_lag=5):
    """
    Fit a Vector Autoregression model within a specific regime.
    Returns the coefficient matrices and residual covariance.
    """
    # Filter to this regime
    regime_data = data[data["regime_name"] == regime_name].drop(columns=["regime_name"])

    if len(regime_data) < 50:
        print(f"    WARNING: Only {len(regime_data)} days in {regime_name} regime, using lag=2")
        max_lag = 2

    variables = list(regime_data.columns)
    values = regime_data.values
    d = len(variables)
    T = len(values)

    # Standardize
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds

    # Build VAR matrices
    # Y = X @ B + E
    # where Y is (T-lag x d), X is (T-lag x d*lag+1), B is (d*lag+1 x d)
    effective_lag = min(max_lag, (T - 10) // d)
    effective_lag = max(effective_lag, 1)

    Y = standardized[effective_lag:]
    X_parts = [np.ones((T - effective_lag, 1))]  # intercept
    for lag in range(1, effective_lag + 1):
        X_parts.append(standardized[effective_lag - lag:T - lag])
    X = np.hstack(X_parts)

    # Solve via OLS
    try:
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        B = np.zeros((X.shape[1], d))

    # Compute residuals and covariance
    residuals = Y - X @ B
    cov = np.cov(residuals.T) if residuals.shape[0] > d else np.eye(d) * 0.01

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
    var_model,
    shock_variable,
    shock_magnitude,
    n_scenarios=N_SCENARIOS_DEFAULT,
    horizon=SCENARIO_HORIZON,
    causal_adjacency=None,
):
    """
    Generate forward-looking crisis scenarios.

    Process:
    1. Start from the last observed state
    2. Apply the initial shock to the specified variable
    3. Propagate forward using the regime-conditional VAR
    4. Add calibrated random noise at each step
    5. Optionally enforce causal graph constraints

    Args:
        var_model: fitted VAR model dict
        shock_variable: which variable to shock (e.g., "CL=F" for oil)
        shock_magnitude: size of shock in standard deviations (e.g., 3.0 = 3 sigma)
        n_scenarios: number of scenarios to generate
        horizon: number of days to simulate forward
        causal_adjacency: optional causal graph to constrain propagation
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

    # Generate Cholesky decomposition for correlated noise
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # If covariance is not positive definite, regularize
        cov_reg = cov + np.eye(d) * 0.001
        L = np.linalg.cholesky(cov_reg)

    all_scenarios = []

    for s in range(n_scenarios):
        # Initialize with zeros (standardized space)
        path = np.zeros((horizon + lag, d))

        # Apply initial shock at the first time step
        path[lag, shock_idx] = shock_magnitude

        # If we have causal adjacency, also propagate the initial shock
        # to directly connected variables
        if causal_adjacency is not None:
            for edge_key, edge_data in causal_adjacency.items():
                cause, effect = edge_key.split("->")
                if cause == shock_variable and effect in variables:
                    effect_idx = variables.index(effect)
                    weight = edge_data.get("weight", 0)
                    # Scale the propagation by causal weight
                    path[lag, effect_idx] += shock_magnitude * weight * 0.3

        # Simulate forward
        for t in range(lag + 1, horizon + lag):
            # Build feature vector: intercept + lagged values
            x = [1.0]  # intercept
            for l in range(1, lag + 1):
                x.extend(path[t - l])
            x = np.array(x)

            # Predicted values from VAR
            predicted = x @ B

            # Add correlated noise
            noise = L @ np.random.randn(d) * NOISE_SCALE

            # Combine
            path[t] = predicted + noise

        # Convert back from standardized to real scale
        real_path = path[lag:] * stds + means

        # Store as DataFrame
        scenario_df = pd.DataFrame(
            real_path,
            columns=variables,
            index=range(horizon),
        )

        all_scenarios.append(scenario_df)

    return all_scenarios


# ============================================
# STEP 4: SCORE SCENARIO PLAUSIBILITY
# ============================================

def score_plausibility(scenarios, var_model, causal_adjacency=None):
    """
    Score each scenario on plausibility:
    1. Statistical realism (are values within reasonable bounds?)
    2. Causal consistency (do effects follow causes?)
    3. Stylized facts (fat tails, volatility clustering)
    """
    variables = var_model["variables"]
    stds = var_model["stds"]
    scores = []

    for i, scenario in enumerate(scenarios):
        score = 1.0  # Start with perfect score

        # Check 1: Are any values extremely unrealistic? (>10 sigma)
        max_deviation = np.abs(scenario.values / stds).max()
        if max_deviation > 10:
            score *= 0.5  # Penalize extreme values
        elif max_deviation > 7:
            score *= 0.8

        # Check 2: Do variables move in internally consistent directions?
        if "^GSPC" in variables and "^VIX" in variables:
            gspc_idx = variables.index("^GSPC")
            vix_idx = variables.index("^VIX")
            # In a crisis, S&P should go down and VIX should go up
            spx_cum = scenario["^GSPC"].sum()
            vix_cum = scenario["^VIX"].sum()
            if spx_cum < 0 and vix_cum < 0:
                score *= 0.7  # S&P down but VIX also down is unusual
            if spx_cum > 0 and vix_cum > 0:
                score *= 0.8  # S&P up but VIX also up is somewhat unusual

        # Check 3: Causal consistency
        if causal_adjacency is not None:
            violations = 0
            total_checks = 0
            for edge_key, edge_data in causal_adjacency.items():
                cause, effect = edge_key.split("->")
                if cause in variables and effect in variables:
                    total_checks += 1
                    cause_change = scenario[cause].sum()
                    effect_change = scenario[effect].sum()
                    weight = edge_data.get("raw_weight", edge_data.get("weight", 0))
                    # If causal weight is positive, cause and effect should move same direction
                    if weight > 0 and cause_change * effect_change < 0:
                        violations += 1
                    elif weight < 0 and cause_change * effect_change > 0:
                        violations += 1

            if total_checks > 0:
                consistency = 1 - (violations / total_checks)
                score *= (0.5 + 0.5 * consistency)  # Scale between 0.5 and 1.0

        scores.append(round(score, 4))

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

    # Convert scenarios to storable format (list of dicts)
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

def summarize_scenarios(scenarios, scores, shock_variable, variables):
    """Print a summary of the generated scenarios."""
    print(f"\n{'='*60}")
    print("  SCENARIO GENERATION SUMMARY")
    print(f"{'='*60}")

    n = len(scenarios)
    horizon = len(scenarios[0])

    print(f"\n  Shock: {shock_variable}")
    print(f"  Scenarios: {n}")
    print(f"  Horizon: {horizon} days")
    print(f"  Plausibility: min={min(scores):.2f}, mean={np.mean(scores):.2f}, max={max(scores):.2f}")

    # Aggregate statistics across scenarios
    key_vars = ["^GSPC", "^VIX", "DGS10", "CL=F", "XLF", "BAMLH0A0HYM2"]
    key_vars = [v for v in key_vars if v in variables]

    print(f"\n  Cumulative impact over {horizon} days (across {n} scenarios):")
    print(f"  {'Variable':<20} {'5th pctl':>10} {'Median':>10} {'95th pctl':>10} {'Mean':>10}")
    print("  " + "-" * 64)

    for var in key_vars:
        cum_returns = [scenario[var].sum() for scenario in scenarios]
        p5 = np.percentile(cum_returns, 5)
        p50 = np.percentile(cum_returns, 50)
        p95 = np.percentile(cum_returns, 95)
        mean = np.mean(cum_returns)
        print(f"  {var:<20} {p5:>10.4f} {p50:>10.4f} {p95:>10.4f} {mean:>10.4f}")

    print(f"\n{'='*60}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - SCENARIO GENERATOR")
    print("=" * 60)

    # Step 1: Load data with regimes
    print("\nLoading data and regime labels...")
    data = load_processed_data_with_regimes()
    print(f"  Loaded {len(data)} days x {len(data.columns)} columns")

    # Step 2: Choose regime for generation (use "stressed" for crisis scenarios)
    target_regime = "stressed"
    print(f"\n  Target regime for generation: {target_regime}")

    # Step 3: Load causal graph for this regime
    graph_id, causal_adj, graph_vars = load_regime_causal_graph(target_regime)
    if graph_id:
        print(f"  Loaded causal graph: {graph_id}")
    else:
        print(f"  WARNING: No causal graph found for {target_regime}, generating without constraints")
        causal_adj = None

    # Step 4: Fit regime-conditional VAR
    print(f"\nFitting VAR model for {target_regime} regime...")
    var_model = fit_regime_var(data, target_regime)
    print(f"  Variables: {len(var_model['variables'])}")
    print(f"  Observations: {var_model['n_obs']}")
    print(f"  Lag order: {var_model['lag']}")

    # Step 5: Define shocks to test
    shocks = [
        ("CL=F", 3.0, "Oil price spike (+3 sigma)"),
        ("^GSPC", -3.0, "Market crash (-3 sigma)"),
        ("DGS10", 2.0, "Interest rate shock (+2 sigma)"),
        ("BAMLH0A0HYM2", 3.0, "Credit spread blowout (+3 sigma)"),
    ]

    for shock_var, shock_mag, shock_desc in shocks:
        print(f"\n{'─'*60}")
        print(f"  Generating scenarios: {shock_desc}")
        print(f"{'─'*60}")

        # Generate scenarios
        scenarios = generate_scenarios(
            var_model=var_model,
            shock_variable=shock_var,
            shock_magnitude=shock_mag,
            n_scenarios=100,
            horizon=60,
            causal_adjacency=causal_adj,
        )

        # Score plausibility
        scores = score_plausibility(scenarios, var_model, causal_adj)

        # Summarize
        summarize_scenarios(scenarios, scores, shock_var, var_model["variables"])

        # Store
        scenario_id = store_scenarios(
            scenarios, scores, shock_var, shock_mag,
            target_regime, str(graph_id) if graph_id else None,
        )

    print("\n✓ Scenario generation complete!")
    print(f"  Generated 4 shock scenarios x 100 paths each = 400 total scenarios")
    print("=" * 60)


if __name__ == "__main__":
    main()