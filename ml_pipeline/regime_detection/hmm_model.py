"""
Hidden Markov Model Regime Detection
======================================
Detects market regimes (calm, stressed, crisis) from observable
financial indicators using a Gaussian Hidden Markov Model.

The "hidden" states are the regimes — we can't directly observe
whether we're in a crisis. But we CAN observe things like:
- VIX (fear index)
- Credit spreads (risk appetite)
- Yield curve slope (recession signal)
- Equity volatility (market turbulence)

The HMM learns to map these observations to hidden regime states.

After regime detection, we re-run causal discovery WITHIN each
regime to see how economic relationships change across states.
This is our key innovation that nobody else has built.
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

# Features to use for regime detection
# These are the most informative indicators of market state
REGIME_FEATURES = {
    "^VIX": "VIX (Fear Index)",
    "BAMLH0A0HYM2": "High Yield Credit Spread",
    "T10Y2Y": "Yield Curve Spread",
    "^GSPC": "S&P 500 Returns",
    "^MOVE": "MOVE Index (Bond Volatility)",
    "TEDRATE": "TED Spread (Interbank Stress)",
    "STLFSI4": "St. Louis Fed Financial Stress Index",
}

# Number of regimes to test
MIN_STATES = 2
MAX_STATES = 6

# Regime labels (assigned after training based on characteristics)
REGIME_NAMES = {
    0: "unknown",
    1: "unknown",
    2: "unknown",
    3: "unknown",
    4: "unknown",
}

# Known crisis periods for validation
KNOWN_CRISES = [
    ("2007-10-01", "2009-03-31", "Global Financial Crisis"),
    ("2010-04-15", "2010-07-15", "Flash Crash / Euro Crisis"),
    ("2011-07-01", "2011-10-31", "US Debt Downgrade / Euro Debt"),
    ("2015-08-01", "2016-02-28", "China Devaluation / Oil Crash"),
    ("2018-01-26", "2018-04-06", "Volmageddon"),
    ("2018-09-20", "2018-12-31", "Fed Tightening Selloff"),
    ("2020-02-19", "2020-03-31", "COVID Crash"),
    ("2022-01-01", "2022-10-31", "Rate Hike Selloff"),
    ("2011-07-22", "2011-08-10", "US Debt Ceiling Crisis"),
    ("2016-06-23", "2016-06-27", "Brexit Shock"),
    ("2020-03-06", "2020-04-21", "Oil Price War (COVID + OPEC)"),
    ("2015-06-12", "2015-08-25", "China Stock Market Crash"),
    ("2018-10-03", "2018-10-29", "October 2018 Correction"),
    ("2020-09-02", "2020-09-23", "September 2020 Tech Selloff"),
    ("2023-03-08", "2023-03-20", "SVB Banking Crisis"),
]


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
# STEP 1: LOAD REGIME FEATURES
# ============================================

def load_regime_features():
    """Load the indicators used for regime detection."""
    print("Loading regime detection features...")

    conn = get_db_connection()

    # Load raw values for VIX and credit spreads (levels, not returns)
    # Load transformed values for S&P returns and yield curve changes
    dfs = []

    for var_code, name in REGIME_FEATURES.items():
        if var_code in ["^VIX", "BAMLH0A0HYM2", "STLFSI4"]:
            # Use RAW values for level-based indicators
            query = f"""
                SELECT date, raw_value as value
                FROM processed.time_series_data
                WHERE variable_code = '{var_code}'
                  AND raw_value IS NOT NULL
                ORDER BY date
            """
        elif var_code == "T10Y2Y":
            # Use raw value for yield curve (it's already a spread)
            query = f"""
                SELECT date, raw_value as value
                FROM processed.time_series_data
                WHERE variable_code = '{var_code}'
                  AND raw_value IS NOT NULL
                ORDER BY date
            """
        else:
            # Use transformed value for returns
            query = f"""
                SELECT date, transformed_value as value
                FROM processed.time_series_data
                WHERE variable_code = '{var_code}'
                  AND transformed_value IS NOT NULL
                ORDER BY date
            """

        df = pd.read_sql(query, conn)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"value": var_code})
        dfs.append(df)

    conn.close()

    # Merge all features on date
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, how="inner")

    # Drop any remaining NaN
    combined = combined.dropna()

    # Also compute rolling 21-day realized volatility of S&P for extra signal
    if "^GSPC" in combined.columns:
        combined["SPX_RVOL_21"] = combined["^GSPC"].rolling(21).std() * np.sqrt(252)
        combined = combined.dropna()

    print(f"  Loaded {len(combined)} days x {len(combined.columns)} features")
    print(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"  Features: {list(combined.columns)}")

    return combined


# ============================================
# STEP 2: SELECT OPTIMAL NUMBER OF STATES
# ============================================

def select_n_states(data):
    """
    Use BIC (Bayesian Information Criterion) to find the optimal
    number of hidden states for the HMM.

    Lower BIC = better model (balances fit vs complexity).
    """
    print(f"\nSelecting optimal number of states ({MIN_STATES}-{MAX_STATES})...")

    # Standardize features for HMM
    means = data.mean()
    stds = data.std()
    standardized = ((data - means) / stds).values

    best_bic = np.inf
    best_n = MIN_STATES
    results = []

    for n in range(MIN_STATES, MAX_STATES + 1):
        try:
            model = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            model.fit(standardized)
            score = model.score(standardized)  # log-likelihood

            # Compute BIC: -2 * log_likelihood + k * log(n_samples)
            n_params = n * (n - 1) + n * data.shape[1] + n * data.shape[1] * (data.shape[1] + 1) / 2
            bic = -2 * score + n_params * np.log(len(standardized))

            results.append({"n_states": n, "log_likelihood": score, "bic": bic})
            print(f"  {n} states: BIC = {bic:.1f}, Log-Likelihood = {score:.1f}")

            if bic < best_bic:
                best_bic = bic
                best_n = n

        except Exception as e:
            print(f"  {n} states: FAILED - {e}")

    print(f"\n  Optimal number of states: {best_n} (lowest BIC)")
    return best_n, results


# ============================================
# STEP 3: TRAIN THE HMM
# ============================================

def train_hmm(data, n_states):
    """
    Train the Gaussian HMM with the optimal number of states.

    Returns the fitted model and regime predictions.
    """
    print(f"\nTraining HMM with {n_states} states...")

    # Standardize
    means = data.mean()
    stds = data.std()
    standardized = ((data - means) / stds).values

    # Train HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=42,
        tol=0.001,
    )
    model.fit(standardized)

    # Predict regimes
    regime_labels = model.predict(standardized)
    regime_probs = model.predict_proba(standardized)

    # Get transition matrix
    transition_matrix = model.transmat_

    print(f"  Model converged: {model.monitor_.converged}")
    print(f"  Final log-likelihood: {model.score(standardized):.1f}")

    return model, regime_labels, regime_probs, transition_matrix, means, stds


# ============================================
# STEP 4: LABEL AND CHARACTERIZE REGIMES
# ============================================

def characterize_regimes(data, regime_labels, n_states):
    """
    Analyze each regime's characteristics to assign meaningful names.

    Strategy: rank regimes by average VIX level
    - Lowest VIX = calm
    - Middle VIX = stressed / transition
    - Highest VIX = crisis
    """
    print("\nCharacterizing regimes...\n")

    data_with_regimes = data.copy()
    data_with_regimes["regime"] = regime_labels

    regime_stats = {}

    for regime in range(n_states):
        mask = data_with_regimes["regime"] == regime
        regime_data = data_with_regimes[mask]

        regime_stats[regime] = {
            "count": int(mask.sum()),
            "pct": float(mask.sum() / len(data_with_regimes) * 100),
            "vix_mean": float(regime_data["^VIX"].mean()) if "^VIX" in regime_data.columns else 0,
            "vix_median": float(regime_data["^VIX"].median()) if "^VIX" in regime_data.columns else 0,
            "spx_return_mean": float(regime_data["^GSPC"].mean()) if "^GSPC" in regime_data.columns else 0,
            "hy_spread_mean": float(regime_data["BAMLH0A0HYM2"].mean()) if "BAMLH0A0HYM2" in regime_data.columns else 0,
            "yield_curve_mean": float(regime_data["T10Y2Y"].mean()) if "T10Y2Y" in regime_data.columns else 0,
        }

    # Sort regimes by VIX mean (lowest = calm, highest = crisis)
    sorted_regimes = sorted(regime_stats.keys(), key=lambda r: regime_stats[r]["vix_mean"])

    # Assign names based on VIX ranking
    if n_states == 2:
        names = ["calm", "crisis"]
    elif n_states == 3:
        names = ["calm", "stressed", "crisis"]
    elif n_states == 4:
        names = ["calm", "normal", "stressed", "crisis"]
    elif n_states == 5:
        names = ["calm", "normal", "elevated", "stressed", "crisis"]
    elif n_states == 6:
        names = ["calm", "normal", "elevated", "stressed", "high_stress", "crisis"]
    elif n_states == 7:
        names = ["calm", "low_normal", "normal", "elevated", "stressed", "high_stress", "crisis"]
    elif n_states == 8:
        names = ["calm", "low_normal", "normal", "elevated", "pre_stress", "stressed", "high_stress", "crisis"]
    else:
        names = [f"regime_{i}" for i in range(n_states)]

    regime_name_map = {}
    for i, regime_id in enumerate(sorted_regimes):
        regime_name_map[regime_id] = names[i]

    # Print regime characteristics
    print(f"  {'Regime':<12} {'Name':<12} {'Days':>7} {'%':>7} {'VIX Mean':>10} {'SPX Ret':>10} {'HY Spread':>10} {'Curve':>10}")
    print("  " + "-" * 82)

    for regime_id in sorted_regimes:
        s = regime_stats[regime_id]
        name = regime_name_map[regime_id]
        print(f"  {regime_id:<12} {name:<12} {s['count']:>7} {s['pct']:>6.1f}% "
              f"{s['vix_mean']:>10.2f} {s['spx_return_mean']:>10.6f} "
              f"{s['hy_spread_mean']:>10.2f} {s['yield_curve_mean']:>10.2f}")

    return regime_stats, regime_name_map


# ============================================
# STEP 5: VALIDATE AGAINST KNOWN CRISES
# ============================================

def validate_regimes(data, regime_labels, regime_name_map):
    """
    Check if the detected regimes align with known historical crises.
    """
    print("\nValidating against known crisis periods...\n")

    data_with_regimes = data.copy()
    data_with_regimes["regime"] = regime_labels
    data_with_regimes["regime_name"] = data_with_regimes["regime"].map(regime_name_map)

    # Find the crisis regime label(s)
    crisis_names = ["crisis", "stressed", "high_stress", "pre_stress"]

    correct = 0
    total = len(KNOWN_CRISES)

    print(f"  {'Crisis Period':<40} {'Detected As':<20} {'Match':>6}")
    print("  " + "-" * 70)

    for start, end, name in KNOWN_CRISES:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Get regimes during this crisis period
        mask = (data_with_regimes.index >= start_date) & (data_with_regimes.index <= end_date)
        crisis_period = data_with_regimes[mask]

        if len(crisis_period) == 0:
            print(f"  {name:<40} {'NO DATA':<20} {'N/A':>6}")
            continue

        # What was the dominant regime during this period?
        dominant_regime = crisis_period["regime_name"].mode().iloc[0]

        # Count as correct if dominant regime is crisis or stressed
        is_correct = dominant_regime in crisis_names
        if is_correct:
            correct += 1

        status = "YES" if is_correct else "NO"
        print(f"  {name:<40} {dominant_regime:<20} {status:>6}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  Crisis detection accuracy: {correct}/{total} = {accuracy:.0f}%")

    return accuracy


# ============================================
# STEP 6: COMPUTE TRANSITION MATRIX
# ============================================

def analyze_transitions(transition_matrix, regime_name_map, n_states):
    """Analyze and display the regime transition probabilities."""
    print("\nRegime transition matrix:\n")

    sorted_regimes = sorted(regime_name_map.keys(), key=lambda r: regime_name_map[r])

    # Header
    header = "  From \\ To    "
    for r in sorted_regimes:
        header += f"{regime_name_map[r]:>12}"
    print(header)
    print("  " + "-" * (14 + 12 * n_states))

    for i in sorted_regimes:
        row = f"  {regime_name_map[i]:<14}"
        for j in sorted_regimes:
            prob = transition_matrix[i, j]
            row += f"{prob:>11.1%} "
        print(row)

    # Expected duration in each regime
    print("\n  Expected regime durations:")
    for r in sorted_regimes:
        # Expected duration = 1 / (1 - P(stay in same state))
        stay_prob = transition_matrix[r, r]
        if stay_prob < 1:
            expected_days = 1 / (1 - stay_prob)
            print(f"    {regime_name_map[r]:<12}: ~{expected_days:.0f} trading days ({expected_days/252:.1f} years)")
        else:
            print(f"    {regime_name_map[r]:<12}: infinite (absorbing state)")


# ============================================
# STEP 7: STORE REGIMES IN DATABASE
# ============================================

def store_regimes(data, regime_labels, regime_probs, regime_name_map, transition_matrix):
    """Store regime classifications in the database."""
    print("\nStoring regimes in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear existing regime data
    cursor.execute("DELETE FROM models.regimes")

    # Insert one row at a time to avoid template issues
    insert_query = """
        INSERT INTO models.regimes
            (id, date, regime_label, regime_name, probability, transition_probs, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    count = 0
    for i, date in enumerate(data.index):
        regime_id = int(regime_labels[i])
        cursor.execute(insert_query, (
            str(uuid.uuid4()),
            date.date(),
            regime_id,
            regime_name_map[regime_id],
            float(regime_probs[i, regime_id]),
            json.dumps({str(k): float(v) for k, v in enumerate(transition_matrix[regime_id])}),
            "v1.0",
        ))
        count += 1

    conn.commit()
    print(f"  Stored {count} regime classifications")

    cursor.close()
    conn.close()


# ============================================
# STEP 8: IDENTIFY CURRENT REGIME
# ============================================

def identify_current_regime(data, regime_labels, regime_name_map, regime_probs):
    """Report the current (most recent) market regime."""
    print("\n" + "=" * 50)
    print("  CURRENT MARKET REGIME")
    print("=" * 50)

    latest_idx = -1
    latest_date = data.index[latest_idx]
    latest_regime = regime_labels[latest_idx]
    latest_name = regime_name_map[latest_regime]
    latest_prob = regime_probs[latest_idx]

    print(f"\n  Date: {latest_date.date()}")
    print(f"  Regime: {latest_name.upper()}")
    print(f"  Confidence: {latest_prob[latest_regime]:.1%}")
    print(f"\n  Probability breakdown:")
    for r in sorted(regime_name_map.keys()):
        bar = "█" * int(latest_prob[r] * 40)
        print(f"    {regime_name_map[r]:<12} {latest_prob[r]:>6.1%} {bar}")

    # How long have we been in this regime?
    streak = 0
    for i in range(len(regime_labels) - 1, -1, -1):
        if regime_labels[i] == latest_regime:
            streak += 1
        else:
            break

    print(f"\n  Current streak: {streak} trading days in {latest_name}")
    print("=" * 50)


# ============================================
# STEP 9: EXPORT FOR FRONTEND
# ============================================

def export_regime_data(data, regime_labels, regime_name_map, regime_probs, transition_matrix):
    """Export regime data as JSON for frontend visualization."""
    filepath = "regime_data.json"
    print(f"\nExporting regime data to {filepath}...")

    timeline = []
    for i, date in enumerate(data.index):
        timeline.append({
            "date": date.strftime("%Y-%m-%d"),
            "regime": int(regime_labels[i]),
            "regime_name": regime_name_map[int(regime_labels[i])],
            "probability": float(regime_probs[i, regime_labels[i]]),
        })

    regime_info = {}
    for r, name in regime_name_map.items():
        regime_info[str(r)] = {
            "name": name,
            "transition_probs": {str(k): float(v) for k, v in enumerate(transition_matrix[r])},
        }

    export_data = {
        "timeline": timeline,
        "regimes": regime_info,
        "n_states": len(regime_name_map),
        "created_at": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f)

    print(f"  Exported {len(timeline)} days of regime data")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - REGIME DETECTION ENGINE")
    print("=" * 60)

    # Step 1: Load features
    data = load_regime_features()

    # Step 2: Select optimal number of states
    n_states, bic_results = select_n_states(data)

    # Step 3: Train HMM
    model, regime_labels, regime_probs, transition_matrix, means, stds = train_hmm(data, n_states)

    # Step 4: Characterize regimes
    regime_stats, regime_name_map = characterize_regimes(data, regime_labels, n_states)

    # Step 5: Validate against known crises
    accuracy = validate_regimes(data, regime_labels, regime_name_map)

    # Step 6: Analyze transitions
    analyze_transitions(transition_matrix, regime_name_map, n_states)

    # Step 7: Store in database
    store_regimes(data, regime_labels, regime_probs, regime_name_map, transition_matrix)

    # Step 8: Current regime
    identify_current_regime(data, regime_labels, regime_name_map, regime_probs)

    # Step 9: Export for frontend
    export_regime_data(data, regime_labels, regime_name_map, regime_probs, transition_matrix)

    print("\n✓ Regime detection complete!")
    print(f"  {n_states} regimes detected")
    print(f"  Crisis detection accuracy: {accuracy:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()