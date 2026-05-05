"""
Calibrate Student-t Degrees-of-Freedom from VAR Residuals
===========================================================
Fits Student-t distributions to VAR residuals on pre-2020 training data
to obtain DEFENSIBLE, DATA-DRIVEN df values for the scenario generator.

This replaces ad-hoc hyperparameter choices with empirical estimates
derived from actual residual distributions.

Strict methodology:
  - Training window: 2005-01-01 to 2019-12-31 (pre-COVID, pre-SVB)
  - Fit regime-VAR separately on calm/normal vs elevated/stressed/high_stress/crisis
  - Extract residuals per variable
  - Fit Student-t via MLE (scipy.stats.t.fit)
  - Aggregate:
      df_normal  = median df across variables on calm+normal residuals
      df_crisis  = median df across variables on stress+ residuals
      mid_df     = geometric mean of the two
  - Report KS test p-values vs Gaussian to justify choosing Student-t

Output:
  - Printed summary with final df values to paste into all_paper_experiments.py
  - Per-variable results stored in models.residual_distributions

Usage:
  python calibrate_df_from_residuals.py
"""

import os
import sys
import json
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# CONFIGURATION (matches all_paper_experiments.py)
# ============================================================

TRAIN_START = "2005-01-01"
TRAIN_END = "2019-12-31"  # Strict holdout: excludes COVID and SVB

CALM_REGIMES = {"calm", "normal"}
STRESS_REGIMES = {"elevated", "stressed", "high_stress", "crisis"}

CORE_VARS = [
    "^GSPC", "^VIX", "^NDX", "^RUT", "DGS10", "DGS2", "T10Y2Y",
    "CL=F", "GC=F", "BAMLH0A0HYM2",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLU",
    "TLT", "LQD", "HYG", "EEM",
]

VAR_LAG = 5
RIDGE = 0.01
MIN_RESIDUALS_FOR_FIT = 100  # Need at least this many residual days per variable


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


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
    pivoted = pivoted.sort_index().dropna(axis=1, thresh=int(len(pivoted) * 0.7)).dropna()
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
    return df.drop_duplicates(subset=["date"]).set_index("date")["regime_name"].rename("regime_name")


def store_results(payload):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models.residual_distributions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            calibration_name VARCHAR(200),
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cursor.execute(
        "INSERT INTO models.residual_distributions (id, calibration_name, results) VALUES (%s, %s, %s)",
        (str(uuid.uuid4()), "Student-t df calibration (pre-2020)", Json(payload)),
    )
    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# VAR FIT + RESIDUAL EXTRACTION
# ============================================================

def ensure_pd(cov, d):
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 0:
        cov = cov + np.eye(d) * (abs(eigvals.min()) + 0.01)
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(d) * 0.1
    return cov


def fit_var_get_residuals(data, avail, lag=VAR_LAG):
    """
    Fit a VAR(lag) via ridge OLS and return standardized residuals per variable.
    Returns: dict {variable_code: np.array of residuals in standardized units}
    """
    values = data[avail].values
    d = len(avail)
    T = len(values)

    if T < lag + d + 10:
        return {}

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    std_data = (values - means) / stds

    effective_lag = min(lag, max(1, (T - 10) // d))
    Y = std_data[effective_lag:]
    X_parts = [np.ones((T - effective_lag, 1))]
    for l in range(1, effective_lag + 1):
        X_parts.append(std_data[effective_lag - l:T - l])
    X = np.hstack(X_parts)

    B = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ Y)
    residuals_std = Y - X @ B  # shape (T - lag, d), each col standardized-unit residual

    return {var: residuals_std[:, i] for i, var in enumerate(avail)}


# ============================================================
# STUDENT-T FITTING + GOODNESS OF FIT
# ============================================================

def fit_student_t(residuals):
    """
    Fit Student-t via MLE. Returns (df, loc, scale, ks_p_t, ks_p_normal).
    KS p-values: higher = better fit; p > 0.05 = can't reject the distribution.
    """
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < MIN_RESIDUALS_FOR_FIT:
        return None

    # Fit Student-t
    try:
        df_fit, loc_fit, scale_fit = stats.t.fit(residuals)
    except Exception:
        return None

    # Clamp df to sensible range; MLE can return huge values on Gaussian-ish data
    df_fit = float(np.clip(df_fit, 2.1, 50.0))

    # KS test vs fitted Student-t
    try:
        _, ks_p_t = stats.kstest(residuals, lambda x: stats.t.cdf(x, df_fit, loc_fit, scale_fit))
    except Exception:
        ks_p_t = np.nan

    # KS test vs Gaussian with same mean/std
    mu, sigma = np.mean(residuals), np.std(residuals)
    try:
        _, ks_p_normal = stats.kstest(residuals, lambda x: stats.norm.cdf(x, mu, sigma))
    except Exception:
        ks_p_normal = np.nan

    return {
        "df": df_fit,
        "loc": float(loc_fit),
        "scale": float(scale_fit),
        "n": len(residuals),
        "ks_p_student_t": float(ks_p_t) if np.isfinite(ks_p_t) else None,
        "ks_p_gaussian": float(ks_p_normal) if np.isfinite(ks_p_normal) else None,
        "kurtosis": float(stats.kurtosis(residuals)),
    }


# ============================================================
# MAIN CALIBRATION
# ============================================================

def calibrate():
    print("=" * 90)
    print("  STUDENT-T df CALIBRATION FROM VAR RESIDUALS")
    print(f"  Training window: {TRAIN_START} to {TRAIN_END} (strict pre-COVID holdout)")
    print("=" * 90)

    all_data = load_all_data()
    regime_series = load_regime_series()
    data = all_data.join(regime_series, how="left")

    # Restrict to training window
    mask = (data.index >= pd.to_datetime(TRAIN_START)) & (data.index <= pd.to_datetime(TRAIN_END))
    data = data[mask]
    print(f"  Total training days: {len(data)}")

    if "regime_name" not in data.columns or data["regime_name"].isna().all():
        print("  ERROR: No regime labels found in training window")
        sys.exit(1)

    # Split by regime
    calm_data = data[data["regime_name"].isin(CALM_REGIMES)].drop(columns=["regime_name"])
    stress_data = data[data["regime_name"].isin(STRESS_REGIMES)].drop(columns=["regime_name"])
    print(f"  Calm+Normal days:          {len(calm_data)}")
    print(f"  Stress+Crisis days:        {len(stress_data)}")

    avail = [v for v in CORE_VARS if v in calm_data.columns and v in stress_data.columns]
    print(f"  Variables for fitting:     {len(avail)} of {len(CORE_VARS)}")

    if len(calm_data) < 500 or len(stress_data) < 200:
        print("  WARNING: Insufficient data in one regime; fits may be unreliable")

    # ---------- CALM REGIME ----------
    print("\n" + "=" * 90)
    print("  FITTING VAR ON CALM+NORMAL DATA")
    print("=" * 90)
    calm_residuals = fit_var_get_residuals(calm_data, avail)
    print(f"  VAR fit: {len(calm_residuals)} variables, "
          f"{len(next(iter(calm_residuals.values()))) if calm_residuals else 0} residual days each")

    # ---------- STRESS REGIME ----------
    print("\n" + "=" * 90)
    print("  FITTING VAR ON STRESS+CRISIS DATA")
    print("=" * 90)
    stress_residuals = fit_var_get_residuals(stress_data, avail)
    print(f"  VAR fit: {len(stress_residuals)} variables, "
          f"{len(next(iter(stress_residuals.values()))) if stress_residuals else 0} residual days each")

    # ---------- STUDENT-T FIT PER VARIABLE ----------
    def fit_all(resid_dict, label):
        print(f"\n  Per-variable Student-t fit on {label} residuals:")
        print(f"  {'Variable':<18} {'df':>7} {'n':>6} {'kurt':>7} {'KS_t':>8} {'KS_N':>8} {'t_better':>10}")
        print(f"  {'-'*75}")
        per_var = {}
        for var in avail:
            if var not in resid_dict:
                continue
            fit = fit_student_t(resid_dict[var])
            if fit is None:
                continue
            t_better = (fit["ks_p_student_t"] or 0) > (fit["ks_p_gaussian"] or 0)
            per_var[var] = fit
            ks_t = f"{fit['ks_p_student_t']:.3f}" if fit["ks_p_student_t"] is not None else "n/a"
            ks_n = f"{fit['ks_p_gaussian']:.3f}" if fit["ks_p_gaussian"] is not None else "n/a"
            print(f"  {var:<18} {fit['df']:>7.2f} {fit['n']:>6} {fit['kurtosis']:>7.2f} "
                  f"{ks_t:>8} {ks_n:>8} {'YES' if t_better else 'no':>10}")
        return per_var

    calm_fits = fit_all(calm_residuals, "CALM+NORMAL")
    stress_fits = fit_all(stress_residuals, "STRESS+CRISIS")

    # ---------- AGGREGATE ----------
    calm_dfs = [v["df"] for v in calm_fits.values()]
    stress_dfs = [v["df"] for v in stress_fits.values()]

    df_normal = float(np.median(calm_dfs)) if calm_dfs else None
    df_crisis = float(np.median(stress_dfs)) if stress_dfs else None
    mid_df = float(np.sqrt(df_normal * df_crisis)) if (df_normal and df_crisis) else None

    # Student-t wins ratio
    t_wins_calm = sum(1 for v in calm_fits.values()
                      if (v["ks_p_student_t"] or 0) > (v["ks_p_gaussian"] or 0))
    t_wins_stress = sum(1 for v in stress_fits.values()
                        if (v["ks_p_student_t"] or 0) > (v["ks_p_gaussian"] or 0))

    # ---------- REPORT ----------
    print("\n" + "=" * 90)
    print("  AGGREGATED df VALUES (FOR PAPER)")
    print("=" * 90)

    if calm_dfs:
        print(f"\n  CALM+NORMAL regime:")
        print(f"    n_variables:     {len(calm_dfs)}")
        print(f"    df range:        [{min(calm_dfs):.2f}, {max(calm_dfs):.2f}]")
        print(f"    df median:       {df_normal:.2f}   <-- df_normal")
        print(f"    df mean:         {np.mean(calm_dfs):.2f}")
        print(f"    Student-t wins:  {t_wins_calm}/{len(calm_fits)} variables")

    if stress_dfs:
        print(f"\n  STRESS+CRISIS regime:")
        print(f"    n_variables:     {len(stress_dfs)}")
        print(f"    df range:        [{min(stress_dfs):.2f}, {max(stress_dfs):.2f}]")
        print(f"    df median:       {df_crisis:.2f}   <-- df_crisis")
        print(f"    df mean:         {np.mean(stress_dfs):.2f}")
        print(f"    Student-t wins:  {t_wins_stress}/{len(stress_fits)} variables")

    print(f"\n  MID regime (geometric mean of calm and stress):")
    print(f"    mid_df:          {mid_df:.2f}" if mid_df else "    mid_df:          n/a")

    print("\n" + "=" * 90)
    print("  PASTE INTO all_paper_experiments.py:")
    print("=" * 90)
    if df_normal and df_crisis and mid_df:
        print(f"""
  PAPER_DF_NORMAL = {df_normal:.2f}   # median df on calm+normal VAR residuals (pre-2020)
  PAPER_DF_CRISIS = {df_crisis:.2f}   # median df on stress+crisis VAR residuals (pre-2020)
  PAPER_MID_DF    = {mid_df:.2f}   # geometric mean of calm and stress df
""")

    # ---------- STORE ----------
    payload = {
        "training_window": {"start": TRAIN_START, "end": TRAIN_END},
        "n_training_days": int(len(data)),
        "n_calm_days": int(len(calm_data)),
        "n_stress_days": int(len(stress_data)),
        "n_variables": int(len(avail)),
        "df_normal": df_normal,
        "df_crisis": df_crisis,
        "mid_df": mid_df,
        "student_t_wins_calm": f"{t_wins_calm}/{len(calm_fits)}",
        "student_t_wins_stress": f"{t_wins_stress}/{len(stress_fits)}",
        "per_variable_calm": calm_fits,
        "per_variable_stress": stress_fits,
    }

    try:
        store_results(payload)
        print(f"\n  Results stored in models.residual_distributions")
    except Exception as e:
        print(f"\n  WARNING: Could not store results: {e}")

    print("=" * 90)
    return payload


if __name__ == "__main__":
    calibrate()