"""
real_mode_loader.py
===================
Real-mode data loading for Phase 2B RL training.

This module bridges from the abstract `make_env(mode='real')` factory call
to the concrete production data pipeline. It loads:

    - Processed time-series data from `processed.time_series_data`
    - Regime labels from `models.regimes`
    - Joins them into the `regime_name`-augmented dataframe that
      `fit_regime_var()` expects
    - Fits the VAR ONCE (not per-env) and returns the fitted model
    - Loads the canonical pruned causal graph

Failure mode: if any production resource (DB, regime table, canonical
graph file) is unavailable, this raises a descriptive RuntimeError. There
is NO silent fallback to fast mode — if real mode is requested, real mode
must work.

This module imports lazily so the rest of generative_engine_rl can be
imported without psycopg2 / dotenv / pandas being available.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# HISTORICAL CRISIS DATES FOR WARM-START SEEDING
# ============================================================================

# 12 named crisis events used as fixed starting states for the beam-teacher
# warm start. Each entry is (event_name, ISO date). The loader finds the
# nearest available trading day in the DB for each date.
CRISIS_DATES: List[Tuple[str, str]] = [
    ("Lehman collapse",         "2008-09-15"),
    ("GFC first panic",         "2008-10-10"),
    ("GFC market bottom",       "2009-03-09"),
    ("US debt downgrade",       "2011-08-05"),
    ("China devaluation",       "2015-08-24"),
    ("Q4 2018 selloff",         "2018-12-24"),
    ("COVID first panic",       "2020-02-24"),
    ("COVID crash",             "2020-03-16"),
    ("COVID second wave fear",  "2020-09-03"),
    ("Rate fear onset",         "2022-01-19"),
    ("Rate shock peak",         "2022-06-13"),
    ("UK gilt crisis",          "2022-09-27"),
]


# ============================================================================
# PATH SETUP
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent  # .../ml_pipeline/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# LAZY IMPORTS
# ============================================================================

def _import_production_deps():
    """Import production deps inside a function so the module can be
    imported even when psycopg2 etc. are missing (fast mode users).
    """
    try:
        import pandas as pd
        import psycopg2  # noqa: F401
        from dotenv import load_dotenv
    except ImportError as e:
        raise RuntimeError(
            f"Real mode requires production dependencies that are missing: {e}. "
            f"Install with: pip install pandas psycopg2-binary python-dotenv"
        ) from e
    load_dotenv()
    return pd


def _get_db_connection():
    """Same connection helper as multi_backtest_v3.get_db()."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================================================
# DATA LOADERS — mirror the helpers in multi_backtest_v3
# ============================================================================

def _load_all_data(pd):
    """Pulls processed time-series data. Mirrors multi_backtest_v3.load_all_data."""
    try:
        conn = _get_db_connection()
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to production database: {e}. "
            f"Is your Postgres instance running on the configured host/port? "
            f"Check POSTGRES_HOST and POSTGRES_PORT environment variables."
        ) from e

    try:
        df = pd.read_sql("""
            SELECT date, variable_code, transformed_value
            FROM processed.time_series_data
            WHERE source != 'engineered'
            ORDER BY date
        """, conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "processed.time_series_data is empty. Has the data ingestion "
            "pipeline run yet? Try: python data_ingestion/data_processor.py"
        )

    pivoted = df.pivot_table(
        index="date", columns="variable_code", values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index()
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.7))
    pivoted = pivoted.dropna()

    if len(pivoted) < 100:
        raise RuntimeError(
            f"Loaded data has only {len(pivoted)} rows after cleaning — "
            f"expected at least several hundred. The DB may have only "
            f"partial data."
        )
    return pivoted


def _load_regime_series(pd):
    """Pulls regime labels. Mirrors multi_backtest_v3.load_regime_series."""
    conn = _get_db_connection()
    try:
        df = pd.read_sql("""
            SELECT date, regime_name
            FROM models.regimes
            ORDER BY date
        """, conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "models.regimes is empty. Has the HMM regime detection "
            "pipeline run yet? The RL agent needs regime labels for the "
            "causal-plausibility mask."
        )
    df["date"] = pd.to_datetime(df["date"])
    series = df.drop_duplicates(subset=["date"]).set_index("date")["regime_name"]
    series.name = "regime_name"
    return series


def _merge_regime_into_data(data, regime_series, pd):
    """Joins regime labels into the variable dataframe."""
    aligned = data.copy()
    aligned["regime_name"] = regime_series.reindex(aligned.index)
    # Forward-fill missing regime labels for short gaps; drop residuals
    aligned["regime_name"] = aligned["regime_name"].ffill().bfill()
    aligned = aligned.dropna(subset=["regime_name"])

    if "regime_name" not in aligned.columns:
        raise RuntimeError("regime_name column was lost during merge")

    return aligned


def _build_rl_state_support(data_with_regime, core_variables, pd) -> Dict[str, Any]:
    """Construct observation support for RL resets.

    Observation = regime one-hot + z-scored current values for the 25 core vars.
    The values are sampled from real processed-history rows in the requested
    regime so resets reflect genuine market states instead of a zero vector.
    """
    missing = [v for v in core_variables if v not in data_with_regime.columns]
    if missing:
        raise RuntimeError(
            "Real-mode state support is missing required core variables: "
            f"{missing}"
        )

    state_df = data_with_regime[core_variables + ["regime_name"]].dropna().copy()
    if len(state_df) < 100:
        raise RuntimeError(
            f"Real-mode state support has only {len(state_df)} clean rows; "
            "expected at least 100."
        )

    raw_core = state_df[core_variables].astype(np.float32)
    z_means = raw_core.mean(axis=0)
    z_stds = raw_core.std(axis=0, ddof=0).replace(0, 1.0)
    z_core = ((raw_core - z_means) / z_stds).clip(-5.0, 5.0)

    regimes = state_df["regime_name"].astype(str).str.lower().to_numpy()
    regime_names = sorted({r for r in regimes})
    regime_to_idx = {r: i for i, r in enumerate(regime_names)}
    one_hot = np.zeros((len(state_df), len(regime_names)), dtype=np.float32)
    for row_idx, regime_name in enumerate(regimes):
        one_hot[row_idx, regime_to_idx[regime_name]] = 1.0

    obs_matrix = np.hstack([one_hot, z_core.to_numpy(dtype=np.float32)])
    regime_to_rows = {
        regime_name: np.flatnonzero(regimes == regime_name).astype(np.int32)
        for regime_name in regime_names
    }

    return {
        "feature_names": (
            [f"regime::{r}" for r in regime_names]
            + [f"z::{v}" for v in core_variables]
        ),
        "regime_names": regime_names,
        "dates": state_df.index.to_numpy(),
        "regimes": regimes,
        "obs_matrix": obs_matrix.astype(np.float32),
        "raw_core_matrix": raw_core.to_numpy(dtype=np.float32),
        "core_variables": list(core_variables),
        "regime_to_rows": regime_to_rows,
        # Stored for crisis-seed construction (must use same normalization)
        "z_means": z_means.to_numpy(dtype=np.float32),
        "z_stds": z_stds.to_numpy(dtype=np.float32),
        "regime_to_idx": regime_to_idx,
    }


def _build_crisis_initial_states(
    data_with_regime: Any,
    state_support: Dict[str, Any],
    core_variables: List[str],
    pd: Any,
) -> List[Dict[str, Any]]:
    """Build one observation vector per CRISIS_DATES entry for warm-start seeding.

    Each vector uses the same z-scoring and regime one-hot encoding as the
    main obs_matrix so it can be dropped directly into env.reset() as a
    crisis_seed_idx starting state.

    Returns a list of dicts: {event_name, date, regime, obs_vector}.
    Dates not found in the data are silently skipped.
    """
    z_means = state_support["z_means"]
    z_stds = state_support["z_stds"]
    regime_to_idx = state_support["regime_to_idx"]
    n_regimes = len(regime_to_idx)

    source = data_with_regime[core_variables + ["regime_name"]].dropna()
    crisis_states: List[Dict[str, Any]] = []

    for event_name, date_str in CRISIS_DATES:
        target = pd.Timestamp(date_str)
        idx_arr = source.index.get_indexer([target], method="nearest")
        if len(idx_arr) == 0 or idx_arr[0] < 0:
            print(f"  [crisis_seeds] {event_name} ({date_str}): no nearby date, skipping")
            continue
        row = source.iloc[idx_arr[0]]
        nearest_date = source.index[idx_arr[0]]
        raw_vals = np.array([float(row[v]) for v in core_variables], dtype=np.float32)
        z_vals = ((raw_vals - z_means) / z_stds).clip(-5.0, 5.0)
        regime_str = str(row["regime_name"]).lower()
        regime_idx = regime_to_idx.get(regime_str, 0)
        one_hot = np.zeros(n_regimes, dtype=np.float32)
        one_hot[regime_idx] = 1.0
        obs_vector = np.concatenate([one_hot, z_vals])
        crisis_states.append({
            "event_name": event_name,
            "date": str(nearest_date)[:10],
            "regime": regime_str,
            "obs_vector": obs_vector,
        })
        print(f"  [crisis_seeds] {event_name}: {str(nearest_date)[:10]} regime={regime_str}")

    print(f"  [crisis_seeds] built {len(crisis_states)}/{len(CRISIS_DATES)} crisis states")
    return crisis_states


def _build_historical_bank(
    data: Any,
    variables: List[str],
    horizon: int,
    pd: Any,
) -> List[np.ndarray]:
    """Build a small reference bank of real historical event trajectories.

    Uses the canonical 11 paper events. Each reference path is the real
    transformed-data window starting at the event's start date and extending
    for `horizon` trading days.
    """
    try:
        from canonical_paper_numbers import EVENTS
    except Exception:
        return []

    bank: List[np.ndarray] = []
    cols_missing = [v for v in variables if v not in data.columns]
    if cols_missing:
        return []

    source = data[variables].copy().sort_index()
    for event in EVENTS:
        event_start = pd.Timestamp(event.start)
        window = source.loc[source.index >= event_start].head(horizon)
        if len(window) != horizon:
            continue
        if window.isna().any().any():
            continue
        bank.append(window.to_numpy(dtype=np.float32))
    return bank


# ============================================================================
# FITTED-MODEL CACHE
# ============================================================================
# fit_regime_var() is expensive; we run it once per process and reuse.
# This is safe across envs because:
#   - var_model is a dict of numpy arrays (read-only by convention)
#   - Different envs in the same process share state without issue
#   - SubprocVecEnv envs each get their own process and therefore
#     their own cache (no cross-process sharing required)

_FITTED_CACHE: Dict[str, Any] = {}


def _fit_or_load_var(data_with_regime, regime_name: str = "stressed",
                     train_regimes: Optional[list] = None,
                     max_lag: int = 2) -> Dict[str, Any]:
    """Fit the VAR for the given regime, or return cached fit."""
    cache_key = f"{regime_name}::{','.join(train_regimes or [])}::lag{max_lag}"
    if cache_key in _FITTED_CACHE:
        return _FITTED_CACHE[cache_key]

    from generative_engine.scenario_generator import fit_regime_var

    print(f"  [real_mode] Fitting VAR for regime={regime_name} "
          f"train_regimes={train_regimes} max_lag={max_lag}...")
    var_model = fit_regime_var(
        data=data_with_regime,
        regime_name=regime_name,
        max_lag=max_lag,
        train_regimes=train_regimes,
    )
    _FITTED_CACHE[cache_key] = var_model
    print(f"  [real_mode] VAR fitted: {len(var_model['variables'])} variables, "
          f"lag={var_model['lag']}, n_obs={var_model['n_obs']}")
    return var_model


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def load_real_var_model_and_graph(
    regime_name: str = "stressed",
    train_regimes: Optional[list] = None,
    max_lag: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, dict], Callable]:
    """Load the production VAR + causal graph + scenario_fn.

    Args:
        regime_name: which regime to fit the VAR for (default 'stressed' to
            match canonical CausalStress configuration)
        train_regimes: list of regime names to include in training data.
            Defaults to CANONICAL_TRAIN_REGIMES.
        max_lag: VAR lag. Default 2 matches scenario_generator's default for
            mid-size datasets.

    Returns:
        (var_model, causal_adjacency, scenario_fn) — three components needed
        by CausalStressEnv to run real-mode trajectories.

    Raises:
        RuntimeError: if any production resource is unavailable. Errors are
        descriptive and tell you exactly which dependency is missing.
    """
    pd = _import_production_deps()

    # ---- Load and merge data ----
    print("  [real_mode] Loading processed time-series data from DB...")
    data = _load_all_data(pd)
    print(f"  [real_mode] Loaded {len(data)} rows × {len(data.columns)} variables")

    print("  [real_mode] Loading regime labels from DB...")
    regime_series = _load_regime_series(pd)
    print(f"  [real_mode] Loaded {len(regime_series)} regime labels")

    data_with_regime = _merge_regime_into_data(data, regime_series, pd)

    # ---- Determine train regimes ----
    if train_regimes is None:
        try:
            from canonical_best_model import CANONICAL_TRAIN_REGIMES
            train_regimes = list(CANONICAL_TRAIN_REGIMES)
        except (ImportError, AttributeError):
            train_regimes = ["elevated", "stressed", "high_stress", "crisis"]

    # ---- Build RL state support before column-subsetting in fit_regime_var ----
    from generative_engine_rl.action_space_loader import load_spec
    spec = load_spec()
    rl_state_support = _build_rl_state_support(data_with_regime, spec.core_variables, pd)

    # ---- Build historical crisis initial states for warm-start seeding ----
    print("  [real_mode] Building historical crisis initial states...")
    crisis_seeds = _build_crisis_initial_states(
        data_with_regime, rl_state_support, list(spec.core_variables), pd
    )

    # ---- Fit VAR (cached) ----
    var_model = _fit_or_load_var(
        data_with_regime,
        regime_name=regime_name,
        train_regimes=train_regimes,
        max_lag=max_lag,
    )

    historical_bank = _build_historical_bank(
        data=data_with_regime,
        variables=list(var_model["variables"]),
        horizon=spec.rl_episode_horizon_days,
        pd=pd,
    )
    var_model["rl_state_support"] = rl_state_support
    var_model["rl_crisis_seeds"] = crisis_seeds
    var_model["rl_historical_bank"] = historical_bank
    var_model["rl_historical_bank_count"] = len(historical_bank)

    # ---- Load causal graph ----
    print("  [real_mode] Loading canonical causal graph...")
    try:
        from canonical_best_model import load_canonical_graph
        causal_adj = load_canonical_graph(str(ROOT))
    except Exception as e:
        raise RuntimeError(
            f"Could not load canonical causal graph: {e}. "
            f"Verify regime_causal_graphs.json exists at {ROOT}."
        ) from e
    print(f"  [real_mode] Loaded canonical graph: {len(causal_adj)} edges")

    # ---- Return scenario_fn ----
    from generative_engine.scenario_generator import generate_scenarios

    return var_model, causal_adj, generate_scenarios


# ============================================================================
# DIAGNOSTIC: smoke-check that real mode is operational
# ============================================================================

def diagnose_real_mode_readiness() -> Dict[str, Any]:
    """Run lightweight checks for real-mode prerequisites without doing the
    expensive VAR fit. Returns a status dict; does not raise.

    Useful as a pre-flight check before launching production training.
    """
    report: Dict[str, Any] = {
        "production_imports": False,
        "db_connection": False,
        "processed_data_present": False,
        "regimes_present": False,
        "canonical_graph_present": False,
        "errors": [],
    }

    try:
        pd = _import_production_deps()
        report["production_imports"] = True
    except Exception as e:
        report["errors"].append(f"production_imports: {e}")
        # Still check the file-based canonical graph since it doesn't need pandas
        try:
            from canonical_best_model import load_canonical_graph
            adj = load_canonical_graph(str(ROOT))
            report["canonical_graph_present"] = len(adj) > 0
            report["canonical_graph_edges"] = len(adj)
        except Exception as e2:
            report["errors"].append(f"canonical_graph: {e2}")
        return report

    # Always check the canonical graph — it doesn't depend on the DB
    try:
        from canonical_best_model import load_canonical_graph
        adj = load_canonical_graph(str(ROOT))
        report["canonical_graph_present"] = len(adj) > 0
        report["canonical_graph_edges"] = len(adj)
    except Exception as e:
        report["errors"].append(f"canonical_graph: {e}")

    try:
        conn = _get_db_connection()
        conn.close()
        report["db_connection"] = True
    except Exception as e:
        report["errors"].append(f"db_connection: {e}")
        # If DB is down, no point checking DB-dependent fields
        return report

    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM processed.time_series_data")
        n_rows = cur.fetchone()[0]
        cur.close()
        conn.close()
        report["processed_data_present"] = n_rows > 0
        report["processed_rows"] = n_rows
    except Exception as e:
        report["errors"].append(f"processed_data: {e}")

    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM models.regimes")
        n_regimes = cur.fetchone()[0]
        cur.close()
        conn.close()
        report["regimes_present"] = n_regimes > 0
        report["regime_rows"] = n_regimes
    except Exception as e:
        report["errors"].append(f"regimes: {e}")

    return report


def print_diagnosis(report: Dict[str, Any]) -> None:
    """Pretty-print a diagnosis report."""
    def line(name, ok, extra=""):
        sym = "OK  " if ok else "FAIL"
        return f"  [{sym}] {name}{('  ' + extra) if extra else ''}"

    print()
    print("Real-mode readiness diagnosis")
    print("-" * 60)
    print(line("production_imports", report["production_imports"]))
    print(line("db_connection", report["db_connection"]))
    print(line(
        "processed_data_present",
        report["processed_data_present"],
        f"({report.get('processed_rows', '?')} rows)",
    ))
    print(line(
        "regimes_present",
        report["regimes_present"],
        f"({report.get('regime_rows', '?')} rows)",
    ))
    print(line(
        "canonical_graph_present",
        report["canonical_graph_present"],
        f"({report.get('canonical_graph_edges', '?')} edges)",
    ))

    if report["errors"]:
        print()
        print("Errors:")
        for e in report["errors"]:
            print(f"  - {e}")
    print()
