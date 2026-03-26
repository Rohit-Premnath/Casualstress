"""
Historical Crisis Library
===========================
Labels and extracts historical crisis windows from our data.
These labeled crises become the training data for the generative model.

Each crisis is characterized by:
- Start and end dates
- Type (equity crash, credit crisis, rate shock, etc.)
- Severity metrics (max drawdown, VIX peak, speed of decline)
- Multi-variable time paths during the crisis
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


# ============================================
# CRISIS DEFINITIONS
# ============================================

CRISIS_CATALOG = [
    {
        "id": "dotcom_2000",
        "name": "Dot-Com Crash",
        "start": "2000-03-10",
        "end": "2002-10-09",
        "type": "equity_crash",
        "description": "Tech bubble burst, NASDAQ fell 78%",
    },
    {
        "id": "gfc_2008",
        "name": "Global Financial Crisis",
        "start": "2007-10-09",
        "end": "2009-03-09",
        "type": "credit_crisis",
        "description": "Subprime mortgage crisis, banking system near-collapse",
    },
    {
        "id": "flash_crash_2010",
        "name": "Flash Crash",
        "start": "2010-04-26",
        "end": "2010-07-02",
        "type": "liquidity_crisis",
        "description": "Dow dropped 1000 points in minutes, Euro crisis fears",
    },
    {
        "id": "euro_debt_2011",
        "name": "US Downgrade / Euro Debt Crisis",
        "start": "2011-07-07",
        "end": "2011-11-25",
        "type": "sovereign_crisis",
        "description": "S&P downgraded US debt, European sovereign debt panic",
    },
    {
        "id": "china_oil_2015",
        "name": "China Devaluation / Oil Crash",
        "start": "2015-08-10",
        "end": "2016-02-11",
        "type": "global_shock",
        "description": "China devalued yuan, oil crashed to $26, global contagion",
    },
    {
        "id": "volmageddon_2018",
        "name": "Volmageddon",
        "start": "2018-01-26",
        "end": "2018-04-02",
        "type": "volatility_shock",
        "description": "VIX spike destroyed inverse VIX products, rapid selloff",
    },
    {
        "id": "fed_tightening_2018",
        "name": "Fed Tightening Selloff",
        "start": "2018-09-20",
        "end": "2018-12-24",
        "type": "rate_shock",
        "description": "Fed raising rates too fast, yield curve inversion fears",
    },
    {
        "id": "covid_2020",
        "name": "COVID Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "type": "pandemic_shock",
        "description": "Fastest 30% decline in history, global lockdowns",
    },
    {
        "id": "rate_hike_2022",
        "name": "Rate Hike Selloff",
        "start": "2022-01-03",
        "end": "2022-10-12",
        "type": "rate_shock",
        "description": "Fed aggressive rate hikes, inflation shock, S&P fell 25%",
    },
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
# EXTRACT CRISIS WINDOWS
# ============================================

def extract_crisis_windows():
    """
    Extract multi-variable time paths for each crisis window.
    Returns a dict of DataFrames, one per crisis.
    """
    print("Extracting crisis windows from processed data...\n")

    conn = get_db_connection()

    # Load all processed data (transformed values)
    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE source != 'engineered'
        ORDER BY date
    """, conn)
    conn.close()

    # Pivot to wide format
    pivoted = df.pivot_table(
        index="date", columns="variable_code", values="transformed_value"
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.sort_index().dropna(axis=1, thresh=int(len(pivoted) * 0.7)).dropna()

    crisis_windows = {}
    valid_count = 0

    print(f"  {'Crisis':<35} {'Start':>12} {'End':>12} {'Days':>6} {'Vars':>6} {'Status':<10}")
    print("  " + "-" * 85)

    for crisis in CRISIS_CATALOG:
        start = pd.to_datetime(crisis["start"])
        end = pd.to_datetime(crisis["end"])

        # Extract window
        mask = (pivoted.index >= start) & (pivoted.index <= end)
        window = pivoted[mask]

        if len(window) < 10:
            print(f"  {crisis['name']:<35} {crisis['start']:>12} {crisis['end']:>12} "
                  f"{len(window):>6} {0:>6} {'SKIPPED':<10}")
            continue

        # Compute crisis characteristics
        if "^GSPC" in window.columns:
            # Cumulative return during crisis
            cum_return = window["^GSPC"].sum()
            max_daily_drop = window["^GSPC"].min()
        else:
            cum_return = 0
            max_daily_drop = 0

        if "^VIX" in window.columns:
            vix_peak = window["^VIX"].max()
        else:
            vix_peak = 0

        crisis_windows[crisis["id"]] = {
            "data": window,
            "metadata": {
                **crisis,
                "n_days": len(window),
                "n_vars": len(window.columns),
                "cum_spx_return": float(cum_return),
                "max_daily_drop": float(max_daily_drop),
            }
        }

        valid_count += 1
        print(f"  {crisis['name']:<35} {crisis['start']:>12} {crisis['end']:>12} "
              f"{len(window):>6} {len(window.columns):>6} {'OK':<10}")

    print(f"\n  Extracted {valid_count}/{len(CRISIS_CATALOG)} crisis windows")

    return crisis_windows, pivoted


def get_non_crisis_windows(pivoted, window_size=60, n_windows=20):
    """
    Extract random non-crisis (calm) windows for contrast training.
    The model needs to know what normal looks like too.
    """
    print(f"\nExtracting {n_windows} non-crisis windows (size={window_size} days)...")

    # Build set of all crisis dates
    crisis_dates = set()
    for crisis in CRISIS_CATALOG:
        start = pd.to_datetime(crisis["start"])
        end = pd.to_datetime(crisis["end"])
        dates = pivoted.index[(pivoted.index >= start) & (pivoted.index <= end)]
        crisis_dates.update(dates)

    # Find non-crisis indices
    non_crisis_mask = ~pivoted.index.isin(crisis_dates)
    non_crisis_indices = np.where(non_crisis_mask)[0]

    # Sample random windows
    np.random.seed(42)
    calm_windows = []

    valid_starts = non_crisis_indices[non_crisis_indices < len(pivoted) - window_size]

    if len(valid_starts) < n_windows:
        n_windows = len(valid_starts)

    selected = np.random.choice(valid_starts, size=n_windows, replace=False)

    for start_idx in selected:
        window = pivoted.iloc[start_idx:start_idx + window_size]
        calm_windows.append(window)

    print(f"  Extracted {len(calm_windows)} calm windows")

    return calm_windows


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CAUSALSTRESS - CRISIS LIBRARY")
    print("=" * 60)

    crisis_windows, pivoted = extract_crisis_windows()

    calm_windows = get_non_crisis_windows(pivoted)

    print(f"\n✓ Crisis library complete!")
    print(f"  {len(crisis_windows)} crisis windows extracted")
    print(f"  {len(calm_windows)} calm windows for contrast")
    print("=" * 60)

    return crisis_windows, calm_windows, pivoted


if __name__ == "__main__":
    main()