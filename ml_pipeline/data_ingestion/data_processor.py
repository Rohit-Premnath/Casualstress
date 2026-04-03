"""
Data Processor
==============
Takes raw data from raw_fred and raw_yahoo schemas, and produces
clean, aligned, stationary time-series in the processed schema.

Pipeline steps:
1. Load raw data from both sources
2. Align everything to daily frequency (forward-fill lower frequencies)
3. Handle missing data (interpolation for small gaps)
4. Apply stationarity transforms (log-returns for prices, first-diff for levels)
5. Compute engineered features (rolling volatility, z-scores)
6. Store everything in processed.time_series_data
"""

import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

# Define how each variable should be transformed for stationarity
# "log_return" = log(today/yesterday) — best for prices
# "first_diff" = today - yesterday — best for rates and percentages
# "pct_change" = (today - yesterday) / yesterday — alternative for prices
# "none" = already stationary or use as-is

FRED_TRANSFORMS = {
    "A191RL1Q225SBEA": {"transform": "none", "name": "GDP Growth Rate"},
    "INDPRO": {"transform": "log_return", "name": "Industrial Production"},
    "CPIAUCSL": {"transform": "log_return", "name": "CPI"},
    "PCEPILFE": {"transform": "log_return", "name": "Core PCE"},
    "UNRATE": {"transform": "first_diff", "name": "Unemployment Rate"},
    "PAYEMS": {"transform": "log_return", "name": "Nonfarm Payrolls"},
    "ICSA": {"transform": "log_return", "name": "Initial Jobless Claims"},
    "FEDFUNDS": {"transform": "first_diff", "name": "Federal Funds Rate"},
    "DGS10": {"transform": "first_diff", "name": "10Y Treasury Yield"},
    "DGS2": {"transform": "first_diff", "name": "2Y Treasury Yield"},
    "T10Y2Y": {"transform": "first_diff", "name": "Yield Curve Spread"},
    "M2SL": {"transform": "log_return", "name": "M2 Money Supply"},
    "HOUST": {"transform": "log_return", "name": "Housing Starts"},
    "UMCSENT": {"transform": "first_diff", "name": "Consumer Sentiment"},
    "RSXFS": {"transform": "log_return", "name": "Retail Sales"},
    "TEDRATE": {"transform": "first_diff", "name": "TED Spread"},
    "BAMLH0A0HYM2": {"transform": "first_diff", "name": "High Yield Spread"},
    # Credit Spread Ladder
    "BAMLC0A0CM": {"transform": "first_diff", "name": "IG Corporate OAS"},
    "BAMLC0A4CBBB": {"transform": "first_diff", "name": "BBB Corporate OAS"},
    "BAMLC0A3CA": {"transform": "first_diff", "name": "A-rated Corporate OAS"},
    "BAMLC0A2CAA": {"transform": "first_diff", "name": "AA-rated Corporate OAS"},
    "BAMLC0A1CAAA": {"transform": "first_diff", "name": "AAA Corporate OAS"},
    "BAMLH0A1HYBB": {"transform": "first_diff", "name": "BB High Yield OAS"},
    "BAMLH0A2HYB": {"transform": "first_diff", "name": "B High Yield OAS"},
    "BAMLH0A3HYC": {"transform": "first_diff", "name": "CCC High Yield OAS"},
    "BAMLEMCBPIOAS": {"transform": "first_diff", "name": "EM Corporate OAS"},
    # Bank Lending
    "DRTSCILM": {"transform": "first_diff", "name": "Banks Tightening C&I Large"},
    "DRTSCIS": {"transform": "first_diff", "name": "Banks Tightening C&I Small"},
    "DRTSSP": {"transform": "first_diff", "name": "Banks Tightening Mortgages"},
    "DRSDCILM": {"transform": "first_diff", "name": "C&I Loan Demand Large"},
    # Funding Stress
    "SOFR": {"transform": "first_diff", "name": "SOFR Rate"},
    "SOFR90DAYAVG": {"transform": "first_diff", "name": "90-Day SOFR Average"},
    "DCPF3M": {"transform": "first_diff", "name": "3M Financial CP Rate"},
    "DCPN3M": {"transform": "first_diff", "name": "3M Nonfinancial CP Rate"},
    # Composite
    "STLFSI4": {"transform": "none", "name": "St. Louis Fed Stress Index"},
}

YAHOO_TRANSFORMS = {
    "^GSPC": {"transform": "log_return", "name": "S&P 500"},
    "^NDX": {"transform": "log_return", "name": "NASDAQ 100"},
    "^RUT": {"transform": "log_return", "name": "Russell 2000"},
    "XLK": {"transform": "log_return", "name": "Tech Sector ETF"},
    "XLF": {"transform": "log_return", "name": "Financial Sector ETF"},
    "XLE": {"transform": "log_return", "name": "Energy Sector ETF"},
    "XLV": {"transform": "log_return", "name": "Healthcare Sector ETF"},
    "XLY": {"transform": "log_return", "name": "Consumer Disc ETF"},
    "XLRE": {"transform": "log_return", "name": "Real Estate Sector ETF"},
    "XLU": {"transform": "log_return", "name": "Utilities Sector ETF"},
    "^VIX": {"transform": "log_return", "name": "VIX"},
    "^VVIX": {"transform": "log_return", "name": "VVIX"},
    "CL=F": {"transform": "log_return", "name": "Crude Oil"},
    "GC=F": {"transform": "log_return", "name": "Gold"},
    "DX-Y.NYB": {"transform": "log_return", "name": "US Dollar Index"},
    "EURUSD=X": {"transform": "log_return", "name": "EUR/USD"},
    "TLT": {"transform": "log_return", "name": "20Y Treasury Bond ETF"},
    "LQD": {"transform": "log_return", "name": "Investment Grade Bond ETF"},
    "HYG": {"transform": "log_return", "name": "High Yield Bond ETF"},
    "EEM": {"transform": "log_return", "name": "Emerging Markets ETF"},
    "^MOVE": {"transform": "log_return", "name": "MOVE Index (Bond Vol)"},
}


# ============================================
# DATABASE CONNECTION
# ============================================

def get_db_connection():
    """Create a connection to PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# STEP 1: LOAD RAW DATA
# ============================================

def load_fred_data():
    """Load all FRED data from raw schema into a pivoted DataFrame."""
    print("Loading FRED data from database...")

    conn = get_db_connection()

    df = pd.read_sql("""
        SELECT date, series_id, value
        FROM raw_fred.observations
        ORDER BY date
    """, conn)

    conn.close()

    if df.empty:
        print("  ERROR: No FRED data found! Run fred_fetcher.py first.")
        sys.exit(1)

    # Pivot so each series is a column: rows=dates, columns=series_ids
    pivoted = df.pivot_table(index="date", columns="series_id", values="value")
    pivoted.index = pd.to_datetime(pivoted.index)

    print(f"  Loaded {len(pivoted)} dates x {len(pivoted.columns)} series from FRED")
    return pivoted


def load_yahoo_data():
    """Load all Yahoo Finance data from raw schema into a pivoted DataFrame."""
    print("Loading Yahoo Finance data from database...")

    conn = get_db_connection()

    # Use adj_close as the primary price (accounts for splits and dividends)
    df = pd.read_sql("""
        SELECT date, ticker, adj_close as value
        FROM raw_yahoo.daily_prices
        WHERE adj_close IS NOT NULL
        ORDER BY date
    """, conn)

    conn.close()

    if df.empty:
        print("  ERROR: No Yahoo data found! Run yahoo_fetcher.py first.")
        sys.exit(1)

    # Pivot so each ticker is a column
    pivoted = df.pivot_table(index="date", columns="ticker", values="value")
    pivoted.index = pd.to_datetime(pivoted.index)

    print(f"  Loaded {len(pivoted)} dates x {len(pivoted.columns)} tickers from Yahoo")
    return pivoted


# ============================================
# STEP 2: ALIGN TO DAILY FREQUENCY
# ============================================

def align_to_daily(fred_df, yahoo_df):
    """
    Align all data to a common daily date index.

    - Yahoo data is already daily (market days)
    - FRED data may be monthly/quarterly — forward-fill to daily
    - Use business day frequency to match market calendar
    """
    print("\nAligning all data to daily frequency...")

    # Use Yahoo's date range as the master calendar (market days)
    date_range = yahoo_df.index

    # Reindex FRED data to daily, forward-filling lower frequency data
    # This means monthly GDP stays constant until the next monthly release
    fred_daily = fred_df.reindex(date_range, method="ffill")

    # Yahoo is already daily, just make sure indexes match
    yahoo_daily = yahoo_df.reindex(date_range)

    # Combine into one DataFrame
    combined = pd.concat([fred_daily, yahoo_daily], axis=1)

    print(f"  Combined dataset: {len(combined)} days x {len(combined.columns)} variables")
    print(f"  Date range: {combined.index.min().date()} to {combined.index.max().date()}")

    return combined


# ============================================
# STEP 3: HANDLE MISSING DATA
# ============================================

def handle_missing_data(df):
    """
    Handle missing values in the aligned dataset.

    Strategy:
    - For gaps <= 5 days: linear interpolation
    - For remaining gaps: forward-fill (carry last known value)
    - Track which values were imputed
    """
    print("\nHandling missing data...")

    # Count missing before
    missing_before = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"  Missing before: {missing_before} / {total_cells} ({missing_before/total_cells*100:.2f}%)")

    # Create a mask of which values are originally missing (for tracking imputation)
    imputed_mask = df.isnull().copy()

    # Step 1: Interpolate small gaps (up to 5 days)
    df = df.interpolate(method="linear", limit=5)

    # Step 2: Forward-fill remaining gaps
    df = df.ffill()

    # Step 3: Backward-fill any remaining NaN at the very start
    df = df.bfill()

    # Count missing after
    missing_after = df.isnull().sum().sum()
    imputed_count = imputed_mask.sum().sum() - missing_after
    print(f"  Missing after: {missing_after} / {total_cells} ({missing_after/total_cells*100:.2f}%)")
    print(f"  Values imputed: {imputed_count}")

    # Report any columns that still have missing data
    still_missing = df.isnull().sum()
    still_missing = still_missing[still_missing > 0]
    if len(still_missing) > 0:
        print(f"  WARNING: {len(still_missing)} columns still have missing data:")
        for col, count in still_missing.items():
            print(f"    {col}: {count} missing")

    return df, imputed_mask


# ============================================
# STEP 4: STATIONARITY TRANSFORMS
# ============================================

def apply_transforms(df):
    """
    Apply stationarity transforms to each variable.

    - log_return: log(price_t / price_t-1) — for prices
    - first_diff: value_t - value_t-1 — for rates and percentages
    - none: keep as-is — already stationary
    """
    print("\nApplying stationarity transforms...")

    raw_values = {}      # Store the raw values for reference
    transformed = {}     # Store the transformed values

    all_transforms = {}
    all_transforms.update(FRED_TRANSFORMS)
    all_transforms.update(YAHOO_TRANSFORMS)

    for col in df.columns:
        if col in all_transforms:
            config = all_transforms[col]
            transform_type = config["transform"]
            name = config["name"]

            raw_values[col] = df[col].copy()

            if transform_type == "log_return":
                # log(price_t / price_t-1) — handles percentage changes naturally
                transformed[col] = np.log(df[col] / df[col].shift(1))
            elif transform_type == "first_diff":
                # value_t - value_t-1
                transformed[col] = df[col] - df[col].shift(1)
            elif transform_type == "pct_change":
                # (value_t - value_t-1) / value_t-1
                transformed[col] = df[col].pct_change()
            else:
                # "none" — keep as-is
                transformed[col] = df[col].copy()

            print(f"  {col:<25} ({name:<30}) -> {transform_type}")
        else:
            print(f"  {col:<25} (unknown) -> skipped")

    raw_df = pd.DataFrame(raw_values, index=df.index)
    transformed_df = pd.DataFrame(transformed, index=df.index)

    # Drop the first row (NaN from differencing/returns)
    transformed_df = transformed_df.iloc[1:]
    raw_df = raw_df.loc[transformed_df.index]

    # Replace any infinite values with NaN, then drop
    transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan)
    transformed_df = transformed_df.ffill()

    print(f"\n  Transformed dataset: {len(transformed_df)} days x {len(transformed_df.columns)} variables")

    return raw_df, transformed_df


# ============================================
# STEP 5: FEATURE ENGINEERING
# ============================================

def compute_features(transformed_df, raw_df):
    """
    Compute additional engineered features:
    - Rolling 21-day volatility (1 month) for key price series
    - Rolling 63-day volatility (3 months) for key price series
    - Z-score of credit spreads
    """
    print("\nComputing engineered features...")

    features = {}

    # Rolling volatilities for key equity indices
    vol_tickers = ["^GSPC", "^NDX", "^RUT", "^VIX"]
    for ticker in vol_tickers:
        if ticker in transformed_df.columns:
            # 21-day rolling volatility (annualized)
            vol_21 = transformed_df[ticker].rolling(21).std() * np.sqrt(252)
            features[f"{ticker}_vol_21d"] = vol_21

            # 63-day rolling volatility (annualized)
            vol_63 = transformed_df[ticker].rolling(63).std() * np.sqrt(252)
            features[f"{ticker}_vol_63d"] = vol_63

            print(f"  {ticker}: 21d and 63d rolling volatility computed")

    # Z-score of high yield spread (how extreme is the current spread?)
    if "BAMLH0A0HYM2" in raw_df.columns:
        hy_spread = raw_df["BAMLH0A0HYM2"]
        rolling_mean = hy_spread.rolling(252).mean()
        rolling_std = hy_spread.rolling(252).std()
        z_score = (hy_spread - rolling_mean) / rolling_std
        features["HY_SPREAD_ZSCORE"] = z_score
        print("  BAMLH0A0HYM2: High yield spread z-score computed")

    # Z-score of VIX
    if "^VIX" in raw_df.columns:
        vix = raw_df["^VIX"]
        rolling_mean = vix.rolling(252).mean()
        rolling_std = vix.rolling(252).std()
        z_score = (vix - rolling_mean) / rolling_std
        features["VIX_ZSCORE"] = z_score
        print("  ^VIX: VIX z-score computed")

    if features:
        features_df = pd.DataFrame(features, index=transformed_df.index)
        # Forward-fill any NaN from rolling windows
        features_df = features_df.ffill().bfill()
        print(f"\n  Engineered {len(features_df.columns)} additional features")
    
    # Credit market engineered features
    if "BAMLH0A0HYM2" in raw_df.columns and "BAMLC0A0CM" in raw_df.columns:
        features["HY_IG_SPREAD_GAP"] = raw_df["BAMLH0A0HYM2"] - raw_df["BAMLC0A0CM"]
        print("  HY_IG_SPREAD_GAP: High yield vs investment grade gap computed")

    if "BAMLH0A3HYC" in raw_df.columns and "BAMLH0A1HYBB" in raw_df.columns:
        features["CCC_BB_SPREAD_GAP"] = raw_df["BAMLH0A3HYC"] - raw_df["BAMLH0A1HYBB"]
        print("  CCC_BB_SPREAD_GAP: Distress spread within high yield computed")

    if "DCPF3M" in raw_df.columns and "SOFR" in raw_df.columns:
        features["SOFR_CP_SPREAD"] = raw_df["DCPF3M"] - raw_df["SOFR"]
        print("  SOFR_CP_SPREAD: Modern TED spread proxy computed")
        return features_df
    else:
        return pd.DataFrame(index=transformed_df.index)


# ============================================
# STEP 6: STORE IN DATABASE
# ============================================

def store_processed_data(raw_df, transformed_df, features_df, imputed_mask):
    """
    Store all processed data in the processed.time_series_data hypertable.
    """
    print("\nStoring processed data in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear existing processed data (we regenerate from scratch each time)
    cursor.execute("DELETE FROM processed.time_series_data")
    print("  Cleared existing processed data")

    records = []

    # Add transformed variables (with their raw values)
    for col in transformed_df.columns:
        for date in transformed_df.index:
            raw_val = raw_df.loc[date, col] if col in raw_df.columns else None
            trans_val = transformed_df.loc[date, col]

            # Determine source
            source = "fred" if col in FRED_TRANSFORMS else "yahoo"

            # Check if this was imputed
            is_imputed = False
            if col in imputed_mask.columns and date in imputed_mask.index:
                is_imputed = bool(imputed_mask.loc[date, col])

            if pd.notna(trans_val):
                records.append((
                    date.date() if hasattr(date, 'date') else date,
                    col,
                    float(raw_val) if pd.notna(raw_val) else None,
                    float(trans_val),
                    source,
                    is_imputed,
                ))

    # Add engineered features
    for col in features_df.columns:
        for date in features_df.index:
            val = features_df.loc[date, col]
            if pd.notna(val):
                records.append((
                    date.date() if hasattr(date, 'date') else date,
                    col,
                    None,  # no raw value for engineered features
                    float(val),
                    "engineered",
                    False,
                ))

    print(f"  Preparing to insert {len(records)} records...")

    # Batch insert for speed
    BATCH_SIZE = 10000
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        execute_values(
            cursor,
            """
            INSERT INTO processed.time_series_data
                (date, variable_code, raw_value, transformed_value, source, is_imputed)
            VALUES %s
            """,
            batch,
            template="(%s, %s, %s, %s, %s, %s)",
        )
        print(f"  Inserted batch {i//BATCH_SIZE + 1} ({min(i+BATCH_SIZE, len(records))}/{len(records)})")

    conn.commit()
    print(f"  Successfully stored {len(records)} records in processed.time_series_data")

    cursor.close()
    conn.close()


# ============================================
# STEP 7: VERIFICATION
# ============================================

def verify_processed_data():
    """Verify the processed data looks correct."""
    print("\nVerifying processed data...\n")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Summary by variable
    cursor.execute("""
        SELECT
            variable_code,
            source,
            COUNT(*) as obs_count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            ROUND(AVG(transformed_value)::numeric, 6) as mean_val,
            ROUND(STDDEV(transformed_value)::numeric, 6) as std_val
        FROM processed.time_series_data
        GROUP BY variable_code, source
        ORDER BY source, variable_code
    """)

    rows = cursor.fetchall()

    print(f"{'Variable':<25} {'Source':<12} {'Count':>7} {'First':>12} {'Last':>12} {'Mean':>10} {'Std':>10}")
    print("-" * 92)

    total = 0
    for row in rows:
        var, source, count, first, last, mean, std_val = row
        print(f"{var:<25} {source:<12} {count:>7} {str(first):>12} {str(last):>12} {mean:>10} {std_val:>10}")
        total += count

    print("-" * 92)
    print(f"{'TOTAL':<25} {'':>12} {total:>7}")

    # Check for any extreme values that might indicate problems
    cursor.execute("""
        SELECT variable_code, COUNT(*) as extreme_count
        FROM processed.time_series_data
        WHERE ABS(transformed_value) > 0.5
          AND source != 'engineered'
        GROUP BY variable_code
        HAVING COUNT(*) > 100
        ORDER BY extreme_count DESC
        LIMIT 5
    """)

    extremes = cursor.fetchall()
    if extremes:
        print("\nVariables with many large values (may need review):")
        for var, count in extremes:
            print(f"  {var}: {count} values > |0.5|")

    cursor.close()
    conn.close()

    return total > 0


# ============================================
# MAIN
# ============================================

def main():
    """Main entry point: process all raw data into clean, aligned, stationary format."""
    print("=" * 60)
    print("CAUSALSTRESS - DATA PROCESSOR")
    print("=" * 60)

    # Step 1: Load raw data
    fred_df = load_fred_data()
    yahoo_df = load_yahoo_data()

    # Step 2: Align to daily frequency
    combined_df = align_to_daily(fred_df, yahoo_df)

    # Step 3: Handle missing data
    clean_df, imputed_mask = handle_missing_data(combined_df)

    # Step 4: Apply stationarity transforms
    raw_df, transformed_df = apply_transforms(clean_df)

    # Step 5: Compute engineered features
    features_df = compute_features(transformed_df, raw_df)

    # Step 6: Store in database
    store_processed_data(raw_df, transformed_df, features_df, imputed_mask)

    # Step 7: Verify
    success = verify_processed_data()

    if success:
        print("\n✓ Data processing complete!")
        print(f"  {len(transformed_df.columns)} base variables + {len(features_df.columns)} engineered features")
        print(f"  {len(transformed_df)} trading days of clean, stationary data")
        print(f"  Ready for causal discovery engine!")
    else:
        print("\n✗ Data processing failed - check errors above")

    print("=" * 60)


if __name__ == "__main__":
    main()