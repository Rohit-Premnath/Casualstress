"""
FRED Data Fetcher
=================
Pulls macro-economic data from the Federal Reserve Economic Data (FRED) API
and stores it in the raw_fred schema in PostgreSQL.

Variables fetched:
- GDP, CPI, Core PCE, Unemployment, Nonfarm Payrolls
- Federal Funds Rate, 10Y/2Y Treasury Yields, Yield Curve Spread
- M2 Money Supply, Housing Starts, Industrial Production
- Consumer Confidence, TED Spread, High Yield Spread
- Initial Jobless Claims, Retail Sales
"""

import os
import sys
from datetime import datetime

import pandas as pd
from fredapi import Fred
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# All 18 FRED series we need
FRED_SERIES = {
    # Output
    "A191RL1Q225SBEA": {
        "name": "GDP Growth Rate",
        "category": "output",
        "frequency": "quarterly",
    },
    "INDPRO": {
        "name": "Industrial Production Index",
        "category": "output",
        "frequency": "monthly",
    },
    # Inflation
    "CPIAUCSL": {
        "name": "CPI All Urban Consumers",
        "category": "inflation",
        "frequency": "monthly",
    },
    "PCEPILFE": {
        "name": "Core PCE Price Index",
        "category": "inflation",
        "frequency": "monthly",
    },
    # Labor
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "labor",
        "frequency": "monthly",
    },
    "PAYEMS": {
        "name": "Nonfarm Payrolls",
        "category": "labor",
        "frequency": "monthly",
    },
    "ICSA": {
        "name": "Initial Jobless Claims",
        "category": "labor",
        "frequency": "weekly",
    },
    # Monetary Policy
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "category": "monetary_policy",
        "frequency": "monthly",
    },
    # Interest Rates
    "DGS10": {
        "name": "10-Year Treasury Yield",
        "category": "rates",
        "frequency": "daily",
    },
    "DGS2": {
        "name": "2-Year Treasury Yield",
        "category": "rates",
        "frequency": "daily",
    },
    "T10Y2Y": {
        "name": "10Y-2Y Treasury Spread (Yield Curve)",
        "category": "rates",
        "frequency": "daily",
    },
    # Monetary
    "M2SL": {
        "name": "M2 Money Supply",
        "category": "monetary",
        "frequency": "monthly",
    },
    # Housing
    "HOUST": {
        "name": "Housing Starts",
        "category": "housing",
        "frequency": "monthly",
    },
    # Consumer
    "UMCSENT": {
        "name": "Consumer Sentiment (UMich)",
        "category": "sentiment",
        "frequency": "monthly",
    },
    "RSXFS": {
        "name": "Retail Sales Ex Food Services",
        "category": "consumer",
        "frequency": "monthly",
    },
    # Credit Risk
    "TEDRATE": {
        "name": "TED Spread",
        "category": "credit_risk",
        "frequency": "daily",
    },
    "BAMLH0A0HYM2": {
        "name": "ICE BofA High Yield Spread",
        "category": "credit_risk",
        "frequency": "daily",
    },
    # ── Credit Spread Ladder (by rating grade) ──
    "BAMLC0A0CM": {
        "name": "IG Corporate OAS (All Investment Grade)",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLC0A4CBBB": {
        "name": "BBB Corporate OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLC0A3CA": {
        "name": "A-rated Corporate OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLC0A2CAA": {
        "name": "AA-rated Corporate OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLC0A1CAAA": {
        "name": "AAA Corporate OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLH0A1HYBB": {
        "name": "BB High Yield OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLH0A2HYB": {
        "name": "B High Yield OAS",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLH0A3HYC": {
        "name": "CCC High Yield OAS (Distress Indicator)",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    "BAMLEMCBPIOAS": {
        "name": "EM Corporate OAS (Contagion Signal)",
        "category": "credit_spreads",
        "frequency": "daily",
    },
    # ── Bank Lending Behavior (SLOOS - quarterly) ──
    "DRTSCILM": {
        "name": "Banks Tightening C&I Standards (Large Firms)",
        "category": "bank_lending",
        "frequency": "quarterly",
    },
    "DRTSCIS": {
        "name": "Banks Tightening C&I Standards (Small Firms)",
        "category": "bank_lending",
        "frequency": "quarterly",
    },
    "DRTSSP": {
        "name": "Banks Tightening Mortgage Standards",
        "category": "bank_lending",
        "frequency": "quarterly",
    },
    "DRSDCILM": {
        "name": "C&I Loan Demand (Large Firms)",
        "category": "bank_lending",
        "frequency": "quarterly",
    },
    # ── Short-Term Funding Stress ──
    "SOFR": {
        "name": "Secured Overnight Financing Rate",
        "category": "funding_stress",
        "frequency": "daily",
    },
    "SOFR90DAYAVG": {
        "name": "90-Day SOFR Average",
        "category": "funding_stress",
        "frequency": "daily",
    },
    "DCPF3M": {
        "name": "3M AA Financial Commercial Paper Rate",
        "category": "funding_stress",
        "frequency": "daily",
    },
    "DCPN3M": {
        "name": "3M AA Nonfinancial Commercial Paper Rate",
        "category": "funding_stress",
        "frequency": "daily",
    },
    # ── Composite Stress Index ──
    "STLFSI4": {
        "name": "St. Louis Fed Financial Stress Index",
        "category": "composite_stress",
        "frequency": "weekly",
    },
}

# How far back to fetch (10+ years)
START_DATE = "2005-01-01"


# ============================================
# DATABASE CONNECTION
# ============================================

def get_db_connection():
    """Create a connection to PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# FETCH DATA FROM FRED
# ============================================

def fetch_single_series(fred_client, series_id, series_info):
    """
    Fetch a single FRED series and return it as a DataFrame.

    Args:
        fred_client: authenticated Fred API client
        series_id: FRED series code (e.g., "CPIAUCSL")
        series_info: dict with name, category, frequency

    Returns:
        DataFrame with columns [date, series_id, value]
    """
    print(f"  Fetching {series_id} ({series_info['name']})...", end=" ")

    try:
        data = fred_client.get_series(
            series_id,
            observation_start=START_DATE,
        )

        # Convert to DataFrame
        df = data.reset_index()
        df.columns = ["date", "value"]
        df["series_id"] = series_id

        # Drop any rows where value is NaN (FRED sometimes has missing observations)
        df = df.dropna(subset=["value"])

        # Make sure date is a proper date (not datetime)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        print(f"OK - {len(df)} observations ({df['date'].min()} to {df['date'].max()})")
        return df

    except Exception as e:
        print(f"FAILED - {str(e)}")
        return pd.DataFrame()


def fetch_all_series():
    """
    Fetch all 18 FRED series and return combined DataFrame.

    Returns:
        DataFrame with all series combined
    """
    api_key = os.getenv("FRED_API_KEY", "")

    if not api_key or api_key == "your_fred_api_key_here":
        print("ERROR: No FRED API key found!")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then add it to your .env file as FRED_API_KEY=your_key")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    all_data = []

    print(f"\nFetching {len(FRED_SERIES)} series from FRED API...")
    print(f"Date range: {START_DATE} to present\n")

    for series_id, series_info in FRED_SERIES.items():
        df = fetch_single_series(fred, series_id, series_info)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("\nERROR: No data fetched from any series!")
        sys.exit(1)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} observations across {len(all_data)} series")

    return combined


# ============================================
# STORE IN DATABASE
# ============================================

def store_in_database(df):
    """
    Store fetched FRED data in the raw_fred.observations table.
    Uses UPSERT (insert or update on conflict) so it's safe to re-run.

    Args:
        df: DataFrame with columns [date, series_id, value]
    """
    print("\nStoring data in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare data as list of tuples
    records = [
        (row["series_id"], row["date"], row["value"])
        for _, row in df.iterrows()
    ]

    # Upsert: insert new rows, update existing ones
    query = """
        INSERT INTO raw_fred.observations (series_id, date, value, fetched_at)
        VALUES %s
        ON CONFLICT (series_id, date)
        DO UPDATE SET
            value = EXCLUDED.value,
            fetched_at = NOW()
    """

    # Use execute_values for fast bulk insert
    template = "(%(series_id)s, %(date)s, %(value)s, NOW())"

    try:
        execute_values(
            cursor,
            """
            INSERT INTO raw_fred.observations (series_id, date, value, fetched_at)
            VALUES %s
            ON CONFLICT (series_id, date)
            DO UPDATE SET value = EXCLUDED.value, fetched_at = NOW()
            """,
            records,
            template="(%s, %s, %s, NOW())",
        )
        conn.commit()
        print(f"Successfully stored {len(records)} records in raw_fred.observations")

    except Exception as e:
        conn.rollback()
        print(f"ERROR storing data: {str(e)}")
        raise

    finally:
        cursor.close()
        conn.close()


# ============================================
# VERIFICATION
# ============================================

def verify_data():
    """
    Run verification queries to confirm data was stored correctly.
    """
    print("\nVerifying stored data...\n")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Count per series
    cursor.execute("""
        SELECT
            series_id,
            COUNT(*) as obs_count,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM raw_fred.observations
        GROUP BY series_id
        ORDER BY series_id
    """)

    rows = cursor.fetchall()

    print(f"{'Series ID':<25} {'Count':>7} {'First Date':>12} {'Last Date':>12}")
    print("-" * 60)

    total = 0
    for row in rows:
        series_id, count, first_date, last_date = row
        name = FRED_SERIES.get(series_id, {}).get("name", "Unknown")
        print(f"{series_id:<25} {count:>7} {str(first_date):>12} {str(last_date):>12}")
        total += count

    print("-" * 60)
    print(f"{'TOTAL':<25} {total:>7}")
    print(f"\nAll {len(rows)}/{len(FRED_SERIES)} series present in database!")

    cursor.close()
    conn.close()

    return len(rows) == len(FRED_SERIES)


# ============================================
# MAIN
# ============================================

def main():
    """Main entry point: fetch all FRED data and store in database."""
    print("=" * 60)
    print("CAUSALSTRESS - FRED DATA FETCHER")
    print("=" * 60)

    # Step 1: Fetch from FRED API
    df = fetch_all_series()

    # Step 2: Store in PostgreSQL
    store_in_database(df)

    # Step 3: Verify
    success = verify_data()

    if success:
        print("\n✓ FRED data ingestion complete!")
    else:
        print("\n✗ Some series are missing - check errors above")

    print("=" * 60)


if __name__ == "__main__":
    main()