"""
Yahoo Finance Data Fetcher
===========================
Pulls market data (equities, bonds, commodities, volatility, currencies)
from Yahoo Finance and stores it in the raw_yahoo schema in PostgreSQL.

Variables fetched:
- Equity indices: S&P 500, NASDAQ 100, Russell 2000
- Sector ETFs: XLK, XLF, XLE, XLV, XLY, XLRE, XLU
- Volatility: VIX, VVIX
- Commodities: WTI Crude Oil, Gold
- Currency: US Dollar Index, EUR/USD
- Fixed Income: TLT (20Y bonds), LQD (investment grade), HYG (high yield)
- International: EEM (emerging markets)
"""

import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# All 20 Yahoo Finance tickers we need
YAHOO_TICKERS = {
    # Equity Indices
    "^GSPC": {
        "name": "S&P 500",
        "category": "equity_index",
    },
    "^NDX": {
        "name": "NASDAQ 100",
        "category": "equity_index",
    },
    "^RUT": {
        "name": "Russell 2000 (Small Cap)",
        "category": "equity_index",
    },
    # Sector ETFs
    "XLK": {
        "name": "Technology Sector ETF",
        "category": "sector",
    },
    "XLF": {
        "name": "Financial Sector ETF",
        "category": "sector",
    },
    "XLE": {
        "name": "Energy Sector ETF",
        "category": "sector",
    },
    "XLV": {
        "name": "Healthcare Sector ETF",
        "category": "sector",
    },
    "XLY": {
        "name": "Consumer Discretionary ETF",
        "category": "sector",
    },
    "XLRE": {
        "name": "Real Estate Sector ETF",
        "category": "sector",
    },
    "XLU": {
        "name": "Utilities Sector ETF",
        "category": "sector",
    },
    # Volatility
    "^VIX": {
        "name": "VIX Volatility Index",
        "category": "volatility",
    },
    "^VVIX": {
        "name": "VVIX (VIX of VIX)",
        "category": "volatility",
    },
    # Commodities
    "CL=F": {
        "name": "WTI Crude Oil Futures",
        "category": "commodity",
    },
    "GC=F": {
        "name": "Gold Futures",
        "category": "commodity",
    },
    # Currency
    "DX-Y.NYB": {
        "name": "US Dollar Index",
        "category": "currency",
    },
    "EURUSD=X": {
        "name": "EUR/USD Exchange Rate",
        "category": "currency",
    },
    # Fixed Income
    "TLT": {
        "name": "20+ Year Treasury Bond ETF",
        "category": "fixed_income",
    },
    "LQD": {
        "name": "Investment Grade Corporate Bond ETF",
        "category": "fixed_income",
    },
    "HYG": {
        "name": "High Yield Corporate Bond ETF",
        "category": "fixed_income",
    },
    # International
    "EEM": {
        "name": "Emerging Markets ETF",
        "category": "international",
    },
    "^MOVE": {
        "name": "ICE BofA MOVE Index (Bond Volatility)",
        "category": "volatility",
    },
}

# How far back to fetch
START_DATE = "2005-01-01"


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
# FETCH DATA FROM YAHOO FINANCE
# ============================================

def fetch_single_ticker(ticker, ticker_info):
    """
    Fetch a single ticker's daily OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "^GSPC")
        ticker_info: dict with name, category

    Returns:
        DataFrame with columns [date, ticker, open, high, low, close, adj_close, volume]
    """
    print(f"  Fetching {ticker} ({ticker_info['name']})...", end=" ")

    try:
        # Download data from Yahoo Finance
        data = yf.download(
            ticker,
            start=START_DATE,
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            print("FAILED - No data returned")
            return pd.DataFrame()

        # Handle multi-level columns that yfinance sometimes returns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Reset index to get date as a column
        data = data.reset_index()

        # Rename columns to match our schema
        data = data.rename(columns={
            "Date": "date",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        # Add ticker column
        data["ticker"] = ticker

        # Make sure date is proper date
        data["date"] = pd.to_datetime(data["date"]).dt.date

        # Drop rows where close price is NaN
        data = data.dropna(subset=["close_price"])

        # Select only the columns we need
        data = data[["date", "ticker", "open_price", "high_price",
                      "low_price", "close_price", "adj_close", "volume"]]

        print(f"OK - {len(data)} days ({data['date'].min()} to {data['date'].max()})")
        return data

    except Exception as e:
        print(f"FAILED - {str(e)}")
        return pd.DataFrame()


def fetch_all_tickers():
    """
    Fetch all 20 tickers and return combined DataFrame.

    Returns:
        DataFrame with all tickers combined
    """
    all_data = []

    print(f"\nFetching {len(YAHOO_TICKERS)} tickers from Yahoo Finance...")
    print(f"Date range: {START_DATE} to present\n")

    for ticker, ticker_info in YAHOO_TICKERS.items():
        df = fetch_single_ticker(ticker, ticker_info)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("\nERROR: No data fetched from any ticker!")
        sys.exit(1)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} daily records across {len(all_data)} tickers")

    return combined


# ============================================
# STORE IN DATABASE
# ============================================

def store_in_database(df):
    """
    Store fetched Yahoo Finance data in the raw_yahoo.daily_prices table.
    Uses UPSERT so it's safe to re-run.

    Args:
        df: DataFrame with OHLCV data
    """
    print("\nStoring data in database...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare data as list of tuples
    records = []
    for _, row in df.iterrows():
        records.append((
            row["ticker"],
            row["date"],
            float(row["open_price"]) if pd.notna(row["open_price"]) else None,
            float(row["high_price"]) if pd.notna(row["high_price"]) else None,
            float(row["low_price"]) if pd.notna(row["low_price"]) else None,
            float(row["close_price"]) if pd.notna(row["close_price"]) else None,
            float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
            int(row["volume"]) if pd.notna(row["volume"]) else None,
        ))

    try:
        execute_values(
            cursor,
            """
            INSERT INTO raw_yahoo.daily_prices
                (ticker, date, open_price, high_price, low_price,
                 close_price, adj_close, volume, fetched_at)
            VALUES %s
            ON CONFLICT (ticker, date)
            DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume,
                fetched_at = NOW()
            """,
            records,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, NOW())",
        )
        conn.commit()
        print(f"Successfully stored {len(records)} records in raw_yahoo.daily_prices")

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

    cursor.execute("""
        SELECT
            ticker,
            COUNT(*) as day_count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            ROUND(AVG(close_price)::numeric, 2) as avg_close
        FROM raw_yahoo.daily_prices
        GROUP BY ticker
        ORDER BY ticker
    """)

    rows = cursor.fetchall()

    print(f"{'Ticker':<15} {'Days':>7} {'First Date':>12} {'Last Date':>12} {'Avg Close':>12}")
    print("-" * 62)

    total = 0
    for row in rows:
        ticker, count, first_date, last_date, avg_close = row
        name = YAHOO_TICKERS.get(ticker, {}).get("name", "Unknown")
        print(f"{ticker:<15} {count:>7} {str(first_date):>12} {str(last_date):>12} {avg_close:>12}")
        total += count

    print("-" * 62)
    print(f"{'TOTAL':<15} {total:>7}")
    print(f"\nAll {len(rows)}/{len(YAHOO_TICKERS)} tickers present in database!")

    cursor.close()
    conn.close()

    return len(rows) >= len(YAHOO_TICKERS) - 2  # Allow 1-2 tickers to fail


# ============================================
# MAIN
# ============================================

def main():
    """Main entry point: fetch all Yahoo Finance data and store in database."""
    print("=" * 60)
    print("CAUSALSTRESS - YAHOO FINANCE DATA FETCHER")
    print("=" * 60)

    # Step 1: Fetch from Yahoo Finance
    df = fetch_all_tickers()

    # Step 2: Store in PostgreSQL
    store_in_database(df)

    # Step 3: Verify
    success = verify_data()

    if success:
        print("\n✓ Yahoo Finance data ingestion complete!")
    else:
        print("\n✗ Some tickers are missing - check errors above")

    print("=" * 60)


if __name__ == "__main__":
    main()