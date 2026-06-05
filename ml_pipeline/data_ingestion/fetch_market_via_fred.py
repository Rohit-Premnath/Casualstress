"""
Market Data via FRED
====================
Fetches core market variables directly from FRED (bypasses Yahoo Finance).
Writes results into raw_yahoo.daily_prices using Yahoo ticker names so the
rest of the pipeline works unchanged.

Covers:
  ^GSPC   ← SP500       (S&P 500 daily close)
  ^VIX    ← VIXCLS      (VIX)
  CL=F    ← DCOILWTICO  (WTI Crude Oil)
  GC=F    ← GOLDAMGBD228NLBM (Gold)
  DX-Y.NYB← DTWEXBGS    (USD Broad Index)
  EURUSD=X← DEXUSEU     (EUR/USD)

Sector ETFs (XLK, XLF …) are not available on FRED; those columns will
simply be absent from the processed data, which the pipeline handles gracefully.
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

# Add repo root to path for psycopg2 shim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import psycopg2
from psycopg2.extras import execute_values

START_DATE = "2005-01-01"

# FRED series → Yahoo Finance ticker name
FRED_TO_YAHOO = {
    "SP500":               "^GSPC",
    "VIXCLS":              "^VIX",
    "DCOILWTICO":          "CL=F",
    "GOLDAMGBD228NLBM":    "GC=F",
    "DTWEXBGS":            "DX-Y.NYB",
    "DEXUSEU":             "EURUSD=X",
}


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def fetch_and_store():
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key or api_key == "your_fred_api_key_here":
        print("ERROR: FRED_API_KEY not set in .env")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    conn = get_db_connection()
    cursor = conn.cursor()

    total_rows = 0

    for fred_series, yahoo_ticker in FRED_TO_YAHOO.items():
        print(f"  Fetching {fred_series} → {yahoo_ticker} ...", end=" ", flush=True)
        try:
            series = fred.get_series(fred_series, observation_start=START_DATE)
            series = series.dropna()
            if series.empty:
                print("empty, skipping")
                continue

            records = []
            for date, value in series.items():
                close = float(value)
                records.append((
                    yahoo_ticker,
                    date.date(),
                    close,   # open  (approximated with close)
                    close,   # high
                    close,   # low
                    close,   # close
                    close,   # adj_close
                    0,       # volume (unavailable from FRED)
                ))

            execute_values(
                cursor,
                """
                INSERT INTO raw_yahoo.daily_prices
                    (ticker, date, open_price, high_price, low_price,
                     close_price, adj_close, volume, fetched_at)
                VALUES %s
                ON CONFLICT (ticker, date) DO UPDATE SET
                    close_price = EXCLUDED.close_price,
                    adj_close   = EXCLUDED.adj_close,
                    fetched_at  = NOW()
                """,
                records,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, NOW())",
            )
            conn.commit()
            print(f"OK — {len(records)} rows")
            total_rows += len(records)
            time.sleep(0.5)

        except Exception as e:
            conn.rollback()
            print(f"FAILED — {e}")

    cursor.close()
    conn.close()
    print(f"\nDone. {total_rows} total rows written to raw_yahoo.daily_prices.")
    print("Tickers loaded:", list(FRED_TO_YAHOO.values()))
    print("\nNote: Sector ETFs (XLK, XLF …) are not available via FRED.")
    print("The pipeline will run without them — those columns will be absent.")


if __name__ == "__main__":
    print("=" * 60)
    print("CAUSALSTRESS — Market Data via FRED (Yahoo Finance bypass)")
    print("=" * 60)
    fetch_and_store()
