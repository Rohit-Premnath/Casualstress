"""Quick test to verify FRED API key and database connection work."""

import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: FRED API
print("Testing FRED API connection...")
try:
    from fredapi import Fred
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    data = fred.get_series("CPIAUCSL", observation_start="2024-01-01")
    print(f"  OK - Got {len(data)} CPI observations")
except Exception as e:
    print(f"  FAILED - {e}")

# Test 2: Database
print("\nTesting database connection...")
try:
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",  # localhost because we're outside Docker
        port="5433",
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_fred.observations")
    count = cursor.fetchone()[0]
    print(f"  OK - raw_fred.observations exists ({count} rows)")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"  FAILED - {e}")

print("\nDone!")