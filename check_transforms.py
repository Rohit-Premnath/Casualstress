import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5433"),
    dbname=os.getenv("POSTGRES_DB", "causalstress"),
    user=os.getenv("POSTGRES_USER", "causalstress"),
    password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
)
cursor = conn.cursor()
cursor.execute("""
    SELECT DISTINCT variable_code, source,
           MIN(transformed_value), MAX(transformed_value),
           AVG(transformed_value), STDDEV(transformed_value)
    FROM processed.time_series_data
    WHERE source != 'engineered'
    GROUP BY variable_code, source
    ORDER BY source, variable_code
""")
rows = cursor.fetchall()
for r in rows:
    print(f"{r[0]:25s} {r[1]:8s} min={r[2]:10.4f} max={r[3]:10.4f} mean={r[4]:10.4f} std={r[5]:10.4f}")
cursor.close()
conn.close()
