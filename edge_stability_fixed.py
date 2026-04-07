"""
Fixed Edge Stability Analysis
Uses the same Lasso approach as our DYNOTEARS engine with proper thresholds
"""

import os
import json
import uuid
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )

TOP_EDGES = [
    ("BAMLH0A0HYM2", "BAMLH0A3HYC", "HY spread causes CCC distress"),
    ("^GSPC", "XLF", "S&P 500 causes financials"),
    ("^GSPC", "XLU", "S&P 500 causes utilities"),
    ("DGS10", "T10Y2Y", "10Y Treasury causes yield curve"),
    ("PAYEMS", "UNRATE", "Employment causes unemployment"),
    ("BAMLC0A0CM", "BAMLC0A3CA", "IG spread causes A-rated spread"),
    ("DRTSCIS", "DRTSCILM", "Small firm tightening causes large firm"),
    ("^GSPC", "^NDX", "S&P causes NASDAQ"),
    ("BAMLH0A0HYM2", "BAMLH0A1HYBB", "HY spread causes BB spread"),
    ("DGS10", "DGS2", "10Y causes 2Y Treasury"),
]

CORE_VARS = [
    "^GSPC", "^VIX", "^NDX", "^RUT", "DGS10", "DGS2", "T10Y2Y",
    "FEDFUNDS", "CL=F", "GC=F", "BAMLH0A0HYM2", "BAMLH0A3HYC",
    "BAMLC0A0CM", "XLF", "XLK", "XLE", "XLV", "XLY",
    "TLT", "EEM", "CPIAUCSL", "UNRATE", "PAYEMS",
    "BAMLC0A3CA", "BAMLH0A1HYBB", "DRTSCIS", "DRTSCILM",
]

def load_data():
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
    pivoted = pivoted.sort_index()
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.7))
    pivoted = pivoted.dropna()
    return pivoted

def compute_edges(data, variables, max_lag=2, threshold=0.02):
    """Compute causal edges using Lasso with lagged features (like DYNOTEARS)."""
    values = data[variables].values
    d = len(variables)
    T = len(values)
    
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    standardized = (values - means) / stds
    
    edges = set()
    
    X_current = standardized[max_lag:]
    X_lags = []
    for lag in range(1, max_lag + 1):
        X_lags.append(standardized[max_lag - lag:T - lag])
    X_lag = np.hstack(X_lags)
    
    for j in range(d):
        y = X_current[:, j]
        # Contemporaneous (exclude self)
        contemp_idx = [i for i in range(d) if i != j]
        X_contemp = X_current[:, contemp_idx]
        features = np.hstack([X_contemp, X_lag])
        other_vars = [variables[i] for i in contemp_idx]
        
        try:
            lasso = LassoCV(cv=3, max_iter=10000, n_jobs=-1, random_state=42, tol=0.01)
            lasso.fit(features, y)
            
            n_contemp = len(contemp_idx)
            for idx in range(n_contemp):
                if abs(lasso.coef_[idx]) > threshold:
                    edges.add(f"{other_vars[idx]}->{variables[j]}")
            
            for lag in range(max_lag):
                start = n_contemp + lag * d
                for idx in range(d):
                    if abs(lasso.coef_[start + idx]) > threshold:
                        edges.add(f"{variables[idx]}->{variables[j]}")
        except Exception:
            continue
    
    return edges

print("=" * 70)
print("  EDGE STABILITY ANALYSIS (Fixed)")
print("=" * 70)

data = load_data()
available_vars = [v for v in CORE_VARS if v in data.columns]
analysis_data = data[available_vars].iloc[-3000:]

print(f"\n  Variables: {len(available_vars)}")
print(f"  Observations: {len(analysis_data)}")

# Baseline with all variables
print(f"\n  Computing baseline edges (with lags)...")
baseline_edges = compute_edges(analysis_data, available_vars)
print(f"    Baseline: {len(baseline_edges)} edges found")

# Check which top edges are in baseline
print(f"\n  Top edges in baseline:")
for cause, effect, desc in TOP_EDGES:
    key = f"{cause}->{effect}"
    in_baseline = key in baseline_edges
    print(f"    {key:<35} {'YES' if in_baseline else 'NO'}")

# Leave-one-out stability
edge_stability = {}
for cause, effect, desc in TOP_EDGES:
    key = f"{cause}->{effect}"
    if cause in available_vars and effect in available_vars:
        edge_stability[key] = {
            "desc": desc,
            "in_baseline": key in baseline_edges,
            "survives": 0,
            "total": 0,
        }

# Remove variables that are NOT part of any top edge (true confounders to test)
vars_to_remove = [v for v in available_vars 
                  if v not in [c for c,e,_ in TOP_EDGES]
                  and v not in [e for c,e,_ in TOP_EDGES]]

# Also test a few edge variables
extra_test = ["^VIX", "CL=F", "FEDFUNDS", "GC=F", "TLT"]
vars_to_remove.extend([v for v in extra_test if v in available_vars])
vars_to_remove = list(set(vars_to_remove))[:12]

print(f"\n  Testing removal of {len(vars_to_remove)} variables...")
for i, remove_var in enumerate(vars_to_remove):
    remaining = [v for v in available_vars if v != remove_var]
    reduced_data = analysis_data[remaining]
    
    edges = compute_edges(reduced_data, remaining)
    
    for key, info in edge_stability.items():
        cause, effect = key.split("->")
        if cause in remaining and effect in remaining:
            info["total"] += 1
            if key in edges:
                info["survives"] += 1
    
    print(f"    [{i+1}/{len(vars_to_remove)}] Removed {remove_var}: {len(edges)} edges")

# Results
print(f"\n  {'='*70}")
print(f"  EDGE STABILITY RESULTS")
print(f"  {'='*70}")
print(f"\n  {'Edge':<35} {'Baseline':>10} {'Survival':>12} {'Rate':>8} {'Status':>10}")
print(f"  {'-'*78}")

stable_count = 0
total = 0

for key, info in edge_stability.items():
    if info["total"] == 0:
        continue
    total += 1
    rate = info["survives"] / info["total"] * 100
    baseline_str = "YES" if info["in_baseline"] else "NO"
    status = "STABLE" if rate >= 70 else "MODERATE" if rate >= 40 else "UNSTABLE"
    if rate >= 70:
        stable_count += 1
    print(f"  {key:<35} {baseline_str:>10} {info['survives']}/{info['total']:>9} {rate:>7.0f}% {status:>10}")

pct = stable_count / total * 100 if total > 0 else 0
print(f"\n  Overall: {stable_count}/{total} ({pct:.0f}%) edges are stable (>=70% survival)")

# Combined with FCI result
fci_pct = 80  # From previous run
combined = (pct + fci_pct) / 2
print(f"\n  FCI confounder test: {fci_pct}% robust")
print(f"  Edge stability test: {pct:.0f}% stable")
print(f"  Combined robustness: {combined:.0f}%")

if combined >= 70:
    print(f"\n  VERDICT: Causal edges are ROBUST to potential confounders")
elif combined >= 50:
    print(f"\n  VERDICT: Causal edges are MODERATELY ROBUST")
else:
    print(f"\n  VERDICT: Some sensitivity — acknowledged in paper limitations")

# Store
conn = get_db()
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS models.confounder_analysis (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        analysis_type VARCHAR(50),
        results JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
results = {
    "fci_robustness": fci_pct,
    "stability_pct": pct,
    "combined": combined,
    "edges": {k: {"survival_rate": v["survives"]/max(1,v["total"])*100, "in_baseline": v["in_baseline"]} for k,v in edge_stability.items()},
}
cursor.execute("""
    INSERT INTO models.confounder_analysis (id, analysis_type, results)
    VALUES (%s, %s, %s)
""", (str(uuid.uuid4()), "edge_stability_fixed", Json(results)))
conn.commit()
cursor.close()
conn.close()
print(f"\n  Results stored in database")
print(f"{'='*70}")
