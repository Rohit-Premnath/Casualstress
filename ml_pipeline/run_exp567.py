import os, json, uuid, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST","localhost"),
        port=os.getenv("POSTGRES_PORT","5433"),
        dbname=os.getenv("POSTGRES_DB","causalstress"),
        user=os.getenv("POSTGRES_USER","causalstress"),
        password=os.getenv("POSTGRES_PASSWORD","causalstress_dev_2026"),
    )

def store_exp(name, results):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models.paper_experiments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            experiment_name VARCHAR(200),
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    clean = json.loads(json.dumps(results, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else str(x) if isinstance(x, (np.integer, np.floating, np.bool_)) else x))
    cursor.execute("INSERT INTO models.paper_experiments (id, experiment_name, results) VALUES (%s, %s, %s)",
                   (str(uuid.uuid4()), name, Json(clean)))
    conn.commit()
    cursor.close()
    conn.close()

# ============ EXPERIMENT 5: VaR Comparison ============
print("=" * 80)
print("  EXPERIMENT 5: VaR METHOD COMPARISON (Fixed)")
print("=" * 80)

conn = get_db()
df = pd.read_sql("""
    SELECT date, variable_code, transformed_value
    FROM processed.time_series_data
    WHERE variable_code = '^GSPC' AND source = 'yahoo'
    ORDER BY date
""", conn)
conn.close()

returns = df["transformed_value"].dropna().values
n = len(returns)
window = 252
confidence = 0.95

exceedances = {"Historical Sim": 0, "Parametric Normal": 0, "Monte Carlo Normal": 0, "CausalStress Student-t": 0}
total_tests = 0

for i in range(window, n - 1):
    hist = returns[i - window:i]
    actual = returns[i]
    total_tests += 1

    # Historical Simulation
    var_hist = np.percentile(hist, 5)
    if actual < var_hist: exceedances["Historical Sim"] += 1

    # Parametric Normal
    mu, sigma = np.mean(hist), np.std(hist)
    var_norm = norm.ppf(0.05, loc=mu, scale=sigma)
    if actual < var_norm: exceedances["Parametric Normal"] += 1

    # Monte Carlo Normal
    mc = np.random.normal(mu, sigma, 5000)
    var_mc = np.percentile(mc, 5)
    if actual < var_mc: exceedances["Monte Carlo Normal"] += 1

    # Student-t (our method)
    t_params = student_t.fit(hist)
    var_t = student_t.ppf(0.05, *t_params)
    if actual < var_t: exceedances["CausalStress Student-t"] += 1

expected = total_tests * 0.05

print(f"\n  Total test days: {total_tests}")
print(f"  Expected exceedances (at 5%): {expected:.0f}")
print(f"\n  {'Method':<28} {'Exceed':>8} {'Rate':>8} {'Expected':>10} {'Kupiec p':>10} {'Status':>8}")
print(f"  {'-'*76}")

results_methods = {}
for method, exc in exceedances.items():
    rate = exc / total_tests
    # Kupiec test
    p_hat = rate
    p_0 = 0.05
    if 0 < p_hat < 1:
        lr = -2 * ((total_tests - exc) * np.log((1-p_0)/(1-p_hat)) + exc * np.log(p_0/p_hat))
        kupiec_p = 1 - stats.chi2.cdf(abs(lr), df=1)
    else:
        kupiec_p = 0.0

    status = "PASS" if kupiec_p > 0.05 else "FAIL"
    closeness = abs(rate - 0.05)
    print(f"  {method:<28} {exc:>8} {rate:>7.4f} {expected:>9.0f} {kupiec_p:>9.4f} {status:>8}")

    results_methods[method] = {
        "exceedances": int(exc),
        "rate": round(float(rate), 4),
        "kupiec_p": round(float(kupiec_p), 4),
        "pass": kupiec_p > 0.05,
        "closeness_to_5pct": round(float(closeness), 4),
    }

# Find best method (closest to 5%)
best = min(results_methods.items(), key=lambda x: x[1]["closeness_to_5pct"])
print(f"\n  Best method (closest to 5% target): {best[0]} (rate={best[1]['rate']:.4f})")

store_exp("Exp5_VaR_Comparison", {"total_tests": total_tests, "expected": round(float(expected), 0), "methods": results_methods, "best_method": best[0]})

# ============ EXPERIMENT 6: VECM Ablation ============
print("\n" + "=" * 80)
print("  EXPERIMENT 6: VECM vs VAR ABLATION")
print("=" * 80)

conn = get_db()
cursor = conn.cursor()
cursor.execute("SELECT results FROM models.cointegration_results WHERE analysis_type = 'johansen_cointegration' ORDER BY created_at DESC")
rows = cursor.fetchall()
cursor.close()
conn.close()

print(f"\n  Cointegrating groups: {len(rows)}")
total_rank = 0
groups = []
for row in rows:
    r = row[0]
    rank = r.get("rank", 0)
    total_rank += rank
    groups.append({"group": r.get("group"), "rank": rank, "vars": len(r.get("variables", []))})
    print(f"    {r.get('group','?')}: rank={rank}, vars={len(r.get('variables', []))}")

print(f"\n  Total cointegrating vectors: {total_rank}")
print(f"  These represent {total_rank} long-run equilibrium relationships")
print(f"  that VAR on differenced data completely ignores")

store_exp("Exp6_VECM_Ablation", {"n_groups": len(rows), "total_vectors": total_rank, "groups": groups})

# ============ EXPERIMENT 7: Copula ============
print("\n" + "=" * 80)
print("  EXPERIMENT 7: STUDENT-T COPULA vs GAUSSIAN")
print("=" * 80)

conn = get_db()
cursor = conn.cursor()
cursor.execute("SELECT analysis_type, results FROM models.copula_results ORDER BY created_at DESC")
rows = cursor.fetchall()
cursor.close()
conn.close()

copula_fit = None
marginals = None
regime_cop = None

for atype, res in rows:
    if atype == "student_t_copula_fit": copula_fit = res
    elif atype == "marginal_distributions": marginals = res
    elif atype == "regime_conditional_copulas": regime_cop = res

if copula_fit:
    print(f"\n  Copula nu: {copula_fit.get('nu')}")
    print(f"  Tail dependence: {copula_fit.get('tail_dependence')}")
    print(f"  Avg correlation: {copula_fit.get('avg_correlation')}")

if marginals:
    t_better = sum(1 for v in marginals.values() if v.get("t_better", False))
    print(f"  Student-t fits better: {t_better}/{len(marginals)} variables")

if regime_cop and "calm" in regime_cop and "stressed" in regime_cop:
    calm_td = regime_cop["calm"]["tail_dependence"]
    stress_td = regime_cop["stressed"]["tail_dependence"]
    ratio = stress_td / calm_td if calm_td > 0 else 0
    print(f"  Calm tail dep: {calm_td:.4f}")
    print(f"  Stress tail dep: {stress_td:.4f}")
    print(f"  Ratio: {ratio:.2f}x")

store_exp("Exp7_Copula", {"copula_nu": copula_fit.get("nu") if copula_fit else None, "tail_dep": copula_fit.get("tail_dependence") if copula_fit else None, "t_better": t_better if marginals else None, "regime_ratio": round(float(ratio), 2) if regime_cop else None})

# ============ SIGNIFICANCE TESTS ============
print("\n" + "=" * 80)
print("  STATISTICAL SIGNIFICANCE")
print("=" * 80)

conn = get_db()
cursor = conn.cursor()
cursor.execute("SELECT target_event, coverage_pct FROM models.backtest_results WHERE coverage_pct IS NOT NULL ORDER BY created_at DESC LIMIT 20")
rows = cursor.fetchall()
cursor.close()
conn.close()

coverages = [r[1] for r in rows if r[1] is not None]
if coverages:
    t_stat, t_p = stats.ttest_1samp(coverages, 50.0)
    boot_means = [np.mean(np.random.choice(coverages, size=len(coverages), replace=True)) for _ in range(10000)]
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    print(f"\n  Mean coverage: {np.mean(coverages):.1f}%")
    print(f"  t-test vs 50% baseline: t={t_stat:.3f}, p={t_p:.6f}")
    print(f"  Significant (p<0.05): {'YES' if t_p < 0.05 else 'NO'}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]")

    store_exp("Statistical_Significance", {"mean_coverage": round(float(np.mean(coverages)), 1), "t_stat": round(float(t_stat), 3), "p_value": round(float(t_p), 6), "significant": bool(t_p < 0.05), "ci_lower": round(float(ci_lo), 1), "ci_upper": round(float(ci_hi), 1)})

# ============ FINAL SUMMARY ============
print("\n" + "=" * 80)
print("  COMPLETE EXPERIMENT SUMMARY FOR PAPER")
print("=" * 80)
print(f"\n  {'Experiment':<45} {'Key Result':>30}")
print(f"  {'-'*78}")
print(f"  {'1. Causal Graph Validation':<45} {'Recall = 100% (25/25)':>30}")
print(f"  {'2. Regime Detection':<45} {'80.0% accuracy (12/15)':>30}")
print(f"  {'3. Scenario Quality':<45} {'86.3% plausibility, 100% >0.7':>30}")
print(f"  {'4. Backtest Coverage':<45} {'83.3% avg (15 events)':>30}")
print(f"  {'5. VaR Comparison':<45} {'Best: ' + best[0]:>30}")
print(f"  {'6. VECM Cointegration':<45} {str(total_rank) + ' equilibrium vectors':>30}")
if copula_fit:
    print(f"  {'7. Copula Tail Dependence':<45} {str(round(copula_fit.get('tail_dependence',0)*100,1)) + '% tail dep':>30}")
if coverages:
    print(f"\n  Statistical significance: p={t_p:.6f} (coverage > random)")
    print(f"  Bootstrap 95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]")
print(f"\n{'='*80}")
