"""
DFAST 2026 Divergence Verification
====================================
Re-computes the gap between our canonical causal model and the REAL Federal
Reserve DFAST 2026 Severely Adverse scenario, to verify the locked paper
numbers:

    DFAST_DIVERGENCES_TOTAL = 34
    DFAST_BBB_YIELD_HIGHER_PCT_RANGE = (17, 26)
    DFAST_TREASURY_HIGHER_PCT_RANGE = (10, 24)

Provenance: Phase 7 of the project loaded the official Fed CSV into
regulatory.scenarios (source="Federal Reserve (Official CSV - Final)").
Earlier runs had a bug where hardcoded approximations were used instead of
the real Fed data. This script verifies the locked numbers came from a run
that used the REAL scenario.

What it does:
  1. Pulls the latest DFAST 2026 scenario from regulatory.scenarios
  2. Confirms source says "Official" / "Final" (not approximated)
  3. Reports key variable values (BBB yield, 10Y Treasury, VIX, etc.)
  4. Loads the most recent causal_difference_reports row for DFAST 2026
  5. Counts total divergences and extracts BBB / Treasury gap ranges
  6. Compares to locked paper numbers and flags any discrepancy
"""

import os
import sys
import json
import warnings
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# LOCKED PAPER NUMBERS (to verify against)
# ============================================================
PAPER_DFAST_DIVERGENCES = 34
PAPER_DFAST_BBB_RANGE = (17, 26)
PAPER_DFAST_TREASURY_RANGE = (10, 24)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def fetch_dfast_scenario():
    """Find the most recent DFAST 2026 scenario in the DB."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT id, name, source, year, scenario_type, description,
               variables, horizon_quarters
        FROM regulatory.scenarios
        WHERE name ILIKE '%%DFAST 2026%%'
        ORDER BY id::text DESC
        LIMIT 5
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def fetch_divergence_reports(scenario_id):
    """Find causal difference reports for a scenario. Column names are plural."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    # Introspect which columns actually exist so we don't SELECT phantoms
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='regulatory' AND table_name='causal_difference_reports'
    """)
    cols = {row["column_name"] for row in cursor.fetchall()}

    wanted = ["id", "regulatory_scenario_id", "causal_projections", "fed_projections",
              "divergences", "metadata", "summary", "created_at"]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        cursor.close(); conn.close()
        return []

    query = f"""
        SELECT {', '.join(select_cols)}
        FROM regulatory.causal_difference_reports
        WHERE regulatory_scenario_id = %s
        ORDER BY created_at DESC
        LIMIT 5
    """
    cursor.execute(query, (str(scenario_id),))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def inspect_schema():
    """Fallback: show what tables/columns exist in the regulatory schema."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'regulatory'
        ORDER BY table_name, ordinal_position
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# ============================================================
# DIVERGENCE EXTRACTION
# ============================================================

def summarize_divergences(divergences):
    """
    divergences can be a list or dict. Returns:
      - total count
      - per-variable max divergence %
      - BBB-related and Treasury-related gap ranges
    """
    # Normalize to a list of records
    if isinstance(divergences, dict):
        records = []
        for var_name, val in divergences.items():
            if isinstance(val, dict):
                rec = {"variable": var_name, **val}
            else:
                rec = {"variable": var_name, "value": val}
            records.append(rec)
    elif isinstance(divergences, list):
        records = divergences
    else:
        return None

    total = len(records)

    # Classify variables and extract percentage divergences
    bbb_pcts, treasury_pcts = [], []
    per_var_max = {}

    for rec in records:
        var_name = rec.get("variable") or rec.get("var") or rec.get("name") or ""
        # Try many possible field names for the gap
        pct = None
        for field in ["pct_difference", "pct_gap", "percentage_gap", "divergence_pct",
                      "pct", "gap_pct", "diff_pct", "percent_diff"]:
            if field in rec and rec[field] is not None:
                try:
                    pct = float(rec[field])
                    break
                except (TypeError, ValueError):
                    continue

        # If no explicit pct, compute from causal vs fed values
        if pct is None:
            causal = rec.get("causal_value") or rec.get("our_projection") or rec.get("causal")
            fed = rec.get("fed_value") or rec.get("fed_projection") or rec.get("fed")
            try:
                causal = float(causal) if causal is not None else None
                fed = float(fed) if fed is not None else None
                if causal is not None and fed is not None and fed != 0:
                    pct = ((causal - fed) / abs(fed)) * 100
            except (TypeError, ValueError):
                pass

        if pct is None:
            continue

        per_var_max[var_name] = max(abs(per_var_max.get(var_name, 0)), abs(pct))

        vname_upper = var_name.upper()
        if "BBB" in vname_upper or "BAMLC0A4" in vname_upper:
            bbb_pcts.append(pct)
        if "TREASURY" in vname_upper or "DGS" in vname_upper or "10Y" in vname_upper or "2Y" in vname_upper:
            treasury_pcts.append(pct)

    bbb_range = (min(bbb_pcts), max(bbb_pcts)) if bbb_pcts else None
    treasury_range = (min(treasury_pcts), max(treasury_pcts)) if treasury_pcts else None

    return {
        "total_divergences": total,
        "bbb_pcts": bbb_pcts,
        "bbb_range": bbb_range,
        "treasury_pcts": treasury_pcts,
        "treasury_range": treasury_range,
        "per_variable_max": per_var_max,
    }


# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 90)
    print("  DFAST 2026 DIVERGENCE VERIFICATION")
    print(f"  Paper numbers to verify: {PAPER_DFAST_DIVERGENCES} divergences, "
          f"BBB={PAPER_DFAST_BBB_RANGE}%, Treasury={PAPER_DFAST_TREASURY_RANGE}%")
    print("=" * 90)

    # ---------------------------------------------------------
    # STEP 1: Find the DFAST 2026 scenario
    # ---------------------------------------------------------
    print("\n  STEP 1: Load DFAST 2026 scenario from regulatory.scenarios")
    print("  " + "-" * 70)

    try:
        scenarios = fetch_dfast_scenario()
    except psycopg2.errors.UndefinedTable as e:
        print(f"  ERROR: regulatory.scenarios table does not exist: {e}")
        print("\n  Dumping regulatory schema structure for debugging:")
        for table, col, dtype in inspect_schema():
            print(f"    {table}.{col} ({dtype})")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR querying scenarios: {type(e).__name__}: {e}")
        sys.exit(1)

    if not scenarios:
        print("  ERROR: No DFAST 2026 scenario found in regulatory.scenarios")
        print("  Phase 7 may not have run, or the scenario was deleted.")
        print("\n  Scenarios matching 'DFAST':")
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, name, source, year FROM regulatory.scenarios ORDER BY year DESC")
        for row in cur.fetchall():
            print(f"    {row}")
        cur.close(); conn.close()
        sys.exit(1)

    print(f"  Found {len(scenarios)} DFAST 2026 scenario(s):\n")
    for i, s in enumerate(scenarios, 1):
        print(f"  Scenario {i}: {s['name']}")
        print(f"    ID:                {s['id']}")
        print(f"    Source:            {s['source']}")
        print(f"    Year:              {s['year']}")
        print(f"    Horizon quarters:  {s['horizon_quarters']}")
        print()

    # Prefer scenarios whose source mentions "Official" or "Final" or "Federal Reserve"
    canonical = None
    for s in scenarios:
        source = str(s["source"]).lower()
        if "official" in source or "final" in source:
            canonical = s
            break
    if canonical is None:
        canonical = scenarios[0]
        print(f"  WARNING: No 'Official' or 'Final' scenario found. Using latest: {canonical['name']}")

    source_flag = "REAL" if ("official" in str(canonical["source"]).lower() or
                              "final" in str(canonical["source"]).lower()) else "APPROXIMATED"
    print(f"  ==> Using canonical scenario (source tagged {source_flag}):")
    print(f"      ID:     {canonical['id']}")
    print(f"      Source: {canonical['source']}")

    # ---------------------------------------------------------
    # STEP 2: Verify key variable values look real
    # ---------------------------------------------------------
    print("\n  STEP 2: Verify scenario variables look like real Fed data")
    print("  " + "-" * 70)

    variables = canonical["variables"] or {}
    print(f"  Variables in scenario: {len(variables)}")

    # Show what keys actually exist so we can debug any misses
    print(f"  Keys present: {sorted(variables.keys())}")

    # Expected Fed 2026 Q1 values (from Phase 7 confirmation):
    #   BBB corporate yield peak: ~8.2%
    #   10Y Treasury peak:        ~4.5%
    #   VIX peak:                 ~72
    # We auto-discover the matching key by substring/code matching — keys may be
    # the Fed's natural label, our FRED code, or our uppercase tag.
    # Each rule: (display label, substring patterns, expected_peak_range)
    expected_rules = [
        ("BBB corporate yield",
         ["BBB", "bbb", "BAMLC0A4"], (5.0, 10.0)),
        ("10-year Treasury yield",
         ["TREASURY_10Y", "DGS10", "10-year", "10Y Treasury", "10-year Treasury"], (2.5, 6.0)),
        ("VIX",
         ["VIX", "MARKET_VOLATILITY", "Volatility Index", "Market Volatility"], (40, 80)),
    ]

    def find_key(patterns):
        """Find the variable key that matches any of the given substrings."""
        for key in variables:
            key_str = str(key)
            for p in patterns:
                if p in key_str or p.lower() in key_str.lower():
                    return key
        return None

    print(f"\n  Sanity checks on key variables:")
    anomalies = []
    for label, patterns, (lo, hi) in expected_rules:
        key = find_key(patterns)
        if key is None:
            print(f"    {label:<30} NOT FOUND (tried patterns: {patterns[:2]}...)")
            continue

        var_data = variables[key]
        # quarterly_path is the usual shape; fall back to the value itself if it's a list
        if isinstance(var_data, dict):
            path = var_data.get("quarterly_path", [])
        elif isinstance(var_data, list):
            path = var_data
        else:
            path = []

        if not path:
            print(f"    {label:<30} key={key!r} has empty path")
            continue

        try:
            path = [float(x) for x in path]
        except (TypeError, ValueError):
            print(f"    {label:<30} key={key!r} path not numeric: {path[:3]}")
            continue

        peak = max(path)
        trough = min(path)
        in_range = lo <= peak <= hi
        flag = "OK   " if in_range else "FLAG "
        print(f"    {label:<30} key={key!r:<25} peak={peak:>6.2f}  trough={trough:>6.2f}  [{flag}] expected peak in [{lo}, {hi}]")
        if not in_range:
            anomalies.append(f"{label}: peak={peak} outside [{lo}, {hi}]")

    if anomalies:
        print(f"\n  WARNING: {len(anomalies)} anomaly/anomalies flagged — may indicate approximated data:")
        for a in anomalies:
            print(f"    - {a}")
    else:
        print(f"\n  All key variables look consistent with real Fed 2026 data.")

    # ---------------------------------------------------------
    # STEP 3: Load divergence reports for this scenario
    # ---------------------------------------------------------
    print("\n  STEP 3: Load causal difference reports for this scenario")
    print("  " + "-" * 70)

    try:
        reports = fetch_divergence_reports(canonical["id"])
    except psycopg2.errors.UndefinedTable as e:
        print(f"  ERROR: regulatory.causal_difference_reports table missing: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR querying divergence reports: {type(e).__name__}: {e}")
        sys.exit(1)

    if not reports:
        print(f"  ERROR: No causal_difference_reports found for scenario {canonical['id']}")
        print(f"  The paper's '34 divergences' number cannot be verified.")
        print(f"  Recommended: re-run the regulatory engine for this scenario.")
        sys.exit(1)

    latest_report = reports[0]
    print(f"  Found {len(reports)} report(s). Using most recent:")
    print(f"    Report ID:  {latest_report['id']}")
    print(f"    Created at: {latest_report['created_at']}")

    # ---------------------------------------------------------
    # STEP 4: Summarize divergences
    # ---------------------------------------------------------
    print("\n  STEP 4: Summarize divergences from the latest report")
    print("  " + "-" * 70)

    divergences_raw = latest_report["divergences"]
    if divergences_raw is None:
        print(f"  ERROR: 'divergences' field is NULL in the report")
        sys.exit(1)

    summary = summarize_divergences(divergences_raw)
    if summary is None:
        print(f"  ERROR: Could not parse divergences field (type={type(divergences_raw).__name__})")
        print(f"  Raw sample: {str(divergences_raw)[:500]}")
        sys.exit(1)

    print(f"\n  Total divergences reported: {summary['total_divergences']}")

    if summary["bbb_range"]:
        lo, hi = summary["bbb_range"]
        print(f"  BBB variable divergences:    [{lo:.1f}%, {hi:.1f}%]  (n={len(summary['bbb_pcts'])})")
    else:
        print(f"  BBB variable divergences:    not found in report")

    if summary["treasury_range"]:
        lo, hi = summary["treasury_range"]
        print(f"  Treasury variable divergences: [{lo:.1f}%, {hi:.1f}%]  (n={len(summary['treasury_pcts'])})")
    else:
        print(f"  Treasury variable divergences: not found in report")

    if summary["per_variable_max"]:
        print(f"\n  Per-variable max divergence (|%|):")
        sorted_vars = sorted(summary["per_variable_max"].items(), key=lambda x: -abs(x[1]))
        for var, mx in sorted_vars[:15]:
            print(f"    {var:<40} {mx:>7.1f}%")
        if len(sorted_vars) > 15:
            print(f"    ... ({len(sorted_vars) - 15} more)")

    # ---------------------------------------------------------
    # STEP 5: Compare to paper numbers
    # ---------------------------------------------------------
    print("\n  STEP 5: Compare to locked paper numbers")
    print("  " + "-" * 70)

    print(f"\n  {'Metric':<30} {'Paper':<15} {'Verified':<15} {'Status'}")
    print("  " + "-" * 70)

    # Divergence count
    matches_count = summary["total_divergences"] == PAPER_DFAST_DIVERGENCES
    status = "MATCH" if matches_count else "MISMATCH"
    print(f"  {'Total divergences':<30} {PAPER_DFAST_DIVERGENCES:<15} "
          f"{summary['total_divergences']:<15} {status}")

    # BBB range
    bbb_ok = False
    if summary["bbb_range"]:
        bbb_verified = f"[{summary['bbb_range'][0]:.1f}, {summary['bbb_range'][1]:.1f}]"
        paper_str = f"[{PAPER_DFAST_BBB_RANGE[0]}, {PAPER_DFAST_BBB_RANGE[1]}]"
        bbb_ok = (abs(summary["bbb_range"][0] - PAPER_DFAST_BBB_RANGE[0]) < 5 and
                  abs(summary["bbb_range"][1] - PAPER_DFAST_BBB_RANGE[1]) < 5)
        status = "MATCH (within 5%)" if bbb_ok else "MISMATCH"
        print(f"  {'BBB divergence range':<30} {paper_str:<15} {bbb_verified:<15} {status}")
    else:
        print(f"  {'BBB divergence range':<30} {'[17, 26]':<15} {'n/a':<15} {'NO DATA'}")

    # Treasury range
    tr_ok = False
    if summary["treasury_range"]:
        tr_verified = f"[{summary['treasury_range'][0]:.1f}, {summary['treasury_range'][1]:.1f}]"
        paper_str = f"[{PAPER_DFAST_TREASURY_RANGE[0]}, {PAPER_DFAST_TREASURY_RANGE[1]}]"
        tr_ok = (abs(summary["treasury_range"][0] - PAPER_DFAST_TREASURY_RANGE[0]) < 5 and
                 abs(summary["treasury_range"][1] - PAPER_DFAST_TREASURY_RANGE[1]) < 5)
        status = "MATCH (within 5%)" if tr_ok else "MISMATCH"
        print(f"  {'Treasury divergence range':<30} {paper_str:<15} {tr_verified:<15} {status}")
    else:
        print(f"  {'Treasury divergence range':<30} {'[10, 24]':<15} {'n/a':<15} {'NO DATA'}")

    # ---------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)

    if source_flag == "APPROXIMATED":
        print(f"\n  WARNING: scenario is tagged APPROXIMATED, not Official/Final.")
        print(f"  The bug from Phase 7 may NOT have been fixed. Paper numbers likely inflated.")
        print(f"  ACTION: re-run the regulatory engine against the real scenario.\n")
    elif anomalies:
        print(f"\n  WARNING: scenario appears REAL but {len(anomalies)} key variable(s) have")
        print(f"  unexpected values. Manual review recommended.\n")
    elif matches_count and bbb_ok and tr_ok:
        print(f"\n  ALL CHECKS PASS. The locked paper numbers (34, 17-26%, 10-24%) match")
        print(f"  the verified divergences from the real Fed DFAST 2026 scenario.\n")
        print(f"  Numbers are paper-ready.\n")
    else:
        print(f"\n  PARTIAL MATCH. Some numbers agree, others drift. Review the table above.\n")
        print(f"  Most likely cause: the report was generated at a different time than the")
        print(f"  headline numbers were recorded. Consider re-running the regulatory engine")
        print(f"  and updating canonical_paper_numbers.py to the verified values.\n")

    print(f"  Suggested paper wording (based on verified data):")
    if summary["bbb_range"] and summary["treasury_range"]:
        bbb_lo, bbb_hi = summary["bbb_range"]
        tr_lo, tr_hi = summary["treasury_range"]
        print(f"    'Applied to the Federal Reserve's DFAST 2026 severely adverse scenario,")
        print(f"     our causal model projects BBB credit yields {bbb_lo:.0f}-{bbb_hi:.0f}% higher")
        print(f"     and Treasury yields {tr_lo:.0f}-{tr_hi:.0f}% higher than the Fed's")
        print(f"     correlation-based methodology, across {summary['total_divergences']} divergence points.'")

    print("=" * 90)

    # Export to JSON for record-keeping
    out = {
        "scenario_id": str(canonical["id"]),
        "scenario_source": canonical["source"],
        "source_flag": source_flag,
        "report_id": str(latest_report["id"]),
        "report_created_at": str(latest_report["created_at"]),
        "verified": {
            "total_divergences": summary["total_divergences"],
            "bbb_range": list(summary["bbb_range"]) if summary["bbb_range"] else None,
            "treasury_range": list(summary["treasury_range"]) if summary["treasury_range"] else None,
        },
        "paper_claims": {
            "total_divergences": PAPER_DFAST_DIVERGENCES,
            "bbb_range": list(PAPER_DFAST_BBB_RANGE),
            "treasury_range": list(PAPER_DFAST_TREASURY_RANGE),
        },
        "all_divergences_verified": matches_count and bbb_ok and tr_ok,
        "anomalies": anomalies,
    }
    out_path = Path(__file__).parent / "dfast_verification.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Verification record: {out_path}")


if __name__ == "__main__":
    run()