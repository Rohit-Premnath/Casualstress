"""
Regime Timeline Data Extract (for Figure 2)
=============================================
Pulls the full HMM regime classification history from models.regimes,
plus the 11 canonical event metadata, and writes regime_timeline.json.

Schema (from hmm_model.py):
  models.regimes columns: id, date, regime_label, regime_name,
                          probability, transition_probs, model_version

Output:
  - regime_timeline.json
    {
      "metadata": {...},
      "timeline": [
        {"date": "2005-01-03", "regime_name": "calm", "regime_label": 0, "probability": 0.98},
        ...
      ],
      "regime_stats": {
        "calm":     {"days": 1174, "pct": 21.2, "color": "#..."},
        ...
      },
      "events": [
        {"name": "2008 GFC", "start": "2008-09-14", "end": "2008-10-28",
         "expected_regime": "crisis", "detected": true, "split": "VAL"},
        ...
      ]
    }

Read-only script — no DB modifications.
"""

import os
import json
import warnings
from pathlib import Path
from collections import Counter

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# EVENT METADATA — matches canonical_paper_numbers.py EVENTS list
# ============================================================
# Each event: (name, start_date, end_date, expected_regime, split)
# expected_regime is what the HMM SHOULD classify this event as
EVENTS_META = [
    ("2008 GFC",              "2008-09-14", "2008-10-28", "crisis",  "VAL"),
    ("2010 Flash Crash",      "2010-05-05", "2010-05-24", "stressed", "VAL"),
    ("2011 US Debt Downgrade","2011-07-28", "2011-08-24", "stressed", "VAL"),
    ("2015 China/Oil Crash",  "2015-08-17", "2015-09-03", "stressed", "VAL"),
    ("2016 Brexit",           "2016-06-23", "2016-07-08", "elevated", "TEST"),
    ("2018 Volmageddon",      "2018-02-02", "2018-02-16", "stressed", "TEST"),
    ("2018 Q4 Selloff",       "2018-10-01", "2018-12-24", "elevated", "TEST"),
    ("2020 COVID",            "2020-02-19", "2020-03-23", "crisis",  "TEST"),
    ("2020 Tech Selloff",     "2020-09-02", "2020-09-24", "elevated", "TEST"),
    ("2022 Rate Hike",        "2022-01-03", "2022-06-17", "elevated", "TEST"),
    ("2023 SVB Crisis",       "2023-03-08", "2023-03-17", "stressed", "TEST"),
]


# Regime color palette (semantic: green=calm -> red=crisis)
REGIME_COLORS = {
    "calm":     "#3a7d44",   # deep green
    "normal":   "#a6d49f",   # pale green
    "elevated": "#f2d98f",   # pale yellow
    "stressed": "#de9b6b",   # orange
    "crisis":   "#b22222",   # deep red
}


def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def fetch_regime_timeline():
    """Pull the full regime classification history, ordered by date."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT date, regime_label, regime_name, probability
        FROM models.regimes
        ORDER BY date ASC
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return rows


def fetch_event_regime_detection(event_start, event_end, strict_stress_set):
    """
    Look up which regimes the HMM assigned during an event window.
    Returns (detected: bool, regime_counts: dict, dominant_regime: str).

    Detection rule (matches paper's REGIME_EVENT_ACCURACY = 72.7%):
      An event is considered "detected as stress" if the DOMINANT regime
      during the event window is in {stressed, crisis}.

    Note: "elevated" is NOT counted as detection, even though it's above
    "calm/normal". This matches the paper's REGIME_CRISIS_SET definition
    and produces the locked 8/11 detection rate.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT regime_name, COUNT(*)
        FROM models.regimes
        WHERE date BETWEEN %s AND %s
        GROUP BY regime_name
    """, (event_start, event_end))
    counts = dict(cursor.fetchall())
    cursor.close(); conn.close()

    # Find dominant regime (most days)
    dominant = None
    if counts:
        dominant = max(counts.items(), key=lambda x: x[1])[0]

    detected = dominant in strict_stress_set if dominant else False
    return detected, counts, dominant


def main():
    print("=" * 80)
    print("  REGIME TIMELINE DATA EXTRACT (for Figure 2)")
    print("=" * 80)

    # Step 1: pull regime timeline
    print("\n  Step 1: Fetching models.regimes timeline...")
    rows = fetch_regime_timeline()
    if not rows:
        print("  ERROR: no rows in models.regimes — HMM hasn't been run?")
        return
    print(f"    Fetched {len(rows):,} daily regime classifications")
    print(f"    Date range: {rows[0]['date']} to {rows[-1]['date']}")

    # Step 2: compute regime stats
    regime_counts = Counter(r["regime_name"] for r in rows)
    total = sum(regime_counts.values())
    regime_stats = {}
    for name, count in regime_counts.most_common():
        regime_stats[name] = {
            "days": count,
            "pct": round(count / total * 100, 1),
            "color": REGIME_COLORS.get(name, "#999999"),
        }

    print(f"\n  Step 2: Regime distribution:")
    for name, stats in regime_stats.items():
        print(f"    {name:<12} {stats['days']:>5,} days  ({stats['pct']:>4.1f}%)")

    # Step 3: event classification check
    # Use STRICT crisis set {stressed, crisis} — matches paper's 8/11 metric.
    # Note: canonical_paper_numbers.py defines both:
    #   REGIME_STRESS_SET = {elevated, stressed, crisis}  (broad)
    #   REGIME_CRISIS_SET = {stressed, crisis}             (strict, used here)
    print(f"\n  Step 3: Per-event HMM detection check")
    print(f"    Detection rule: dominant regime must be in {{stressed, crisis}}")
    print(f"    ('elevated' does NOT count — matches paper's REGIME_CRISIS_SET)\n")
    strict_stress_set = {"stressed", "crisis"}

    events_out = []
    detected_count = 0
    for name, start, end, expected, split in EVENTS_META:
        detected, counts, dominant = fetch_event_regime_detection(
            start, end, strict_stress_set
        )
        if detected:
            detected_count += 1

        events_out.append({
            "name": name,
            "start": start,
            "end": end,
            "expected_regime": expected,
            "dominant_regime": dominant,
            "regime_counts": counts,
            "detected": detected,
            "split": split,
        })
        detected_marker = "MATCH" if detected else "MISS "
        print(f"    [{detected_marker}] {name:<30} dominant={dominant:<10}  expected={expected}")

    print(f"\n  Detection rate: {detected_count}/{len(EVENTS_META)} "
          f"({detected_count / len(EVENTS_META) * 100:.1f}%)")

    # Cross-check against paper's locked number
    PAPER_DETECTION_RATE_COUNT = 8
    PAPER_DETECTION_RATE_PCT = 72.7
    if detected_count == PAPER_DETECTION_RATE_COUNT:
        print(f"  [OK] Matches canonical paper claim: "
              f"{PAPER_DETECTION_RATE_COUNT}/{len(EVENTS_META)} "
              f"({PAPER_DETECTION_RATE_PCT}%)")
    else:
        print(f"  [WARN] Paper claims {PAPER_DETECTION_RATE_COUNT}/{len(EVENTS_META)} "
              f"({PAPER_DETECTION_RATE_PCT}%) — detected count drifted!")

    # Step 4: build timeline payload
    # For JSON size reasons, we don't ship every individual day if there are many.
    # The figure can aggregate into runs at render time.
    timeline = [
        {
            "date": str(r["date"]),
            "regime_name": r["regime_name"],
            "regime_label": int(r["regime_label"]),
            "probability": float(r["probability"]) if r["probability"] else None,
        }
        for r in rows
    ]

    payload = {
        "metadata": {
            "n_days": len(rows),
            "start_date": str(rows[0]["date"]),
            "end_date": str(rows[-1]["date"]),
            "regime_count": len(regime_stats),
        },
        "regime_stats": regime_stats,
        "events": events_out,
        "timeline": timeline,
    }

    out_path = Path(__file__).parent / "regime_timeline.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, default=str)  # no indent — large file
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Wrote: {out_path}  ({size_kb:,.1f} KB)")

    # Step 5: sanity-check against canonical paper numbers
    print(f"\n  Sanity check vs canonical_paper_numbers.py:")
    print(f"    REGIME_COUNT = 5   vs actual {len(regime_stats)}")
    print(f"    REGIME_DISTRIBUTION:")
    # Print canonical comparison for each regime
    # (canonical has specific day counts we can cross-check)
    canonical_dist = {
        "calm":     {"days": 1174, "pct": 21.2},
        "normal":   {"days":  980, "pct": 17.7},
        "elevated": {"days": 1795, "pct": 32.5},
        "stressed": {"days": 1011, "pct": 18.3},
        "crisis":   {"days":  568, "pct": 10.3},
    }
    for name, canon in canonical_dist.items():
        if name in regime_stats:
            actual = regime_stats[name]
            delta_days = actual["days"] - canon["days"]
            delta_pct = actual["pct"] - canon["pct"]
            flag = "OK" if abs(delta_days) < 20 else "DRIFT"
            print(f"      {name:<10} canonical={canon['days']:>5}d ({canon['pct']:>4.1f}%)  "
                  f"actual={actual['days']:>5}d ({actual['pct']:>4.1f}%)  "
                  f"delta={delta_days:+5}d  [{flag}]")
        else:
            print(f"      {name:<10} canonical={canon['days']:>5}d  actual=MISSING")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()