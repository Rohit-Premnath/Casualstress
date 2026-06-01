"""
Crisis Alignment Scan
=====================
Scans seeds to find which historical starting states land near known crisis
dates, then runs the bandit on those seeds to check whether the adversarial
pathways it discovers align with what actually happened historically.

Three crisis periods:
  1. Oct 2008  — Lehman collapse, global financial crisis peak
  2. Mar 2020  — COVID crash (fastest 30% drawdown in S&P history)
  3. Oct 2022  — Rate shock peak (Fed most aggressive hiking cycle in 40yrs)

Run from repo root:
    cd ml_pipeline
    python ../docs/paper/01_crisis_alignment_scan.py

Output: docs/paper/02_crisis_alignment_results.md (auto-written)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

ML_ROOT = Path(__file__).resolve().parent.parent / "ml_pipeline"
sys.path.insert(0, str(ML_ROOT))

CRISIS_DATES = {
    "GFC_peak":       datetime(2008, 10, 10),   # Lehman +1 month, VIX hit 80
    "COVID_crash":    datetime(2020, 3, 20),     # S&P trough −34% from Feb high
    "Rate_shock_22":  datetime(2022, 10, 14),   # 10Y yield hit 4.1%, bond worst yr
}

PROFILES = ["bond_heavy", "tech_heavy", "balanced", "credit_heavy"]
SCAN_SEEDS = range(50000, 50500)     # clear of training/eval/inference ranges
TOP_N = 3                            # closest seeds per crisis date

CAUSAL_CONTEXT = {
    "GFC_peak": (
        "Lehman Brothers filed Ch.11 on Sep 15 2008. By Oct 10, VIX hit 80, "
        "interbank credit markets frozen. Expected bandit: credit spread shocks "
        "(HYG/LQD/BAMLH), financial sector contagion (XLF). "
        "Key variables: BAMLH0A0HYM2 (HY spreads), XLF, ^GSPC, TLT (flight-to-safety bid)."
    ),
    "COVID_crash": (
        "S&P 500 fell 34% in 33 days (Feb 19 – Mar 23 2020). Volatility spike: "
        "VIX >80. Broad equity selloff, credit spreads blew out, then massive "
        "Fed intervention. Expected bandit: broad equity shocks (^GSPC, XLK, XLY), "
        "VIX spike, credit spread widening."
    ),
    "Rate_shock_22": (
        "Fed hiked 425bp in 2022 — fastest cycle in 40 years. 20Y Treasury "
        "(TLT) fell ~40% YTD by Oct 2022. Both bonds AND equities down (unusual). "
        "Expected bandit (bond_heavy portfolio): TLT shock → bank/financial "
        "contagion (XLF), because rising rates compress bank NII and mark bond "
        "portfolios to market. Expected for tech_heavy: rate-sensitive growth "
        "stocks (QQQ/XLK) re-rated sharply."
    ),
}


def parse_date(d) -> datetime:
    if d is None:
        return None
    try:
        if isinstance(d, str):
            # handles "2008-10-10 00:00:00" and "2008-10-10"
            return datetime.strptime(d.split(" ")[0], "%Y-%m-%d")
        return datetime(d.year, d.month, d.day)
    except Exception:
        return None


def days_diff(d1: datetime, d2: datetime) -> int:
    if d1 is None or d2 is None:
        return 9999
    return abs((d1 - d2).days)


def scan_seeds(engine, seeds):
    """Return {seed: date_str} by running one bandit step per seed."""
    from generative_engine_rl.neural_bandit import bandit_sequence
    seed_dates = {}
    for seed in seeds:
        result = bandit_sequence(engine.net, engine.catalog, engine.env,
                                 seed, ucb_beta=0.0)  # greedy, fastest
        state = result.get("sampled_state") or {}
        raw = state.get("date")
        dt = parse_date(raw)
        seed_dates[seed] = dt
    return seed_dates


def find_closest_seeds(seed_dates: dict, target: datetime, top_n: int):
    scored = [(s, d, days_diff(d, target))
              for s, d in seed_dates.items() if d is not None]
    scored.sort(key=lambda x: x[2])
    return scored[:top_n]


def run_alignment(engine, profile, best_seeds):
    """Run bandit on best seeds and return structured results."""
    from generative_engine_rl.neural_bandit import bandit_sequence
    results = []
    for seed, date, gap_days in best_seeds:
        r = bandit_sequence(engine.net, engine.catalog, engine.env,
                            seed, ucb_beta=0.5)
        seq_str = " → ".join(
            f"{s['target_var']} {s['magnitude']:+.1f}σ" for s in r["sequence"]
        )
        results.append({
            "seed": seed,
            "date": date,
            "gap_days": gap_days,
            "sequence": seq_str,
            "portfolio_loss": r["portfolio_loss"],
            "causal_fidelity": r["causal_fidelity"],
        })
    return results


def main():
    from generative_engine_rl.adversarial_serve import (
        load_all_engines, get_engine
    )

    print("Loading engines...")
    load_all_engines(PROFILES)

    # Use bond_heavy engine for the seed scan (same dataset for all profiles)
    primary_engine = get_engine("bond_heavy")

    print(f"Scanning {len(list(SCAN_SEEDS))} seeds to find dates...")
    seed_dates = scan_seeds(primary_engine, SCAN_SEEDS)
    found = sum(1 for d in seed_dates.values() if d is not None)
    print(f"  Found dates for {found}/{len(list(SCAN_SEEDS))} seeds")

    lines = [
        "# Crisis Alignment Results",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Seeds scanned:** {SCAN_SEEDS.start}–{SCAN_SEEDS.stop - 1} ({len(list(SCAN_SEEDS))} total)",
        f"**Profiles:** {', '.join(PROFILES)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "For each crisis period, the 3 seeds whose historical starting date is",
        "closest to the crisis date are selected. The bandit is then run on those",
        "seeds. If the pathway it discovers matches the historically observed",
        "transmission mechanism, that is evidence of historical-crisis alignment.",
        "",
    ]

    for crisis_name, crisis_dt in CRISIS_DATES.items():
        print(f"\n--- {crisis_name} ({crisis_dt.date()}) ---")
        closest = find_closest_seeds(seed_dates, crisis_dt, TOP_N)
        if not closest:
            lines += [f"## {crisis_name}", "No seeds found near this date.", ""]
            continue

        lines += [
            f"## {crisis_name} — {crisis_dt.strftime('%b %Y')}",
            "",
            f"**Historical context:** {CAUSAL_CONTEXT[crisis_name]}",
            "",
            "### Closest seeds",
            "",
            "| Seed | Sampled Date | Gap (days) |",
            "|---|---|---|",
        ]
        for seed, date, gap in closest:
            lines.append(f"| {seed} | {date.date()} | {gap} |")
        lines.append("")

        lines += [
            "### Bandit pathways by profile",
            "",
            "| Profile | Bandit sequence | Portfolio loss | Causal fidelity |",
            "|---|---|:---:|:---:|",
        ]

        for profile in PROFILES:
            engine = get_engine(profile)
            results = run_alignment(engine, profile, closest[:1])  # use single closest seed
            for r in results:
                lines.append(
                    f"| {profile} | {r['sequence']} "
                    f"| {r['portfolio_loss']:.3f} | {r['causal_fidelity']:.3f} |"
                )
        lines += [
            "",
            "### Alignment verdict",
            "",
            "<!-- Fill this in manually after reviewing the table above -->",
            "<!-- Does the bond_heavy bandit fire TLT-type shocks for Rate_shock_22? -->",
            "<!-- Does tech_heavy fire XLK/QQQ shocks for COVID_crash? -->",
            "",
            "---",
            "",
        ]

    out_path = Path(__file__).parent / "02_crisis_alignment_results.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
