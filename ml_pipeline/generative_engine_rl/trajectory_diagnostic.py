"""
trajectory_diagnostic.py
========================
Post-training diagnostic: what scenarios does the trained adversarial policy find?

Produces:
    1. Action distribution — which variables/families the policy concentrates on
       vs a random baseline, proving the agent has learned to target specific shocks
    2. Top-10 worst episodes ranked by portfolio_loss with decoded actions
    3. 60-day portfolio path figure for the single worst episode found
    4. Sanity-check table: plausibility verdicts for every variable in the
       best episode's trajectory (ensures the scenario is physically realistic)

USAGE
-----
Run on the most recently trained model:
    python -m generative_engine_rl.trajectory_diagnostic

Inspect a specific run:
    python -m generative_engine_rl.trajectory_diagnostic --run-dir runs/<name>

Skip matplotlib figure (text output only):
    python -m generative_engine_rl.trajectory_diagnostic --no-figure

Increase episode count for a richer action distribution sample:
    python -m generative_engine_rl.trajectory_diagnostic --n-episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# VARIABLE TRANSFORM LOOKUP
# ============================================================================
# Mapping from variable code to its transform class. Sourced from
# data_ingestion/data_processor.py (FRED_TRANSFORMS + YAHOO_TRANSFORMS).
# Used to convert daily transform values back into absolute level paths.

VAR_TRANSFORMS: Dict[str, str] = {
    # FRED — log_return (rates of change for prices/levels)
    "INDPRO": "log_return", "CPIAUCSL": "log_return", "PCEPILFE": "log_return",
    "PAYEMS": "log_return", "ICSA": "log_return", "M2SL": "log_return",
    "HOUST": "log_return", "RSXFS": "log_return",
    # FRED — first_diff (level changes for rates and percentages)
    "UNRATE": "first_diff", "FEDFUNDS": "first_diff", "DGS10": "first_diff",
    "DGS2": "first_diff", "T10Y2Y": "first_diff", "UMCSENT": "first_diff",
    "TEDRATE": "first_diff", "BAMLH0A0HYM2": "first_diff",
    "BAMLC0A0CM": "first_diff", "BAMLC0A4CBBB": "first_diff",
    "BAMLC0A3CA": "first_diff", "BAMLC0A2CAA": "first_diff",
    "BAMLC0A1CAAA": "first_diff", "BAMLH0A1HYBB": "first_diff",
    "BAMLH0A2HYB": "first_diff", "BAMLH0A3HYC": "first_diff",
    "BAMLEMCBPIOAS": "first_diff", "DRTSCILM": "first_diff",
    "DRTSCIS": "first_diff", "DRTSSP": "first_diff",
    "DRSDCILM": "first_diff", "SOFR": "first_diff",
    "SOFR90DAYAVG": "first_diff", "DCPF3M": "first_diff",
    "DCPN3M": "first_diff",
    # FRED — none (raw values)
    "A191RL1Q225SBEA": "none",
    # Yahoo — log_return (all are prices/indices)
    "^GSPC": "log_return", "^NDX": "log_return", "^RUT": "log_return",
    "XLK": "log_return", "XLF": "log_return", "XLE": "log_return",
    "XLV": "log_return", "XLI": "log_return", "XLP": "log_return",
    "XLY": "log_return", "XLU": "log_return", "XLB": "log_return",
    "XLRE": "log_return", "TLT": "log_return", "GLD": "log_return",
    "HYG": "log_return", "LQD": "log_return", "GC=F": "log_return",
    "CL=F": "log_return", "BTC-USD": "log_return", "EURUSD=X": "log_return",
    "^VIX": "log_return",
}


# ============================================================================
# REASONABLENESS BANDS FOR SANITY CHECK
# ============================================================================
# For each variable, what's a "plausible" terminal value range over a 60-day
# adversarial scenario? Bands sourced from historical extremes — values inside
# the band are plausible-stress, values outside flag the trajectory as
# unrealistic. Format: (var_code, plausible_min, plausible_max, baseline)
# Bands are deliberately wide — a "fail" verdict means clearly unphysical.

PLAUSIBILITY_BANDS: Dict[str, Dict[str, float]] = {
    # Equity indices: at most -60% drop, at most +30% gain over 60 days
    "^GSPC": {"min_pct": -60.0, "max_pct": 30.0},
    "^NDX":  {"min_pct": -65.0, "max_pct": 35.0},
    "^RUT":  {"min_pct": -65.0, "max_pct": 35.0},
    "XLK":   {"min_pct": -65.0, "max_pct": 35.0},
    "XLF":   {"min_pct": -70.0, "max_pct": 35.0},
    "XLE":   {"min_pct": -75.0, "max_pct": 50.0},
    "XLV":   {"min_pct": -50.0, "max_pct": 30.0},
    "XLU":   {"min_pct": -45.0, "max_pct": 25.0},
    "XLP":   {"min_pct": -40.0, "max_pct": 25.0},
    "XLY":   {"min_pct": -55.0, "max_pct": 35.0},
    "XLI":   {"min_pct": -55.0, "max_pct": 35.0},
    "XLB":   {"min_pct": -60.0, "max_pct": 40.0},
    "XLRE":  {"min_pct": -55.0, "max_pct": 30.0},
    # Safe havens
    "TLT":   {"min_pct": -25.0, "max_pct": 35.0},
    "GLD":   {"min_pct": -20.0, "max_pct": 50.0},
    "HYG":   {"min_pct": -25.0, "max_pct": 15.0},
    "LQD":   {"min_pct": -20.0, "max_pct": 15.0},
    "^VIX":  {"min_pct": -50.0, "max_pct": 400.0},   # VIX can spike massively
    # Rates (absolute deltas in pp)
    "UNRATE":       {"min_diff": -1.0, "max_diff": 12.0},   # COVID hit +11pp
    "FEDFUNDS":     {"min_diff": -5.0, "max_diff": 5.0},
    "DGS10":        {"min_diff": -3.0, "max_diff": 3.0},
    "DGS2":         {"min_diff": -3.0, "max_diff": 3.0},
    "BAMLH0A0HYM2": {"min_diff": -2.0, "max_diff": 15.0},   # spreads widen 100s of bp
    "T10Y2Y":       {"min_diff": -3.0, "max_diff": 3.0},
}


def get_plausibility_band(code: str) -> Optional[Dict[str, float]]:
    """Return plausibility band if available (else None for unbounded vars)."""
    return PLAUSIBILITY_BANDS.get(code)


# ============================================================================
# TRAJECTORY RECONSTRUCTION
# ============================================================================

def reconstruct_levels(
    trajectory: np.ndarray,
    var_codes: List[str],
    initial_levels: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Convert a daily-transform trajectory back into absolute level paths.

    Args:
        trajectory: (horizon, n_vars) — daily transform values
        var_codes: column → code mapping
        initial_levels: starting absolute level per variable code
            (e.g. UNRATE=4.0, ^GSPC=4500). Variables missing here default
            to a sensible value or NaN.

    Returns:
        dict of code → (horizon+1,) array of absolute levels, where index 0
        is the initial level and index horizon is the terminal level.
    """
    horizon = trajectory.shape[0]
    levels: Dict[str, np.ndarray] = {}

    for col, code in enumerate(var_codes):
        transform = VAR_TRANSFORMS.get(code, "none")
        initial = initial_levels.get(code, np.nan)

        path = np.zeros(horizon + 1)
        path[0] = initial

        if transform == "log_return":
            # Each row is log(L_t / L_{t-1}); cumulative product of exp gives
            # L_t = L_0 * exp(cumsum(log_returns))
            cum_log = np.cumsum(trajectory[:, col])
            path[1:] = initial * np.exp(cum_log)
        elif transform == "first_diff":
            # Each row is L_t - L_{t-1}; cumulative sum gives
            # L_t = L_0 + cumsum(diffs)
            cum_diff = np.cumsum(trajectory[:, col])
            path[1:] = initial + cum_diff
        else:  # "none" — raw values
            path[1:] = trajectory[:, col]

        levels[code] = path

    return levels


# ============================================================================
# SANITY-CHECK TABLE
# ============================================================================

def sanity_check_table(
    levels: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Build a row per variable with plausibility verdict."""
    rows = []
    for code, path in levels.items():
        initial = float(path[0])
        terminal = float(path[-1])

        if np.isnan(initial) or np.isnan(terminal):
            verdict = "no_initial"
            move_str = "n/a"
        else:
            band = get_plausibility_band(code)
            transform = VAR_TRANSFORMS.get(code, "none")

            if transform == "log_return":
                pct_change = (terminal / initial - 1.0) * 100 if initial else float("nan")
                move_str = f"{pct_change:+.1f}%"
                if band is None:
                    verdict = "no_band"
                elif "min_pct" in band:
                    if pct_change < band["min_pct"]:
                        verdict = f"BELOW band (min {band['min_pct']:+.0f}%)"
                    elif pct_change > band["max_pct"]:
                        verdict = f"ABOVE band (max {band['max_pct']:+.0f}%)"
                    else:
                        verdict = "plausible"
                else:
                    verdict = "no_band"
            elif transform == "first_diff":
                diff = terminal - initial
                move_str = f"{diff:+.2f}pp"
                if band is None:
                    verdict = "no_band"
                elif "min_diff" in band:
                    if diff < band["min_diff"]:
                        verdict = f"BELOW band (min {band['min_diff']:+.1f}pp)"
                    elif diff > band["max_diff"]:
                        verdict = f"ABOVE band (max {band['max_diff']:+.1f}pp)"
                    else:
                        verdict = "plausible"
                else:
                    verdict = "no_band"
            else:
                move_str = "raw"
                verdict = "no_band"

        rows.append({
            "var_code": code,
            "initial": initial,
            "terminal": terminal,
            "move": move_str,
            "verdict": verdict,
            "transform": VAR_TRANSFORMS.get(code, "none"),
        })
    return rows


def print_sanity_table(rows: List[Dict[str, Any]]) -> None:
    plausible = sum(1 for r in rows if r["verdict"] == "plausible")
    above = sum(1 for r in rows if r["verdict"].startswith("ABOVE"))
    below = sum(1 for r in rows if r["verdict"].startswith("BELOW"))
    no_band = sum(1 for r in rows if r["verdict"] in ("no_band", "no_initial"))
    n = len(rows)

    print()
    print("-" * 84)
    print(f"Trajectory plausibility check ({n} variables)")
    print("-" * 84)
    print(f"  {'code':<16}{'transform':<12}{'initial':>11}{'terminal':>11}{'move':>10}  verdict")
    print(f"  {'-'*16}{'-'*12}{'-'*11}{'-'*11}{'-'*10}  {'-'*30}")
    for r in rows:
        ini = f"{r['initial']:>11.4f}" if not np.isnan(r['initial']) else "        n/a"
        ter = f"{r['terminal']:>11.4f}" if not np.isnan(r['terminal']) else "        n/a"
        print(f"  {r['var_code']:<16}{r['transform']:<12}{ini}{ter}{r['move']:>10}  "
              f"{r['verdict']}")
    print()
    print(f"  Summary: {plausible}/{n} plausible, {above} above band, "
          f"{below} below band, {no_band} unscored")
    if above + below > 0:
        print(f"  WARNING: {above + below} variables outside their plausibility band — "
              f"trajectory may be unrealistic")
    print()


# ============================================================================
# COMPARISON DATA LOADERS
# ============================================================================

def load_covid_unrate_path() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the COVID-era UNRATE trajectory from raw_value in the DB.

    Returns (days_from_start, raw_values) covering Feb 1 2020 through
    Apr 30 2020 (3 months — the canonical COVID shock window).
    Returns None if DB is unreachable.
    """
    try:
        from generative_engine_rl.real_mode_loader import _get_db_connection
        import pandas as pd
    except Exception as e:
        print(f"  [warn] cannot load COVID data: {e}")
        return None

    try:
        conn = _get_db_connection()
        try:
            df = pd.read_sql("""
                SELECT date, raw_value
                FROM processed.time_series_data
                WHERE variable_code = 'UNRATE'
                  AND date BETWEEN '2020-02-01' AND '2020-04-30'
                ORDER BY date
            """, conn)
        finally:
            conn.close()
    except Exception as e:
        print(f"  [warn] DB query for COVID UNRATE failed: {e}")
        return None

    if df.empty:
        print("  [warn] no UNRATE data found in DB for COVID window")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    days = np.arange(len(df))
    raw = df["raw_value"].astype(float).to_numpy()
    return days, raw


def load_dfast_2026_unrate_quarterly() -> List[float]:
    """Return DFAST 2026 severely-adverse UNRATE quarterly_path."""
    try:
        from regulatory.regulatory_engine import DFAST_2026_SEVERELY_ADVERSE
        return list(DFAST_2026_SEVERELY_ADVERSE["variables"]["UNEMPLOYMENT"]["quarterly_path"])
    except Exception as e:
        print(f"  [warn] cannot load DFAST 2026 path: {e}")
        # Fallback to known values from regulatory_engine source
        return [5.5, 7.0, 8.5, 9.5, 10.0, 9.8, 9.5, 9.0, 8.5]


# ============================================================================
# COMPARISON FIGURE
# ============================================================================

def build_comparison_figure(
    agent_unrate: np.ndarray,
    covid_data: Optional[Tuple[np.ndarray, np.ndarray]],
    dfast_quarterly: List[float],
    initial_unrate: float,
    out_path: Path,
    headline_action: str,
) -> None:
    """Build the central paper figure — agent's UNRATE path vs COVID vs DFAST."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib not available: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120)

    horizon_days = len(agent_unrate) - 1    # path[0] is initial, so n-1 days
    days = np.arange(len(agent_unrate))

    # Agent's trajectory (60 trading days)
    ax.plot(days, agent_unrate,
            color="#c0392b", linewidth=2.4,
            label=f"RL agent: {headline_action}",
            zorder=3)

    # COVID 2020 actual
    if covid_data is not None:
        covid_days, covid_vals = covid_data
        # Re-base COVID start to match agent start so the curves are visually
        # comparable (anchor both at t=0)
        covid_days_aligned = covid_days
        ax.plot(covid_days_aligned, covid_vals,
                color="#2c3e50", linewidth=2.0, linestyle="--",
                label="COVID 2020 actual (Feb–Apr)",
                zorder=2)

    # DFAST 2026 quarterly path — convert quarter index → trading day index
    # Each quarter ≈ 63 trading days; we plot at quarter-end day positions.
    # Q0 baseline = initial level (extends agent's t=0 anchor).
    dfast_days = [0] + [i * 63 for i in range(1, len(dfast_quarterly) + 1)]
    dfast_vals = [initial_unrate] + list(dfast_quarterly)
    # Truncate DFAST line to the agent's horizon for visual comparison
    cutoff_quarters = (horizon_days // 63) + 2    # show one extra quarter past horizon
    dfast_days_plot = dfast_days[: cutoff_quarters + 1]
    dfast_vals_plot = dfast_vals[: cutoff_quarters + 1]
    ax.plot(dfast_days_plot, dfast_vals_plot,
            color="#27ae60", linewidth=1.8, linestyle=":",
            marker="o", markersize=6,
            label="DFAST 2026 severely adverse (quarterly)",
            zorder=2)
    # Mark the Q1 threshold horizontal reference
    ax.axhline(y=5.5, color="#27ae60", linewidth=0.8, alpha=0.4, linestyle="-",
               label=None)
    ax.text(horizon_days * 0.97, 5.5, " DFAST Q1 = 5.5%",
            fontsize=8, color="#27ae60", ha="right", va="bottom", alpha=0.8)

    # Mark the trajectory boundary
    ax.axvline(x=horizon_days, color="#7f8c8d", linewidth=0.6, alpha=0.4,
               linestyle="-")
    ax.text(horizon_days, ax.get_ylim()[1] * 0.99, " agent horizon",
            fontsize=8, color="#7f8c8d", ha="left", va="top", alpha=0.7)

    # Annotations
    ax.set_xlabel("Trading days from shock onset", fontsize=11)
    ax.set_ylabel("Unemployment rate (%)", fontsize=11)
    ax.set_title(
        "RL Agent's Discovered Worst-Case vs Historical COVID and DFAST Baseline\n"
        "Unemployment Rate (UNRATE) Trajectory",
        fontsize=12, pad=14,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison figure: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

# ============================================================================
# MULTI-EPISODE COLLECTION
# ============================================================================

def collect_episodes(
    model,
    env,
    n_episodes: int,
    deterministic: bool = True,
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Run n_episodes, returning per-episode records with decoded action + breakdown."""
    from collections import Counter
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=deterministic)
            action = np.asarray(action, dtype=np.int64).flatten()
            decoded = dict(env.decode_action(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("action_rejected"):
            continue
        br = info.get("reward_breakdown")
        episodes.append({
            "ep": ep,
            "target_var": info.get("target_var", decoded.get("target_var", "?")),
            "family_name": info.get("family_name", decoded.get("family_name", "?")),
            "magnitude": float(info.get("magnitude", decoded.get("magnitude", 0.0))),
            "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
            "dfast_breach": float(br.dfast_breach) if br else 0.0,
            "causal_fidelity": float(br.causal_fidelity) if br else 0.0,
            "diversity": float(br.diversity) if br else 0.0,
            "total": float(br.total) if br else float(reward),
            "trajectory": info.get("trajectory"),
            "trajectory_vars": info.get("trajectory_vars", []),
            "initial_state_dict": info.get("initial_state", {}),
        })
    return episodes


def collect_random_episodes(env, n_episodes: int, seed: int = 99999) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    nvec = env.action_space.nvec
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = np.array([
                rng.integers(nvec[0]), rng.integers(nvec[1]), rng.integers(nvec[2])
            ])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("action_rejected"):
            continue
        br = info.get("reward_breakdown")
        decoded = dict(env.decode_action(action)) if not info.get("action_rejected") else {}
        episodes.append({
            "target_var": info.get("target_var", decoded.get("target_var", "?")),
            "family_name": info.get("family_name", decoded.get("family_name", "?")),
            "portfolio_loss": float(br.portfolio_loss) if br else 0.0,
            "total": float(br.total) if br else float(reward),
        })
    return episodes


def print_action_distribution(trained: List[Dict], random: List[Dict]) -> None:
    from collections import Counter
    trained_vars = Counter(ep["target_var"] for ep in trained)
    random_vars = Counter(ep["target_var"] for ep in random)
    n_tr, n_rn = max(len(trained), 1), max(len(random), 1)

    print()
    print("-" * 72)
    print("Action distribution: Trained vs Random policy")
    print("-" * 72)
    print(f"  {'Variable':<20}  {'Trained':>10}  {'Random':>9}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*9}")
    all_vars = sorted(set(list(trained_vars) + list(random_vars)), key=lambda x: -trained_vars.get(x, 0))
    for v in all_vars:
        tp = trained_vars.get(v, 0) / n_tr * 100
        rp = random_vars.get(v, 0) / n_rn * 100
        marker = "  <-- concentrated" if tp > rp * 2.0 and tp > 4 else ""
        print(f"  {v:<20}  {tp:>9.1f}%  {rp:>8.1f}%{marker}")

    trained_fam = Counter(ep["family_name"] for ep in trained)
    random_fam = Counter(ep["family_name"] for ep in random)
    all_fam = sorted(set(list(trained_fam) + list(random_fam)), key=lambda x: -trained_fam.get(x, 0))

    print()
    print(f"  {'Event family':<32}  {'Trained':>10}  {'Random':>9}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*9}")
    for f in all_fam:
        tp = trained_fam.get(f, 0) / n_tr * 100
        rp = random_fam.get(f, 0) / n_rn * 100
        marker = "  <-- concentrated" if tp > rp * 2.0 and tp > 4 else ""
        print(f"  {f:<32}  {tp:>9.1f}%  {rp:>8.1f}%{marker}")


def print_top_episodes(episodes: List[Dict], n: int = 10) -> None:
    ranked = sorted(episodes, key=lambda e: -e["portfolio_loss"])[:n]
    print()
    print("-" * 84)
    print(f"Top {n} episodes by portfolio loss")
    print("-" * 84)
    print(f"  {'#':<4} {'Target Var':<14} {'Family':<28} {'Mag':>6}  {'P/L':>7}  {'CF':>6}  {'Total':>7}")
    print(f"  {'-'*4} {'-'*14} {'-'*28} {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}")
    for i, ep in enumerate(ranked, 1):
        print(
            f"  {i:<4} {ep['target_var']:<14} {ep['family_name']:<28} "
            f"{ep['magnitude']:>+6.2f}  {ep['portfolio_loss']:>7.4f}  "
            f"{ep['causal_fidelity']:>6.4f}  {ep['total']:>7.4f}"
        )


def print_portfolio_path(episode: Dict, portfolio_profile: str = "balanced") -> None:
    traj = episode.get("trajectory")
    traj_vars = episode.get("trajectory_vars", [])
    initial = episode.get("initial_state_dict", {})
    if traj is None or not traj_vars:
        print("  (trajectory data not available)")
        return

    traj = np.asarray(traj)
    var_idx = {v: i for i, v in enumerate(traj_vars)}
    horizon = traj.shape[0]

    from generative_engine_rl.portfolio_model import get_portfolio_weights
    from generative_engine_rl.action_space_loader import load_spec
    pw = get_portfolio_weights(portfolio_profile)
    spec = load_spec()

    display_vars = [v for v in ["^GSPC", "XLF", "XLK", "XLE", "^VIX", "TLT", "UNRATE", "BAMLH0A0HYM2"]
                    if v in var_idx]
    checkpoints = [0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1]

    print()
    print("-" * 84)
    print(f"60-Day trajectory: {episode['target_var']} x {episode['family_name']} x {episode['magnitude']:+.2f}")
    print(f"  portfolio_loss={episode['portfolio_loss']:.4f}  "
          f"causal_fidelity={episode['causal_fidelity']:.4f}  total={episode['total']:.4f}")
    print("-" * 84)
    header = f"  {'Variable':<14}  wt%  {'Day 1':>9}"
    for cp in checkpoints[1:]:
        header += f"  {'Day '+str(cp+1):>9}"
    print(header)
    print(f"  {'-'*14}  {'-'*3}  {'-'*9}" + f"  {'-'*9}" * len(checkpoints[1:]))

    for v in display_vars:
        col = var_idx[v]
        wt = pw.get(v, 0.0) * 100
        cum_lr = np.cumsum(traj[:, col])
        if v in spec.log_return_vars:
            price = 100.0 * np.exp(cum_lr)
            row = f"  {v:<14}  {wt:>3.0f}  {'100.00':>9}"
            for cp in checkpoints[1:]:
                row += f"  {price[cp]:>9.2f}"
        else:
            init = initial.get(v, 0.0)
            level = init + cum_lr
            row = f"  {v:<14}  {wt:>3.0f}  {init:>9.4f}"
            for cp in checkpoints[1:]:
                row += f"  {level[cp]:>9.4f}"
        print(row)

    # Portfolio composite
    port_lr = np.zeros(horizon)
    for asset, weight in pw.items():
        if asset in var_idx and asset in spec.log_return_vars:
            port_lr += weight * np.cumsum(traj[:, var_idx[asset]])
    port_idx = 100.0 * np.exp(port_lr)
    worst_day = int(np.argmin(port_idx))
    print(f"  {'-'*14}  {'-'*3}  {'-'*9}" + f"  {'-'*9}" * len(checkpoints[1:]))
    row = f"  {'PORTFOLIO':<14}  100  {'100.00':>9}"
    for cp in checkpoints[1:]:
        row += f"  {port_idx[cp]:>9.2f}"
    print(row)
    print(f"\n  Worst day: Day {worst_day+1}, index = {port_idx[worst_day]:.2f}  "
          f"({port_idx[worst_day]-100:.1f}%)")


def build_portfolio_path_figure(
    episodes: List[Dict],
    portfolio_profile: str,
    out_path: Path,
) -> None:
    """Plot the portfolio index path for the top-3 worst episodes + random baseline."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib not available: {e}")
        return

    from generative_engine_rl.portfolio_model import get_portfolio_weights
    from generative_engine_rl.action_space_loader import load_spec
    pw = get_portfolio_weights(portfolio_profile)
    spec = load_spec()

    top3 = sorted(episodes, key=lambda e: -e["portfolio_loss"])[:3]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)
    colors = ["#c0392b", "#e67e22", "#8e44ad"]

    for rank, (ep, color) in enumerate(zip(top3, colors), 1):
        traj = ep.get("trajectory")
        traj_vars = ep.get("trajectory_vars", [])
        if traj is None or not traj_vars:
            continue
        traj = np.asarray(traj)
        var_idx = {v: i for i, v in enumerate(traj_vars)}
        horizon = traj.shape[0]
        port_lr = np.zeros(horizon)
        for asset, weight in pw.items():
            if asset in var_idx and asset in spec.log_return_vars:
                port_lr += weight * np.cumsum(traj[:, var_idx[asset]])
        port_idx = 100.0 * np.exp(port_lr)
        days = np.arange(1, horizon + 1)
        label = (f"#{rank}: {ep['target_var']} x {ep['family_name'][:20]}"
                 f" x {ep['magnitude']:+.1f}  (P/L={ep['portfolio_loss']:.3f})")
        ax.plot(days, port_idx, color=color, linewidth=2.2, label=label, zorder=3)
        ax.scatter([int(np.argmin(port_idx)) + 1], [float(np.min(port_idx))],
                   color=color, s=80, zorder=4)

    ax.axhline(y=100, color="#95a5a6", linewidth=0.9, linestyle="--", alpha=0.7, label="Baseline (100)")
    ax.axhline(y=70, color="#e74c3c", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.text(1, 70.5, " -30% level", fontsize=8, color="#e74c3c", alpha=0.7)

    ax.set_xlabel("Trading day", fontsize=11)
    ax.set_ylabel("Portfolio index (start = 100)", fontsize=11)
    ax.set_title(
        f"RL Agent: Top-3 Worst-Case Scenarios\n"
        f"Portfolio: {portfolio_profile}  (portfolio_adversarial reward)",
        fontsize=12, pad=12,
    )
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved portfolio path figure: {out_path}")


def find_latest_run() -> Path:
    # Prefer ml_pipeline/runs/ (canonical location), fall back to repo root runs/
    for runs_dir in [ROOT / "runs", ROOT.parent / "runs"]:
        if runs_dir.exists():
            candidates = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "final.zip").exists()]
            if candidates:
                return max(candidates, key=lambda d: d.stat().st_mtime)
    raise FileNotFoundError("No run with final.zip found in ml_pipeline/runs/ or runs/.")


def find_model_path(run_dir: Path) -> Path:
    for c in [run_dir / "final.zip", run_dir / "best" / "best_model.zip"]:
        if c.exists():
            return c
    ckpts = sorted(run_dir.glob("ckpt_*.zip"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No model file in {run_dir}")


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None,
                        help="Run directory to inspect. Default: latest in ml_pipeline/runs/.")
    parser.add_argument("--mode", choices=["fast", "real"], default=None,
                        help="Env mode. Default: read from run config.")
    parser.add_argument("--seed", type=int, default=20000,
                        help="Base seed for episode collection.")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of episodes to collect for action distribution.")
    parser.add_argument("--unrate-baseline", type=float, default=4.0,
                        help="Initial UNRATE level for plausibility anchor.")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip matplotlib figures (text output only).")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample stochastically from the policy distribution (not argmax). "
                             "Reveals the full range of scenarios the policy considers high-value.")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from generative_engine_rl.action_space_loader import load_spec
    from generative_engine_rl.env_factory import make_env

    # ---- Locate run + model ----
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run()
    model_path = find_model_path(run_dir)

    run_cfg = load_run_config(run_dir)
    mode = args.mode if args.mode is not None else run_cfg.get("mode", "fast")
    portfolio_profile = run_cfg.get("portfolio_profile", "balanced")
    actions_per_episode = int(run_cfg.get("actions_per_episode", 1))
    use_family_templates = bool(run_cfg.get("use_family_templates", True))
    reward_mode = run_cfg.get("reward_mode", "portfolio_adversarial")

    print()
    print("=" * 84)
    print("Trajectory Diagnostic")
    print("=" * 84)
    print(f"  run_dir:     {run_dir.name}")
    print(f"  model:       {model_path.name}")
    print(f"  mode:        {mode}")
    print(f"  portfolio:   {portfolio_profile}")
    print(f"  n_episodes:  {args.n_episodes}")
    print(f"  reward_mode: {reward_mode}")
    print()

    # ---- Build env + load model ----
    print("Loading env...")
    spec = load_spec()
    env = make_env(
        mode=mode,
        seed=args.seed,
        spec=spec,
        portfolio_profile=portfolio_profile,
        actions_per_episode=actions_per_episode,
        use_family_templates=use_family_templates,
        reward_mode=reward_mode,
    )
    print("Loading model...")
    model = PPO.load(str(model_path))
    print(f"  action space: MultiDiscrete{list(env.action_space.nvec)}")

    # ---- Collect multiple episodes for action distribution ----
    deterministic = not args.stochastic
    mode_label = "stochastic" if args.stochastic else "deterministic"
    print(f"\nCollecting {args.n_episodes} trained-policy episodes ({mode_label})...")
    trained_eps = collect_episodes(model, env, args.n_episodes, deterministic=deterministic, seed_offset=args.seed)

    print(f"Collecting {args.n_episodes} random-policy episodes...")
    random_eps = collect_random_episodes(env, args.n_episodes, seed=args.seed + 99999)

    # ---- Summary stats ----
    tr_pl = [e["portfolio_loss"] for e in trained_eps]
    rn_pl = [e["portfolio_loss"] for e in random_eps]
    print()
    print("=" * 84)
    print("Summary")
    print("=" * 84)
    print(f"  Trained  portfolio_loss: mean={np.mean(tr_pl):+.4f}  "
          f"max={np.max(tr_pl):+.4f}  nonzero={np.mean(np.array(tr_pl)>0)*100:.0f}%")
    print(f"  Random   portfolio_loss: mean={np.mean(rn_pl):+.4f}  "
          f"max={np.max(rn_pl):+.4f}  nonzero={np.mean(np.array(rn_pl)>0)*100:.0f}%")
    print(f"  Trained mean total reward: {np.mean([e['total'] for e in trained_eps]):+.4f}")
    print(f"  Random  mean total reward: {np.mean([e['total'] for e in random_eps]):+.4f}")

    # ---- Action distribution ----
    print_action_distribution(trained_eps, random_eps)

    # ---- Top-10 episodes ----
    print_top_episodes(trained_eps, n=10)

    # ---- Best episode: portfolio path table + sanity check ----
    best = max(trained_eps, key=lambda e: e["portfolio_loss"])
    print_portfolio_path(best, portfolio_profile=portfolio_profile)

    traj = best.get("trajectory")
    traj_vars = best.get("trajectory_vars", [])
    if traj is not None and traj_vars:
        initial_levels = {
            "^GSPC": 4500.0, "^NDX": 16000.0, "^RUT": 2000.0,
            "XLK": 200.0, "XLF": 40.0, "XLE": 90.0, "XLV": 140.0,
            "XLU": 75.0, "XLP": 80.0, "XLY": 175.0, "XLI": 130.0,
            "XLB": 90.0, "XLRE": 40.0, "TLT": 95.0, "GLD": 200.0,
            "HYG": 78.0, "LQD": 105.0, "GC=F": 2000.0, "CL=F": 75.0,
            "^VIX": 18.0, "UNRATE": args.unrate_baseline, "FEDFUNDS": 4.5,
            "DGS10": 4.2, "DGS2": 4.5, "T10Y2Y": -0.3,
            "BAMLH0A0HYM2": 3.5, "BAMLC0A0CM": 1.2, "BAMLH0A3HYC": 8.0,
            "SOFR": 4.3, "DCPF3M": 4.5, "DCPN3M": 4.5,
            "INDPRO": 102.0, "CPIAUCSL": 305.0, "PAYEMS": 156000.0,
        }
        levels = reconstruct_levels(
            trajectory=np.asarray(traj),
            var_codes=traj_vars,
            initial_levels=initial_levels,
        )
        rows = sanity_check_table(levels)
        print_sanity_table(rows)

    # ---- Save results ----
    out_dir = run_dir / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(run_dir),
        "model": str(model_path),
        "n_episodes": args.n_episodes,
        "portfolio_profile": portfolio_profile,
        "trained_mean_portfolio_loss": float(np.mean(tr_pl)),
        "trained_max_portfolio_loss": float(np.max(tr_pl)),
        "random_mean_portfolio_loss": float(np.mean(rn_pl)),
        "trained_mean_total": float(np.mean([e["total"] for e in trained_eps])),
        "random_mean_total": float(np.mean([e["total"] for e in random_eps])),
        "top_10_episodes": [
            {k: v for k, v in ep.items() if k not in ("trajectory", "trajectory_vars", "initial_state_dict")}
            for ep in sorted(trained_eps, key=lambda e: -e["portfolio_loss"])[:10]
        ],
    }
    diag_path = out_dir / "diagnostic_summary.json"
    diag_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Results saved: {diag_path}")

    # ---- Portfolio path figure ----
    if not args.no_figure:
        print("\nBuilding portfolio path figure...")
        build_portfolio_path_figure(
            episodes=trained_eps,
            portfolio_profile=portfolio_profile,
            out_path=out_dir / "portfolio_path_top3.png",
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
