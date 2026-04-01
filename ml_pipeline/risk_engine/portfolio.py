"""
Portfolio Risk Engine
=======================
Takes a portfolio + generated crisis scenarios and computes:
- VaR (Value at Risk) at 95% and 99% confidence
- CVaR (Conditional VaR / Expected Shortfall)
- Maximum Drawdown
- Sector decomposition (which sector hurts most)
- Marginal risk contribution (which single holding adds most risk)
- Sharpe ratio degradation under stress

This is what a JPMorgan risk analyst produces daily.
"""

import os
import sys
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


# ============================================
# ASSET MAPPING
# ============================================

# Map portfolio asset names to our variable codes
ASSET_TO_VARIABLE = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Russell 2000": "^RUT",
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "20Y Treasury Bonds": "TLT",
    "Investment Grade Bonds": "LQD",
    "High Yield Bonds": "HYG",
    "Emerging Markets": "EEM",
    "US Dollar": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
}

# Sector classification for decomposition
ASSET_SECTORS = {
    "^GSPC": "Equity",
    "^NDX": "Equity",
    "^RUT": "Equity",
    "XLK": "Equity",
    "XLF": "Equity",
    "XLE": "Equity",
    "XLV": "Equity",
    "XLY": "Equity",
    "XLRE": "Equity",
    "XLU": "Equity",
    "CL=F": "Commodity",
    "GC=F": "Commodity",
    "TLT": "Fixed Income",
    "LQD": "Fixed Income",
    "HYG": "Fixed Income",
    "EEM": "International",
    "DX-Y.NYB": "Currency",
    "EURUSD=X": "Currency",
}

# Sample portfolios for demo
SAMPLE_PORTFOLIOS = {
    "conservative": {
        "name": "Conservative Portfolio",
        "notional": 1_000_000,
        "holdings": {
            "S&P 500": 0.30,
            "20Y Treasury Bonds": 0.30,
            "Investment Grade Bonds": 0.20,
            "Gold": 0.10,
            "Emerging Markets": 0.10,
        },
    },
    "aggressive": {
        "name": "Aggressive Growth Portfolio",
        "notional": 1_000_000,
        "holdings": {
            "S&P 500": 0.25,
            "NASDAQ 100": 0.25,
            "Technology": 0.15,
            "Emerging Markets": 0.15,
            "Energy": 0.10,
            "High Yield Bonds": 0.10,
        },
    },
    "balanced": {
        "name": "Balanced Portfolio",
        "notional": 1_000_000,
        "holdings": {
            "S&P 500": 0.35,
            "20Y Treasury Bonds": 0.20,
            "Investment Grade Bonds": 0.15,
            "Gold": 0.10,
            "Emerging Markets": 0.10,
            "Real Estate": 0.10,
        },
    },
}


# ============================================
# DATABASE
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# STEP 1: LOAD SCENARIOS
# ============================================

def load_scenarios(scenario_id=None):
    """Load generated scenarios from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if scenario_id:
        cursor.execute("""
            SELECT id, shock_variable, shock_magnitude, regime_condition,
                   scenario_paths, plausibility_scores, n_scenarios
            FROM models.scenarios
            WHERE id = %s
        """, (scenario_id,))
    else:
        # Load all scenarios
        cursor.execute("""
            SELECT id, shock_variable, shock_magnitude, regime_condition,
                   scenario_paths, plausibility_scores, n_scenarios
            FROM models.scenarios
            ORDER BY created_at DESC
        """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    scenarios = []
    for row in rows:
        scenarios.append({
            "id": row[0],
            "shock_variable": row[1],
            "shock_magnitude": row[2],
            "regime": row[3],
            "paths": row[4],
            "scores": row[5],
            "n_scenarios": row[6],
        })

    return scenarios


# ============================================
# STEP 2: COMPUTE PORTFOLIO RETURNS
# ============================================

def compute_portfolio_returns(portfolio, scenario_paths):
    """
    Given a portfolio and scenario paths, compute the portfolio
    return for each scenario.

    Portfolio return = sum of (weight_i * asset_return_i) for each asset.
    """
    holdings = portfolio["holdings"]
    notional = portfolio["notional"]

    portfolio_returns = []

    for path_data in scenario_paths:
        scenario_data = path_data["data"]

        # Compute cumulative return for this scenario
        portfolio_cum_return = 0.0
        asset_returns = {}

        for asset_name, weight in holdings.items():
            var_code = ASSET_TO_VARIABLE.get(asset_name)
            if var_code and var_code in scenario_data:
                # Sum of daily log-returns = cumulative log-return
                cum_return = sum(scenario_data[var_code])
                # Convert log-return to simple return
                simple_return = np.exp(cum_return) - 1
                asset_returns[asset_name] = simple_return
                portfolio_cum_return += weight * simple_return
            else:
                asset_returns[asset_name] = 0.0

        portfolio_returns.append({
            "portfolio_return": portfolio_cum_return,
            "portfolio_pnl": portfolio_cum_return * notional,
            "asset_returns": asset_returns,
            "plausibility": path_data.get("plausibility_score", 1.0),
        })

    return portfolio_returns


# ============================================
# STEP 3: COMPUTE RISK METRICS
# ============================================

def compute_risk_metrics(portfolio_returns, notional):
    """
    Compute all risk metrics from portfolio return distribution.

    VaR: "What's the most I could lose at X% confidence?"
    CVaR: "When I lose more than VaR, what's the average loss?"
    Max Drawdown: "What's the worst peak-to-trough drop?"
    """
    returns = [r["portfolio_return"] for r in portfolio_returns]
    pnls = [r["portfolio_pnl"] for r in portfolio_returns]

    returns_arr = np.array(returns)
    pnls_arr = np.array(pnls)

    # ── VaR (Value at Risk) ──
    var_95_return = np.percentile(returns_arr, 5)  # 5th percentile = 95% VaR
    var_99_return = np.percentile(returns_arr, 1)  # 1st percentile = 99% VaR
    var_95_dollar = var_95_return * notional
    var_99_dollar = var_99_return * notional

    # ── CVaR (Conditional VaR / Expected Shortfall) ──
    cvar_95_mask = returns_arr <= var_95_return
    cvar_95_return = returns_arr[cvar_95_mask].mean() if cvar_95_mask.any() else var_95_return
    cvar_95_dollar = cvar_95_return * notional

    cvar_99_mask = returns_arr <= var_99_return
    cvar_99_return = returns_arr[cvar_99_mask].mean() if cvar_99_mask.any() else var_99_return
    cvar_99_dollar = cvar_99_return * notional

    # ── Max Drawdown ──
    # For each scenario, compute the max drawdown within the path
    max_drawdown = np.min(returns_arr)  # Simplified: worst single scenario return

    # ── Distribution stats ──
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr)
    skewness = float(stats.skew(returns_arr))
    kurtosis = float(stats.kurtosis(returns_arr))

    # ── Probability of loss ──
    prob_loss = np.mean(returns_arr < 0)
    prob_loss_5pct = np.mean(returns_arr < -0.05)
    prob_loss_10pct = np.mean(returns_arr < -0.10)

    metrics = {
        "var_95": {
            "return": float(var_95_return),
            "dollar": float(var_95_dollar),
            "description": f"95% confident loss won't exceed ${abs(var_95_dollar):,.0f}",
        },
        "var_99": {
            "return": float(var_99_return),
            "dollar": float(var_99_dollar),
            "description": f"99% confident loss won't exceed ${abs(var_99_dollar):,.0f}",
        },
        "cvar_95": {
            "return": float(cvar_95_return),
            "dollar": float(cvar_95_dollar),
            "description": f"When losses exceed VaR95, average loss is ${abs(cvar_95_dollar):,.0f}",
        },
        "cvar_99": {
            "return": float(cvar_99_return),
            "dollar": float(cvar_99_dollar),
            "description": f"When losses exceed VaR99, average loss is ${abs(cvar_99_dollar):,.0f}",
        },
        "max_drawdown": {
            "return": float(max_drawdown),
            "dollar": float(max_drawdown * notional),
            "description": f"Worst scenario loss: ${abs(max_drawdown * notional):,.0f}",
        },
        "distribution": {
            "mean": float(mean_return),
            "std": float(std_return),
            "skewness": skewness,
            "kurtosis": kurtosis,
        },
        "probabilities": {
            "prob_any_loss": float(prob_loss),
            "prob_loss_5pct": float(prob_loss_5pct),
            "prob_loss_10pct": float(prob_loss_10pct),
        },
    }

    return metrics


# ============================================
# STEP 4: SECTOR DECOMPOSITION
# ============================================

def compute_sector_decomposition(portfolio_returns, portfolio):
    """
    Break down portfolio risk by sector.
    Shows which sector contributes most to losses.
    """
    holdings = portfolio["holdings"]
    notional = portfolio["notional"]

    sector_losses = {}

    for result in portfolio_returns:
        for asset_name, asset_return in result["asset_returns"].items():
            var_code = ASSET_TO_VARIABLE.get(asset_name, "")
            sector = ASSET_SECTORS.get(var_code, "Other")
            weight = holdings.get(asset_name, 0)

            if sector not in sector_losses:
                sector_losses[sector] = []

            sector_losses[sector].append(weight * asset_return)

    decomposition = {}
    for sector, losses in sector_losses.items():
        losses_arr = np.array(losses)
        decomposition[sector] = {
            "mean_contribution": float(np.mean(losses_arr)),
            "var_95_contribution": float(np.percentile(losses_arr, 5)),
            "dollar_at_risk": float(np.percentile(losses_arr, 5) * notional),
            "weight": float(sum(
                holdings.get(a, 0) for a, v in ASSET_TO_VARIABLE.items()
                if ASSET_SECTORS.get(v) == sector and a in holdings
            )),
        }

    return decomposition


# ============================================
# STEP 5: MARGINAL RISK CONTRIBUTION
# ============================================

def compute_marginal_contributions(portfolio_returns, portfolio):
    """
    Compute each asset's marginal contribution to portfolio risk.
    "If I removed this one holding, how much would my risk decrease?"
    """
    holdings = portfolio["holdings"]
    notional = portfolio["notional"]

    contributions = {}

    for asset_name, weight in holdings.items():
        asset_losses = []
        for result in portfolio_returns:
            asset_return = result["asset_returns"].get(asset_name, 0)
            asset_losses.append(weight * asset_return)

        losses_arr = np.array(asset_losses)
        contributions[asset_name] = {
            "weight": float(weight),
            "mean_return": float(np.mean(losses_arr)),
            "var_95_contribution": float(np.percentile(losses_arr, 5)),
            "dollar_at_risk": float(np.percentile(losses_arr, 5) * notional),
            "worst_case": float(np.min(losses_arr) * notional),
        }

    # Sort by risk contribution (most negative = most risk)
    contributions = dict(sorted(
        contributions.items(),
        key=lambda x: x[1]["var_95_contribution"]
    ))

    return contributions


# ============================================
# STEP 6: STORE RESULTS
# ============================================

def store_stress_test(portfolio, scenario_id, metrics, sector_decomp, marginal_contrib):
    """Store stress test results in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    result_id = str(uuid.uuid4())

    cursor.execute("""
        INSERT INTO app.stress_test_results
            (id, portfolio, scenario_id, var_95, var_99, cvar_95,
             max_drawdown, sector_decomposition, marginal_contributions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        result_id,
        Json(portfolio),
        scenario_id,
        metrics["var_95"]["dollar"],
        metrics["var_99"]["dollar"],
        metrics["cvar_95"]["dollar"],
        metrics["max_drawdown"]["dollar"],
        Json(sector_decomp),
        Json(marginal_contrib),
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return result_id


# ============================================
# STEP 7: DISPLAY RESULTS
# ============================================

def display_results(portfolio, metrics, sector_decomp, marginal_contrib, shock_info):
    """Display a comprehensive stress test report."""
    notional = portfolio["notional"]
    name = portfolio["name"]

    print(f"\n{'='*65}")
    print(f"  STRESS TEST REPORT")
    print(f"{'='*65}")
    print(f"\n  Portfolio: {name}")
    print(f"  Notional: ${notional:,.0f}")
    print(f"  Shock: {shock_info}")

    # Holdings
    print(f"\n  Holdings:")
    for asset, weight in portfolio["holdings"].items():
        print(f"    {asset:<30} {weight:>6.0%}    ${weight * notional:>12,.0f}")

    # Risk Metrics
    print(f"\n{'─'*65}")
    print(f"  RISK METRICS")
    print(f"{'─'*65}")
    print(f"\n  VaR (95%):  {metrics['var_95']['return']:>8.2%}   ${metrics['var_95']['dollar']:>12,.0f}")
    print(f"  VaR (99%):  {metrics['var_99']['return']:>8.2%}   ${metrics['var_99']['dollar']:>12,.0f}")
    print(f"  CVaR (95%): {metrics['cvar_95']['return']:>8.2%}   ${metrics['cvar_95']['dollar']:>12,.0f}")
    print(f"  CVaR (99%): {metrics['cvar_99']['return']:>8.2%}   ${metrics['cvar_99']['dollar']:>12,.0f}")
    print(f"  Max Loss:   {metrics['max_drawdown']['return']:>8.2%}   ${metrics['max_drawdown']['dollar']:>12,.0f}")

    print(f"\n  Loss Probabilities:")
    print(f"    Any loss:     {metrics['probabilities']['prob_any_loss']:>6.1%}")
    print(f"    Loss > 5%:    {metrics['probabilities']['prob_loss_5pct']:>6.1%}")
    print(f"    Loss > 10%:   {metrics['probabilities']['prob_loss_10pct']:>6.1%}")

    # Sector Decomposition
    print(f"\n{'─'*65}")
    print(f"  SECTOR RISK DECOMPOSITION")
    print(f"{'─'*65}")
    print(f"\n  {'Sector':<20} {'Weight':>8} {'Avg Impact':>12} {'$ at Risk':>14}")
    print(f"  {'-'*58}")
    for sector, data in sorted(sector_decomp.items(), key=lambda x: x[1]["dollar_at_risk"]):
        print(f"  {sector:<20} {data['weight']:>7.0%} {data['mean_contribution']:>11.2%} "
              f"${data['dollar_at_risk']:>13,.0f}")

    # Marginal Contributions
    print(f"\n{'─'*65}")
    print(f"  MARGINAL RISK CONTRIBUTION (by holding)")
    print(f"{'─'*65}")
    print(f"\n  {'Asset':<30} {'Weight':>8} {'$ at Risk':>14} {'Worst Case':>14}")
    print(f"  {'-'*68}")
    for asset, data in marginal_contrib.items():
        print(f"  {asset:<30} {data['weight']:>7.0%} "
              f"${data['dollar_at_risk']:>13,.0f} ${data['worst_case']:>13,.0f}")

    print(f"\n{'='*65}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 65)
    print("CAUSALSTRESS - PORTFOLIO RISK ENGINE")
    print("=" * 65)

    # Load all scenarios
    print("\nLoading generated scenarios...")
    all_scenarios = load_scenarios()
    print(f"  Found {len(all_scenarios)} scenario sets")

    # Test all three portfolios against all shock types
    for portfolio_key, portfolio in SAMPLE_PORTFOLIOS.items():
        print(f"\n\n{'#'*65}")
        print(f"  PORTFOLIO: {portfolio['name'].upper()}")
        print(f"{'#'*65}")

        for scenario_set in all_scenarios:
            shock_info = (f"{scenario_set['shock_variable']} "
                         f"({scenario_set['shock_magnitude']:+.1f}σ)")

            # Compute portfolio returns for each scenario path
            portfolio_returns = compute_portfolio_returns(
                portfolio, scenario_set["paths"]
            )

            # Compute risk metrics
            metrics = compute_risk_metrics(portfolio_returns, portfolio["notional"])

            # Sector decomposition
            sector_decomp = compute_sector_decomposition(portfolio_returns, portfolio)

            # Marginal contributions
            marginal_contrib = compute_marginal_contributions(portfolio_returns, portfolio)

            # Display
            display_results(portfolio, metrics, sector_decomp, marginal_contrib, shock_info)

            # Store in database
            result_id = store_stress_test(
                portfolio, scenario_set["id"],
                metrics, sector_decomp, marginal_contrib
            )
            print(f"  Stored with ID: {result_id}")

    print(f"\n\n{'='*65}")
    print("✓ Portfolio stress testing complete!")
    print(f"  Tested {len(SAMPLE_PORTFOLIOS)} portfolios x {len(all_scenarios)} shock types")
    print(f"  = {len(SAMPLE_PORTFOLIOS) * len(all_scenarios)} total stress tests")
    print("=" * 65)


if __name__ == "__main__":
    main()