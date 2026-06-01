# Risk Engine

## Overview

The risk engine takes a batch of scenario trajectories (200 × 60 days × 25 variables) and a portfolio specification (asset weights), and computes portfolio-level risk metrics: Value at Risk, Conditional Value at Risk, maximum drawdown, sector decomposition, and marginal risk contributions.

---

## Implementation

| File | Purpose |
|------|---------|
| [ml_pipeline/risk_engine/portfolio.py](ml_pipeline/risk_engine/portfolio.py) | All risk metrics: VaR, CVaR, drawdown, sector decomp, marginal contribution |
| [backend/app/routers/stress_test.py](backend/app/routers/stress_test.py) | REST endpoint wrapping the risk engine, asset-to-variable mapping |

---

## Portfolio Specification

The API accepts holdings as a list of assets with weights. 16 assets are supported:

| Asset Label | Mapped Variable | Category |
|-------------|----------------|----------|
| S&P 500 | ^GSPC | equity |
| Nasdaq 100 | ^NDX | equity |
| Russell 2000 | ^RUT | equity |
| 20Y Treasury | TLT | fixed_income |
| Investment Grade Bonds | LQD | fixed_income |
| High Yield Bonds | HYG | fixed_income |
| Gold | GC=F | commodity |
| Crude Oil | CL=F | commodity |
| Emerging Markets | EEM | international |
| Financials | XLF | sector |
| Tech | XLK | sector |
| Energy | XLE | sector |
| Healthcare | XLV | sector |
| Consumer Discretionary | XLY | sector |
| Utilities | XLU | sector |
| Real Estate | XLRE | sector |

**Four named portfolio profiles** used by the RL adversarial module:

| Profile | Composition |
|---------|-------------|
| `balanced` | 30% SPY (^GSPC proxy), 20% AGG (LQD proxy), 10% XLK, 10% XLF, 10% XLE, 10% XLY |
| `tech_heavy` | 50% ^NDX, 30% XLK, 10% XLY, 10% AGG |
| `bond_heavy` | 60% TLT, 20% LQD, 10% SPY, 10% AGG |
| `credit_heavy` | 40% HYG, 30% LQD, 20% SPY, 10% TLT |

---

## Portfolio Return Computation

For each of the 200 scenario trajectories, the portfolio daily return on day t is:

```
r_portfolio_t = Σ_i w_i × r_i_t

where:
  w_i = normalized weight of asset i
  r_i_t = log-return of the mapped variable on day t
```

Cumulative portfolio return on day t:

```
R_t = Σ_{s=1}^{t} r_portfolio_s    (sum of daily log-returns, approximating cumulative return)
```

This gives a 200 × 60 matrix of cumulative portfolio returns across scenarios and days.

---

## Risk Metrics

### Value at Risk (VaR)

VaR is computed from the distribution of **terminal** (day-60) cumulative returns across the 200 scenarios:

```
VaR_95 = -percentile_5(R_60 across 200 scenarios)
VaR_99 = -percentile_1(R_60 across 200 scenarios)
```

Interpretation: With 95% probability, the portfolio will not lose more than VaR_95 over the 60-day horizon under the stressed scenario distribution.

Note: Terminal return is used here for VaR (day-60 endpoint), while max drawdown captures the worst intermediate point. This distinction matters for portfolios that crash and partially recover.

### Conditional Value at Risk (CVaR / Expected Shortfall)

```
CVaR_95 = -mean(R_60 for scenarios where R_60 < -VaR_95)
```

CVaR answers: given that we are in the worst 5% of scenarios, what is the expected loss? CVaR is a coherent risk measure (subadditive) and is preferred over VaR for regulatory purposes.

### Maximum Drawdown

```
max_drawdown = -min(R_t for t = 1..60)
```

The deepest cumulative loss at any point during the 60-day horizon, averaged across all 200 scenarios. This captures the worst intermediate point, which is more conservative than terminal return — important for portfolios that may experience forced liquidation before recovery.

Note: This is the metric used in the RL adversarial reward function (portfolio_loss). The RL agent was initially trained with terminal return but was corrected to max drawdown after discovering the exploit where portfolios that crashed and recovered showed zero loss.

### Sector Decomposition

The contribution of each sector to total portfolio VaR, computed by grouping assets into categories (equity, fixed_income, commodity, international, sector) and computing their proportional contribution to the total variance under the scenario distribution.

### Marginal Risk Contribution

For each holding i, the marginal contribution to VaR:

```
MRC_i = w_i × (∂VaR / ∂w_i)
```

Approximated numerically by re-computing VaR with a small weight perturbation (±0.01) for each asset. Identifies which single holding adds the most risk to the portfolio.

---

## Example Output (market_crash, balanced portfolio)

```json
{
  "var_95": 0.082,          // 8.2% loss at 95th percentile
  "var_99": 0.121,          // 12.1% loss at 99th percentile
  "cvar_95": 0.097,         // 9.7% expected shortfall given worst 5%
  "max_drawdown": 0.143,    // 14.3% maximum intermediate drawdown
  "sector_decomp": {
    "equity": 0.52,         // 52% of risk from equity holdings
    "sector": 0.31,         // 31% from sector ETFs
    "fixed_income": 0.12,   // 12% from bonds (small due to flight-to-quality)
    "commodity": 0.05       // 5% from commodities
  },
  "marginal_contributions": {
    "^GSPC": 0.31,          // core equity is biggest contributor
    "XLF": 0.18,            // financials amplified in credit scenario
    "XLK": 0.14,
    ...
  }
}
```

---

## How This Feeds the RL Adversarial Module

The RL agent's reward function uses `max_drawdown` as its primary signal:

```python
portfolio_loss = -min(cumsum(daily_returns))   # always ≥ 0
reward = w_pl × portfolio_loss + w_cf × causal_fidelity + w_dv × diversity
```

The four portfolio profiles let the agent learn profile-specific worst cases: the balanced portfolio is most vulnerable to energy-sector pandemic shocks (XLE), the tech-heavy portfolio to Nasdaq/tech crashes (^NDX), bond-heavy to equity contagion via credit linkages, credit-heavy to oil price collapses that trigger energy HY default cascades (CL=F).

---

## Backend Integration

The stress test endpoint ([backend/app/routers/stress_test.py](backend/app/routers/stress_test.py)) handles the full request:

1. Accept `POST /api/v1/stress-test/run` with `{ holdings: [{asset, weight, amount, category}], scenario_id }`.
2. Load the referenced scenario trajectories from `models.scenarios`.
3. Map user-specified asset labels to variable tickers.
4. Pass trajectories and weights to `ml_pipeline.risk_engine.portfolio`.
5. Return VaR, CVaR, drawdown, sector decomp, marginal contributions, plus per-asset stress impact.
