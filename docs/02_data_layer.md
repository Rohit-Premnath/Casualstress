# Data Layer

## Overview

CausalStress ingests data from two sources: the Federal Reserve Economic Data (FRED) API and Yahoo Finance. All data is aligned to daily frequency, transformed for stationarity, and stored in a PostgreSQL + TimescaleDB database. The full dataset spans 5,548 trading days from 2005-01-04 to 2026-04-14.

**Sources:** 35 variables from FRED, 21 from Yahoo Finance = 56 total.
**Core variables for scenario generation:** 25 (defined in `CORE_VARIABLES` in [ml_pipeline/generative_engine/scenario_generator.py](ml_pipeline/generative_engine/scenario_generator.py)).
**Key variables for backtest evaluation:** 6 (^GSPC, ^VIX, DGS10, CL=F, XLF, BAMLH0A0HYM2).

---

## Implementation Files

| File | Purpose |
|------|---------|
| [ml_pipeline/data_ingestion/fred_fetcher.py](ml_pipeline/data_ingestion/fred_fetcher.py) | Fetches FRED series via `fredapi`, writes to `raw_fred.observations` |
| [ml_pipeline/data_ingestion/yahoo_fetcher.py](ml_pipeline/data_ingestion/yahoo_fetcher.py) | Fetches price data via `yfinance`, writes to `raw_yahoo.daily_prices` |
| [ml_pipeline/data_ingestion/data_processor.py](ml_pipeline/data_ingestion/data_processor.py) | Aligns, transforms, deduplicates, writes to `processed.time_series_data` |

---

## Stationarity Transforms

Two transform classes are used, assigned per variable:

**Log-returns** (`LOG_RETURN_VARS`): Applied to price-level series. `r_t = log(P_t / P_{t-1})`. Used for equity indices, sectors, ETFs, commodities, currencies, price-based macro series.

**First differences** (`FIRST_DIFF_VARS`): Applied to rate and spread series measured in percentage points or basis points. `Δx_t = x_t - x_{t-1}`. Used for interest rates, credit spreads, lending standards, unemployment.

Missing values: forward-fill up to 5 consecutive trading days, then linear interpolation for longer gaps. Observations with more than 20 consecutive missing days are flagged and excluded from that run.

---

## Variable Inventory

### Equity Indices (5 variables)

| Ticker | Label | Source | Transform | Category |
|--------|-------|--------|-----------|----------|
| ^GSPC | S&P 500 | Yahoo | log-return | equity_index |
| ^NDX | Nasdaq 100 | Yahoo | log-return | equity_index |
| ^RUT | Russell 2000 | Yahoo | log-return | equity_index |
| ^VIX | CBOE Volatility Index | Yahoo | log-return | volatility |
| ^MOVE | MOVE Bond Volatility Index | Yahoo | log-return | volatility |

### US Equity Sectors (8 variables)

| Ticker | Label | Source | Transform |
|--------|-------|--------|-----------|
| XLK | Technology SPDR | Yahoo | log-return |
| XLF | Financials SPDR | Yahoo | log-return |
| XLE | Energy SPDR | Yahoo | log-return |
| XLV | Healthcare SPDR | Yahoo | log-return |
| XLY | Consumer Discretionary SPDR | Yahoo | log-return |
| XLU | Utilities SPDR | Yahoo | log-return |
| XLRE | Real Estate SPDR | Yahoo | log-return |
| XLI | Industrials SPDR | Yahoo | log-return |

### Fixed Income (4 variables)

| Ticker | Label | Source | Transform |
|--------|-------|--------|-----------|
| TLT | iShares 20+ Year Treasury ETF | Yahoo | log-return |
| LQD | iShares Investment Grade Corp ETF | Yahoo | log-return |
| HYG | iShares High Yield Corp ETF | Yahoo | log-return |
| EEM | iShares Emerging Markets ETF | Yahoo | log-return |

### US Treasury Yields and Rates (5 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| DGS10 | 10-Year Treasury Yield | FRED | first-diff |
| DGS2 | 2-Year Treasury Yield | FRED | first-diff |
| T10Y2Y | 10Y−2Y Yield Spread | FRED | first-diff |
| FEDFUNDS | Federal Funds Rate | FRED | first-diff |
| TEDRATE | TED Spread | FRED | first-diff |

### Short-Term Funding Rates (4 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| SOFR | Secured Overnight Financing Rate | FRED | first-diff |
| SOFR90DAYAVG | 90-Day Average SOFR | FRED | first-diff |
| DCPF3M | 3-Month Financial CP Rate | FRED | first-diff |
| DCPN3M | 3-Month Non-Financial CP Rate | FRED | first-diff |

### High Yield Credit Spreads (4 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| BAMLH0A0HYM2 | ICE BofA US HY OAS (Master) | FRED | first-diff |
| BAMLH0A1HYBB | ICE BofA BB HY OAS | FRED | first-diff |
| BAMLH0A2HYB | ICE BofA B HY OAS | FRED | first-diff |
| BAMLH0A3HYC | ICE BofA CCC HY OAS | FRED | first-diff |

### Investment Grade Credit Spreads (5 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| BAMLC0A0CM | ICE BofA US IG Master OAS | FRED | first-diff |
| BAMLC0A1CAAA | ICE BofA AAA IG OAS | FRED | first-diff |
| BAMLC0A2CAA | ICE BofA AA IG OAS | FRED | first-diff |
| BAMLC0A3CA | ICE BofA A IG OAS | FRED | first-diff |
| BAMLC0A4CBBB | ICE BofA BBB IG OAS | FRED | first-diff |

### Bank Lending Standards (4 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| DRTSCILM | C&I Loan Standards: Large/Medium Firms | FRED | first-diff |
| DRTSCIS | C&I Loan Standards: Small Firms | FRED | first-diff |
| DRTSSP | Consumer Loan Standards | FRED | first-diff |
| DRSDCILM | C&I Loan Demand: Large/Medium | FRED | first-diff |

### Macro (Inflation, Labor, Activity) (8 variables)

| Series ID | Label | Source | Transform | Frequency |
|-----------|-------|--------|-----------|-----------|
| CPIAUCSL | CPI All Urban (All Items) | FRED | log-return | Monthly |
| PCEPILFE | Core PCE Price Index | FRED | log-return | Monthly |
| UNRATE | Unemployment Rate | FRED | first-diff | Monthly |
| PAYEMS | Nonfarm Payrolls | FRED | log-return | Monthly |
| INDPRO | Industrial Production | FRED | log-return | Monthly |
| M2SL | M2 Money Supply | FRED | log-return | Monthly |
| HOUST | Housing Starts | FRED | log-return | Monthly |
| RSXFS | Retail Sales ex-Food Service | FRED | log-return | Monthly |

Note: monthly FRED series are forward-filled to daily frequency.

### Financial Stress Indices (2 variables)

| Series ID | Label | Source | Transform |
|-----------|-------|--------|-----------|
| STLFSI4 | St. Louis Fed Financial Stress Index | FRED | first-diff |
| BAMLEMCBPIOAS | EM Broad Composite Spread | FRED | first-diff |

### Commodities and FX (3 variables)

| Ticker | Label | Source | Transform |
|--------|-------|--------|-----------|
| CL=F | Crude Oil WTI Front Month | Yahoo | log-return |
| GC=F | Gold Spot | Yahoo | log-return |
| DX-Y.NYB | US Dollar Index | Yahoo | log-return |

### International and FX Rates (2 variables)

| Ticker | Label | Source | Transform |
|--------|-------|--------|-----------|
| EURUSD=X | EUR/USD Exchange Rate | Yahoo | log-return |
| ICSA | Initial Jobless Claims | FRED | log-return |

---

## The 25 Core Variables

These are the variables included in the VAR model and used for scenario generation. Selected to balance market breadth, causal connectivity, and data availability.

```python
CORE_VARIABLES = [
    "^GSPC", "^VIX", "^NDX", "^RUT",          # equity indices
    "DGS10", "DGS2", "T10Y2Y", "FEDFUNDS",     # rates and curve
    "CL=F", "GC=F",                             # commodities
    "BAMLH0A0HYM2", "BAMLH0A3HYC", "BAMLC0A0CM",  # credit spreads
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLU", # sector ETFs
    "TLT", "LQD", "HYG", "EEM",               # fixed income ETFs
    "CPIAUCSL", "UNRATE",                      # macro
]
```

---

## The 6 Key Backtest Variables

Used for computing coverage, direction, and pairwise metrics in all backtest experiments:

```python
KEY_VARIABLES = ["^GSPC", "^VIX", "DGS10", "CL=F", "XLF", "BAMLH0A0HYM2"]
```

Chosen to represent: equity market direction (^GSPC), fear (^VIX), rates (DGS10), commodity (CL=F), financial sector (XLF), and credit stress (BAMLH0A0HYM2). These six capture the main dimensions of financial crisis dynamics.

---

## Database Schema

All data lands in PostgreSQL (port 5433 in Docker Compose, port 5432 internal):

```sql
-- Raw staging
raw_fred.observations        (series_id, date, value, fetch_timestamp)
raw_yahoo.daily_prices       (ticker, date, open, high, low, close, adj_close, volume)

-- Processed (TimescaleDB hypertable, partitioned by date)
processed.time_series_data   (variable_code, date, raw_value, transformed_value,
                               transform_type, data_source)

-- ML outputs
models.causal_graphs         (id, method, regime, adjacency_matrix jsonb,
                               variables jsonb, created_at)
models.regimes               (date, regime_label, regime_name, probability,
                               transition_probs jsonb)
models.scenarios             (id, model_name, event_type, regime, shock_variable,
                               shock_magnitude, scenario_paths jsonb,
                               plausibility_scores jsonb, created_at)
```

---

## Data Coverage Notes

**Equities and ETFs:** 2003-present on Yahoo Finance (we use from 2005-01-04 for alignment).

**FRED monthly series:** Forward-filled to daily. This means macro variables (CPI, UNRATE) repeat for 20-22 days before a new reading arrives. The VAR model handles this via the stationarity transform — first-differenced UNRATE is mostly zero with a non-zero observation once per month.

**Yield spreads (T10Y2Y):** Available from FRED from 1976. We start at 2005 for consistency with the full panel.

**STLFSI4:** St. Louis Fed Financial Stress Index, updated weekly. Forward-filled to daily.

**TEDRATE:** TED Spread discontinued by FRED in 2023. Post-2023 values are extrapolated from SOFR − T-Bill spread; the regime detection model was validated against the pre-2023 data where TEDRATE is authentic.

**Lending standards (DRTSCILM, DRTSCIS, DRTSSP, DRSDCILM):** Quarterly survey data from the Fed Senior Loan Officer Opinion Survey. Forward-filled for the intervening days. Important: these variables move slowly but are key indicators of regime change — the regime HMM doesn't use them directly but they appear in the causal graph.

---

## Regime-Specific Data Usage

The canonical model is trained only on data from stressed-regime periods:

```python
CANONICAL_TRAIN_REGIMES = ["elevated", "stressed", "high_stress", "crisis"]
```

This means the VAR is estimated from approximately 3,374 observations (the days classified as elevated, stressed, or crisis by the HMM). The model is deliberately not trained on calm and normal periods — the goal is to capture how variables co-move specifically under financial pressure.
