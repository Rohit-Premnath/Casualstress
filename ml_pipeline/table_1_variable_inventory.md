# Table 1: Variable Inventory

56 financial variables used in the causal discovery pipeline, grouped by semantic category.

## Equity Indices (n=4)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `^GSPC` | S&P 500 | Yahoo | daily |
| `^NDX` | Nasdaq 100 | Yahoo | daily |
| `^RUT` | Russell 2000 | Yahoo | daily |
| `EEM` | MSCI Emerging Markets ETF | Yahoo | daily |

## Sector ETFs (n=7)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `XLF` | Financials SPDR | Yahoo | daily |
| `XLK` | Technology SPDR | Yahoo | daily |
| `XLE` | Energy SPDR | Yahoo | daily |
| `XLV` | Healthcare SPDR | Yahoo | daily |
| `XLY` | Consumer Discretionary SPDR | Yahoo | daily |
| `XLU` | Utilities SPDR | Yahoo | daily |
| `XLRE` | Real Estate SPDR | Yahoo | daily |

## Volatility (n=3)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `^VIX` | CBOE Volatility Index | Yahoo | daily |
| `^VVIX` | VIX of VIX | Yahoo | daily |
| `^MOVE` | ICE BofA Treasury MOVE Index | Yahoo | daily |

## Rates (n=9)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `DGS2` | 2-Year Treasury Constant Maturity | FRED | daily |
| `DGS10` | 10-Year Treasury Constant Maturity | FRED | daily |
| `T10Y2Y` | 10Y-2Y Treasury Spread | FRED | daily |
| `FEDFUNDS` | Effective Federal Funds Rate | FRED | monthly |
| `SOFR` | Secured Overnight Financing Rate | FRED | daily |
| `SOFR90DAYAVG` | 90-Day Average SOFR | FRED | daily |
| `DCPF3M` | 3M AA Financial Comm. Paper | FRED | daily |
| `DCPN3M` | 3M AA Nonfinancial Comm. Paper | FRED | daily |
| `TEDRATE` | TED Spread (LIBOR–TBill) | FRED | daily |

## Credit Spreads (n=13)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `BAMLC0A0CM` | IG Corporate Master OAS | FRED (BofA ML) | daily |
| `BAMLC0A1CAAA` | IG AAA Spread | FRED (BofA ML) | daily |
| `BAMLC0A2CAA` | IG AA Spread | FRED (BofA ML) | daily |
| `BAMLC0A3CA` | IG A Spread | FRED (BofA ML) | daily |
| `BAMLC0A4CBBB` | IG BBB Spread | FRED (BofA ML) | daily |
| `BAMLH0A0HYM2` | HY Master Spread | FRED (BofA ML) | daily |
| `BAMLH0A1HYBB` | HY BB Spread | FRED (BofA ML) | daily |
| `BAMLH0A2HYB` | HY B Spread | FRED (BofA ML) | daily |
| `BAMLH0A3HYC` | HY CCC-and-Below Spread | FRED (BofA ML) | daily |
| `BAMLEMCBPIOAS` | EM USD Sovereign OAS | FRED (BofA ML) | daily |
| `HYG` | iShares HY Corporate ETF | Yahoo | daily |
| `LQD` | iShares IG Corporate ETF | Yahoo | daily |
| `TLT` | iShares 20+Y Treasury ETF | Yahoo | daily |

## Commodities (n=2)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `CL=F` | WTI Crude Oil Futures | Yahoo | daily |
| `GC=F` | Gold Futures | Yahoo | daily |

## FX (n=2)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `DX-Y.NYB` | US Dollar Index | Yahoo | daily |
| `EURUSD=X` | EUR/USD Exchange Rate | Yahoo | daily |

## Macro Indicators (n=16)

| Code | Display Name | Source | Freq. |
|------|--------------|--------|-------|
| `CPIAUCSL` | CPI All Urban Consumers | FRED | monthly |
| `PCEPILFE` | Core PCE Price Index | FRED | monthly |
| `UNRATE` | Unemployment Rate | FRED | monthly |
| `PAYEMS` | Nonfarm Payroll Employment | FRED | monthly |
| `INDPRO` | Industrial Production Index | FRED | monthly |
| `ICSA` | Initial Jobless Claims | FRED | weekly |
| `UMCSENT` | Consumer Sentiment (Michigan) | FRED | monthly |
| `HOUST` | Housing Starts | FRED | monthly |
| `M2SL` | M2 Money Supply | FRED | monthly |
| `RSXFS` | Retail Sales ex Food Services | FRED | monthly |
| `A191RL1Q225SBEA` | Real GDP Growth Rate | FRED | quarterly |
| `STLFSI4` | St. Louis Fed Financial Stress | FRED | weekly |
| `DRTSCIS` | Loan Tightening, Small Firms | FRED | quarterly |
| `DRTSCILM` | Loan Tightening, Large/Med Firms | FRED | quarterly |
| `DRTSSP` | Loan Standards, Small Firms | FRED | quarterly |
| `DRSDCILM` | Loan Demand, Large/Med Firms | FRED | quarterly |

**Total: 56 variables**
