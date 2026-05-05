"""
Table 1: Variable Inventory
============================
Lists all 56 financial variables used in the causal discovery pipeline,
grouped by 8 semantic categories with display names, sources, and FRED IDs.

Outputs:
  - table_1_variable_inventory.tex   (LaTeX booktabs, ready for paper)
  - table_1_variable_inventory.md    (Markdown for README/docs)
  - table_1_variable_inventory.csv   (raw data)

Data source:
  - CATEGORIES dict (same as causal_graph_extract.py)
  - Hard-coded display name + source mapping (metadata not in DB)

No DB queries needed. This is a static metadata table.
"""

from pathlib import Path


# ============================================================
# VARIABLE METADATA
# ============================================================
# Each entry: (code, display_name, source, frequency, notes)
# Source values: "FRED", "Yahoo", "FRED (BofA ML)" for BofA indices via FRED

VARIABLES = [
    # ---------- Equity Indices ----------
    ("^GSPC",      "S&P 500",                          "Yahoo",  "daily"),
    ("^NDX",       "Nasdaq 100",                       "Yahoo",  "daily"),
    ("^RUT",       "Russell 2000",                     "Yahoo",  "daily"),
    ("EEM",        "MSCI Emerging Markets ETF",        "Yahoo",  "daily"),

    # ---------- Sector ETFs ----------
    ("XLF",        "Financials SPDR",                  "Yahoo",  "daily"),
    ("XLK",        "Technology SPDR",                  "Yahoo",  "daily"),
    ("XLE",        "Energy SPDR",                      "Yahoo",  "daily"),
    ("XLV",        "Healthcare SPDR",                  "Yahoo",  "daily"),
    ("XLY",        "Consumer Discretionary SPDR",      "Yahoo",  "daily"),
    ("XLU",        "Utilities SPDR",                   "Yahoo",  "daily"),
    ("XLRE",       "Real Estate SPDR",                 "Yahoo",  "daily"),

    # ---------- Volatility ----------
    ("^VIX",       "CBOE Volatility Index",            "Yahoo",  "daily"),
    ("^VVIX",      "VIX of VIX",                       "Yahoo",  "daily"),
    ("^MOVE",      "ICE BofA Treasury MOVE Index",     "Yahoo",  "daily"),

    # ---------- Rates ----------
    ("DGS2",       "2-Year Treasury Constant Maturity","FRED",   "daily"),
    ("DGS10",      "10-Year Treasury Constant Maturity","FRED",  "daily"),
    ("T10Y2Y",     "10Y-2Y Treasury Spread",           "FRED",   "daily"),
    ("FEDFUNDS",   "Effective Federal Funds Rate",     "FRED",   "monthly"),
    ("SOFR",       "Secured Overnight Financing Rate", "FRED",   "daily"),
    ("SOFR90DAYAVG","90-Day Average SOFR",             "FRED",   "daily"),
    ("DCPF3M",     "3M AA Financial Comm. Paper",      "FRED",   "daily"),
    ("DCPN3M",     "3M AA Nonfinancial Comm. Paper",   "FRED",   "daily"),
    ("TEDRATE",    "TED Spread (LIBOR\u2013TBill)",    "FRED",   "daily"),

    # ---------- Credit Spreads ----------
    ("BAMLC0A0CM",      "IG Corporate Master OAS",     "FRED (BofA ML)", "daily"),
    ("BAMLC0A1CAAA",    "IG AAA Spread",               "FRED (BofA ML)", "daily"),
    ("BAMLC0A2CAA",     "IG AA Spread",                "FRED (BofA ML)", "daily"),
    ("BAMLC0A3CA",      "IG A Spread",                 "FRED (BofA ML)", "daily"),
    ("BAMLC0A4CBBB",    "IG BBB Spread",               "FRED (BofA ML)", "daily"),
    ("BAMLH0A0HYM2",    "HY Master Spread",            "FRED (BofA ML)", "daily"),
    ("BAMLH0A1HYBB",    "HY BB Spread",                "FRED (BofA ML)", "daily"),
    ("BAMLH0A2HYB",     "HY B Spread",                 "FRED (BofA ML)", "daily"),
    ("BAMLH0A3HYC",     "HY CCC-and-Below Spread",     "FRED (BofA ML)", "daily"),
    ("BAMLEMCBPIOAS",   "EM USD Sovereign OAS",        "FRED (BofA ML)", "daily"),
    ("HYG",             "iShares HY Corporate ETF",    "Yahoo",          "daily"),
    ("LQD",             "iShares IG Corporate ETF",    "Yahoo",          "daily"),
    ("TLT",             "iShares 20+Y Treasury ETF",   "Yahoo",          "daily"),

    # ---------- Commodities ----------
    ("CL=F",       "WTI Crude Oil Futures",            "Yahoo",  "daily"),
    ("GC=F",       "Gold Futures",                     "Yahoo",  "daily"),

    # ---------- FX ----------
    ("DX-Y.NYB",   "US Dollar Index",                  "Yahoo",  "daily"),
    ("EURUSD=X",   "EUR/USD Exchange Rate",            "Yahoo",  "daily"),

    # ---------- Macro Indicators ----------
    ("CPIAUCSL",           "CPI All Urban Consumers",          "FRED",  "monthly"),
    ("PCEPILFE",           "Core PCE Price Index",             "FRED",  "monthly"),
    ("UNRATE",             "Unemployment Rate",                "FRED",  "monthly"),
    ("PAYEMS",             "Nonfarm Payroll Employment",       "FRED",  "monthly"),
    ("INDPRO",             "Industrial Production Index",      "FRED",  "monthly"),
    ("ICSA",               "Initial Jobless Claims",           "FRED",  "weekly"),
    ("UMCSENT",            "Consumer Sentiment (Michigan)",    "FRED",  "monthly"),
    ("HOUST",              "Housing Starts",                   "FRED",  "monthly"),
    ("M2SL",               "M2 Money Supply",                  "FRED",  "monthly"),
    ("RSXFS",              "Retail Sales ex Food Services",    "FRED",  "monthly"),
    ("A191RL1Q225SBEA",    "Real GDP Growth Rate",             "FRED",  "quarterly"),
    ("STLFSI4",            "St. Louis Fed Financial Stress",   "FRED",  "weekly"),
    ("DRTSCIS",            "Loan Tightening, Small Firms",     "FRED",  "quarterly"),
    ("DRTSCILM",           "Loan Tightening, Large/Med Firms", "FRED",  "quarterly"),
    ("DRTSSP",             "Loan Standards, Small Firms",      "FRED",  "quarterly"),
    ("DRSDCILM",           "Loan Demand, Large/Med Firms",     "FRED",  "quarterly"),
]


# Category assignment (same as causal_graph_extract.py)
CATEGORIES = {
    "equity_index":  {"label": "Equity Indices",
                      "members": ["^GSPC", "^NDX", "^RUT", "EEM"]},
    "equity_sector": {"label": "Sector ETFs",
                      "members": ["XLF", "XLK", "XLE", "XLV", "XLY", "XLU", "XLRE"]},
    "volatility":    {"label": "Volatility",
                      "members": ["^VIX", "^VVIX", "^MOVE"]},
    "rates":         {"label": "Rates",
                      "members": ["DGS2", "DGS10", "T10Y2Y", "FEDFUNDS",
                                  "SOFR", "SOFR90DAYAVG", "DCPF3M", "DCPN3M",
                                  "TEDRATE"]},
    "credit":        {"label": "Credit Spreads",
                      "members": ["BAMLC0A0CM", "BAMLC0A1CAAA", "BAMLC0A2CAA",
                                  "BAMLC0A3CA", "BAMLC0A4CBBB", "BAMLH0A0HYM2",
                                  "BAMLH0A1HYBB", "BAMLH0A2HYB", "BAMLH0A3HYC",
                                  "BAMLEMCBPIOAS", "HYG", "LQD", "TLT"]},
    "commodity":     {"label": "Commodities",
                      "members": ["CL=F", "GC=F"]},
    "fx":            {"label": "FX",
                      "members": ["DX-Y.NYB", "EURUSD=X"]},
    "macro":         {"label": "Macro Indicators",
                      "members": ["CPIAUCSL", "PCEPILFE", "UNRATE", "PAYEMS",
                                  "INDPRO", "ICSA", "UMCSENT", "HOUST", "M2SL",
                                  "RSXFS", "A191RL1Q225SBEA", "STLFSI4",
                                  "DRTSCIS", "DRTSCILM", "DRTSSP", "DRSDCILM"]},
}

# Category display order (matches Figure 3)
CATEGORY_ORDER = ["equity_index", "equity_sector", "volatility", "rates",
                  "credit", "commodity", "fx", "macro"]


def row_category(code):
    for cat_key, cat_data in CATEGORIES.items():
        if code in cat_data["members"]:
            return cat_key
    return "other"


# ============================================================
# OUTPUT GENERATORS
# ============================================================

def latex_escape(s):
    """Escape LaTeX special characters in strings."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("^", "\\textasciicircum{}"),
        ("~", "\\textasciitilde{}"),
        ("\u2013", "--"),   # en dash
        ("\u2014", "---"),  # em dash
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def build_latex():
    """LaTeX booktabs table with section breaks between categories."""
    lines = []
    lines.append(r"% Table 1: Variable Inventory")
    lines.append(r"% Generated by build_table_1_variable_inventory.py")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Variable inventory: 56 financial variables used in the causal discovery pipeline, grouped by semantic category.}")
    lines.append(r"\label{tab:variable-inventory}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Code} & \textbf{Display Name} & \textbf{Source} & \textbf{Freq.} \\")
    lines.append(r"\midrule")

    total_count = 0
    for cat_key in CATEGORY_ORDER:
        cat_label = CATEGORIES[cat_key]["label"]
        members = CATEGORIES[cat_key]["members"]
        # Category section header
        lines.append(
            r"\multicolumn{4}{l}{\textit{"
            + latex_escape(cat_label)
            + f" (n={len(members)})"
            + r"}} \\"
        )
        # Member rows
        for code in members:
            # Find full row in VARIABLES
            row = next((v for v in VARIABLES if v[0] == code), None)
            if not row:
                continue
            c, display, source, freq = row
            lines.append(
                f"\\quad {latex_escape(c)} & "
                f"{latex_escape(display)} & "
                f"{latex_escape(source)} & "
                f"{latex_escape(freq)} \\\\"
            )
            total_count += 1
        lines.append(r"\addlinespace[2pt]")

    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{Total: {total_count} variables}}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_markdown():
    lines = []
    lines.append("# Table 1: Variable Inventory\n")
    lines.append("56 financial variables used in the causal discovery pipeline, "
                 "grouped by semantic category.\n")

    total = 0
    for cat_key in CATEGORY_ORDER:
        cat_label = CATEGORIES[cat_key]["label"]
        members = CATEGORIES[cat_key]["members"]
        lines.append(f"## {cat_label} (n={len(members)})\n")
        lines.append("| Code | Display Name | Source | Freq. |")
        lines.append("|------|--------------|--------|-------|")
        for code in members:
            row = next((v for v in VARIABLES if v[0] == code), None)
            if not row:
                continue
            c, display, source, freq = row
            lines.append(f"| `{c}` | {display} | {source} | {freq} |")
            total += 1
        lines.append("")

    lines.append(f"**Total: {total} variables**\n")
    return "\n".join(lines)


def build_csv():
    lines = ["code,display_name,category,source,frequency"]
    for cat_key in CATEGORY_ORDER:
        cat_label = CATEGORIES[cat_key]["label"]
        members = CATEGORIES[cat_key]["members"]
        for code in members:
            row = next((v for v in VARIABLES if v[0] == code), None)
            if not row:
                continue
            c, display, source, freq = row
            # CSV-safe quoting
            display_q = '"' + display.replace('"', '""') + '"'
            lines.append(f"{c},{display_q},{cat_label},{source},{freq}")
    return "\n".join(lines) + "\n"


def main():
    print("=" * 70)
    print("  TABLE 1: Variable Inventory")
    print("=" * 70)

    # Sanity check: every variable in VARIABLES must be in exactly one category
    all_categorized = set()
    for cat in CATEGORIES.values():
        all_categorized.update(cat["members"])
    all_variables = {v[0] for v in VARIABLES}

    uncategorized = all_variables - all_categorized
    unlisted = all_categorized - all_variables
    if uncategorized:
        print(f"\n  [WARN] Uncategorized variables: {sorted(uncategorized)}")
    if unlisted:
        print(f"\n  [WARN] Listed in category but no metadata: {sorted(unlisted)}")

    total = len(all_variables)
    print(f"\n  Total variables: {total}")
    for cat_key in CATEGORY_ORDER:
        cat = CATEGORIES[cat_key]
        print(f"    {cat['label']:<20} {len(cat['members']):>3}")

    # Cross-check against canonical
    print("\n  Sanity check vs canonical_paper_numbers.py:")
    if total == 56:
        print(f"    [OK]   Total = 56 (paper: 56)")
    else:
        print(f"    [WARN] Total = {total} (paper: 56) \u2014 drift detected")

    # Generate outputs
    script_dir = Path(__file__).parent
    stem = "table_1_variable_inventory"

    tex_path = script_dir / f"{stem}.tex"
    md_path = script_dir / f"{stem}.md"
    csv_path = script_dir / f"{stem}.csv"

    tex_path.write_text(build_latex())
    md_path.write_text(build_markdown())
    csv_path.write_text(build_csv())

    print(f"\n  Wrote:")
    print(f"    {tex_path}  ({tex_path.stat().st_size} bytes)")
    print(f"    {md_path}   ({md_path.stat().st_size} bytes)")
    print(f"    {csv_path}  ({csv_path.stat().st_size} bytes)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()