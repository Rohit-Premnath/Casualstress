"""
Figure 6: COVID Fan Chart
==========================
Reads covid_fan_chart.json (produced by covid_fan_chart_extract.py) and
renders a publication-quality 2x3 grid of fan charts — one per key variable.

For each variable:
  - 100 faint scenario path lines (spaghetti background)
  - 5-95% shaded band (outer envelope)
  - 25-75% shaded band (inner envelope)
  - Scenario median as dashed line
  - Actual market trajectory as solid line (red if outside 5-95%, black if inside)

Outputs:
  - figure_6_covid_fan_chart.pdf  (vector, for paper)
  - figure_6_covid_fan_chart.png  (raster, 300 DPI, for slides)

Design choices for paper readability:
  - Serif-free clean typography, ~8-9pt body text
  - Semantic colors: blue for scenarios, black for actual, red when missed
  - Shared-style axes, consistent tick spacing
  - Editable text in PDF (fonttype 42)
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter


# ============================================================
# PAPER-QUALITY RC PARAMS
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.titlecolor": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.linestyle": (0, (1, 3)),
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "grid.color": "#d0d0d0",
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#bbbbbb",
    "legend.fancybox": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,   # TrueType, editable in PDF
    "ps.fonttype": 42,
})


# ============================================================
# COLORS
# ============================================================

COLOR_SCENARIO_LINE = "#7a9dd4"       # individual scenario paths
COLOR_BAND_OUTER = "#b8c9e3"          # 5-95% band
COLOR_BAND_INNER = "#7a9dd4"          # 25-75% band
COLOR_MEDIAN = "#1f3d7a"              # scenario median
COLOR_ACTUAL_COVERED = "#111111"      # actual path when within 5-95%
COLOR_ACTUAL_MISSED = "#b22222"       # actual path when outside 5-95%

DISPLAY_NAMES = {
    "^GSPC":        "S&P 500",
    "^VIX":         "VIX",
    "DGS10":        "10Y Treasury Yield",
    "CL=F":         "WTI Crude Oil",
    "XLF":          "Financials (XLF)",
    "BAMLH0A0HYM2": "HY OAS Spread",
}


# ============================================================
# HELPERS
# ============================================================

def format_y_axis(ax, unit: str):
    if unit == "%":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    elif unit == "bps":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.0f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.1f}"))


# Per-variable y-axis limits. These clip scenario-path outliers so the
# display is dominated by the 5-95% band + actual trajectory, not by
# heavy-tailed Student-t draws that occasionally run to extreme values.
# Chosen to comfortably enclose (p5, p95) + actual trajectory for COVID event.
Y_LIMITS = {
    "^GSPC":        (-70,   20),    # %
    "^VIX":         (-100, 800),    # %; actual +315%, p95 ~+460%
    "XLF":          (-90,   30),    # %
    "DGS10":        (-300, 150),    # bps
    "CL=F":         (-90,   30),    # %
    "BAMLH0A0HYM2": (-300, 1400),   # bps; actual +726, p95 ~+607
}


def compute_ylim_dynamic(var_data, horizon, margin_pct=10):
    """
    Fallback: compute y-limits from 1st/99th percentile of ALL plotted values
    (scenarios + actual + bands), with some margin. Used when variable isn't
    in the Y_LIMITS dict.
    """
    scenarios = np.array(var_data["scenario_paths"])
    actual = np.array(var_data["actual_path"])
    p5 = np.array(var_data["percentiles"]["p5"])
    p95 = np.array(var_data["percentiles"]["p95"])

    lo = min(np.percentile(scenarios, 1), actual.min(), p5.min())
    hi = max(np.percentile(scenarios, 99), actual.max(), p95.max())
    span = hi - lo
    return (lo - span * margin_pct / 100, hi + span * margin_pct / 100)


def plot_one_panel(ax, var: str, var_data: Dict, horizon: int, window: int,
                   show_legend: bool = False):
    horizon_x = np.arange(horizon)
    actual_x = np.arange(min(window, len(var_data["actual_path"])))

    percentiles = var_data["percentiles"]
    p5 = np.array(percentiles["p5"])
    p25 = np.array(percentiles["p25"])
    p50 = np.array(percentiles["p50"])
    p75 = np.array(percentiles["p75"])
    p95 = np.array(percentiles["p95"])

    actual = np.array(var_data["actual_path"])
    unit = var_data["display_unit"]
    in_band = var_data["sanity_check"]["in_band_at_end"]

    # Layer 1: faint scenario paths (100 of 200 to reduce overplotting)
    scenarios = var_data["scenario_paths"]
    n_shown = min(100, len(scenarios))
    for path in scenarios[:n_shown]:
        ax.plot(horizon_x, path, color=COLOR_SCENARIO_LINE,
                alpha=0.06, linewidth=0.35, zorder=1)

    # Layer 2: percentile bands
    ax.fill_between(horizon_x, p5, p95,
                    color=COLOR_BAND_OUTER, alpha=0.35,
                    label="5\u201395% range", zorder=2, linewidth=0)
    ax.fill_between(horizon_x, p25, p75,
                    color=COLOR_BAND_INNER, alpha=0.45,
                    label="25\u201375% range", zorder=3, linewidth=0)

    # Layer 3: scenario median
    ax.plot(horizon_x, p50, color=COLOR_MEDIAN, linewidth=1.3,
            linestyle=(0, (4, 2)), label="Scenario median", zorder=4)

    # Layer 4: actual trajectory
    actual_color = COLOR_ACTUAL_COVERED if in_band else COLOR_ACTUAL_MISSED
    ax.plot(actual_x, actual, color=actual_color, linewidth=1.9,
            label="Actual", zorder=5, solid_capstyle="round")

    # Zero baseline
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="-", alpha=0.6, zorder=0)

    # Title with status
    display_name = DISPLAY_NAMES.get(var, var)
    status_marker = "" if in_band else "  [outside 5\u201395%]"
    ax.set_title(f"{display_name}{status_marker}",
                 loc="left", fontweight="bold", pad=6)

    ax.set_xlabel("Trading days from event start")
    ax.set_ylabel(f"Cumulative change ({unit})")
    ax.set_xlim(0, horizon - 1)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Apply per-variable y-axis clip (fixed if specified, else dynamic)
    if var in Y_LIMITS:
        ax.set_ylim(*Y_LIMITS[var])
    else:
        ax.set_ylim(*compute_ylim_dynamic(var_data, horizon))

    format_y_axis(ax, unit)

    # Event-end marker
    ax.axvline(window - 1, color="#666666", linewidth=0.5,
               linestyle=(0, (2, 2)), alpha=0.7, zorder=1)

    if show_legend:
        leg = ax.legend(loc="upper left", ncol=2, fontsize=7.2,
                        columnspacing=1.0, handlelength=1.8,
                        borderpad=0.4, handletextpad=0.5)
        leg.get_frame().set_linewidth(0.5)


def make_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    horizon = data["horizon_days"]
    window = data["window_days"]
    n_scenarios = data["n_scenarios"]

    preferred_order = ["^GSPC", "^VIX", "XLF", "DGS10", "CL=F", "BAMLH0A0HYM2"]
    variables_in_order = [v for v in preferred_order if v in data["variables"]]
    if len(variables_in_order) != 6:
        variables_in_order = list(data["variables"].keys())[:6]

    n_covered = sum(
        1 for v in variables_in_order
        if data["variables"][v]["sanity_check"]["in_band_at_end"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
    axes_flat = axes.flatten()

    for i, var in enumerate(variables_in_order):
        var_data = data["variables"][var]
        plot_one_panel(axes_flat[i], var, var_data, horizon, window,
                       show_legend=(i == 0))

    # Title + subtitle using figure-relative coordinates, with tight_layout
    # reserving 12% of the figure for the header
    fig.suptitle(
        f"COVID-19 scenarios: {n_scenarios} paths vs. actual market trajectory",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.975,
    )
    fig.text(
        0.015, 0.943,
        f"Event: {data['event_start']} \u2192 {data['event_end']}  \u00b7  "
        f"Horizon: {horizon} days  \u00b7  "
        f"Covered at event-end: {n_covered} / {len(variables_in_order)} variables  \u00b7  "
        f"Seed: {data['seed']}",
        fontsize=8.8, color="#555555", ha="left", style="italic",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2.5, w_pad=2.0)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path, n_covered, len(variables_in_order)


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "covid_fan_chart.json",
        script_dir.parent / "ml_pipeline" / "covid_fan_chart.json",
        Path.cwd() / "covid_fan_chart.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: covid_fan_chart.json not found.")
        print("Expected locations tried:")
        for p in candidates:
            print(f"  {p}")
        print("\nRun covid_fan_chart_extract.py first.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_6_covid_fan_chart"

    pdf_path, png_path, n_covered, n_total = make_figure(json_path, output_stem)

    print(f"\nFigure 6 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  Coverage at event end: {n_covered}/{n_total} variables")


if __name__ == "__main__":
    main()