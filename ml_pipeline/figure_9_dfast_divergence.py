"""
Figure 9: DFAST 2026 Causal vs Fed Projections
=================================================
Reads dfast_figure_data.json (from dfast_figure_extract.py) and renders a
publication-quality 2-row figure comparing our causal model's projections
against the Fed's DFAST 2026 Severely Adverse scenario.

Layout (2 rows x 2 columns):
  Top row:    Line chart overlay of causal vs Fed projections over 13 quarters
    - Left:   BBB corporate yield
    - Right:  10Y Treasury yield
  Bottom row: Per-quarter percentage divergence (bar chart), same variables

Story:
  - BBB: our model projects 16.8% to 26.3% higher than Fed
  - 10Y Treasury panel: 10.4% to 23.9% higher across 12 above-threshold quarters
  - 34 total divergence points across all 13 variables
  - Real Fed CSV data (not approximated)

Outputs:
  - figure_9_dfast_divergence.pdf
  - figure_9_dfast_divergence.png
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter


# ============================================================
# RC PARAMS (consistent with Figures 5, 6, 8)
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# COLORS
# ============================================================

COLOR_CAUSAL = "#1f3d7a"       # deep blue for our causal model
COLOR_FED = "#b22222"          # red for Fed projections
COLOR_DIVERGENCE_POS = "#8a5aa3"  # purple for positive divergences
COLOR_DIVERGENCE_NEG = "#6b8e6f"  # muted green for negative divergences
COLOR_GRID_ANNOTATION = "#555555"
COLOR_ACCENT = "#1f3d7a"


# Variables to plot (in order). Two headline variables shown in 2x2 layout.
PLOT_VARIABLES = [
    ("BBB_CORPORATE_YIELD",    "BBB Corporate Yield"),
    ("10-YEAR_TREASURY_YIELD", "10Y Treasury Yield"),
]


# ============================================================
# HELPERS
# ============================================================

def format_pct_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.0f}%"))


def format_yield_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))


def plot_projection_overlay(ax, var_data: Dict, display_name: str):
    """Top-row panel: causal vs Fed projections over 13 quarters."""
    causal = var_data.get("causal_path") or []
    fed = var_data.get("fed_path") or []

    n = max(len(causal), len(fed))
    x = np.arange(1, n + 1)

    if fed:
        ax.plot(x[:len(fed)], fed, marker="o", markersize=4,
                color=COLOR_FED, linewidth=1.7, label="Fed projection",
                zorder=4, solid_capstyle="round")
    if causal:
        ax.plot(x[:len(causal)], causal, marker="s", markersize=4,
                color=COLOR_CAUSAL, linewidth=1.8, label="Our causal model",
                zorder=5, solid_capstyle="round")

    # Fill between to visually emphasize the gap
    if causal and fed and len(causal) == len(fed):
        ax.fill_between(
            x[:len(causal)], causal, fed,
            where=np.array(causal) > np.array(fed),
            color=COLOR_CAUSAL, alpha=0.08, zorder=2,
            interpolate=True,
        )

    ax.set_title(display_name, loc="left", fontweight="bold", pad=6)
    ax.set_xlabel("Quarter (2026 Q1 \u2192 2029 Q1)")
    ax.set_ylabel("Yield (%)")
    ax.set_xlim(0.5, n + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    format_yield_axis(ax)

    leg = ax.legend(loc="best", fontsize=8, borderpad=0.4, handlelength=2.0)
    leg.get_frame().set_linewidth(0.5)


def plot_divergence_bars(ax, var_data: Dict, display_name: str):
    """Bottom-row panel: per-quarter % divergence as bar chart.

    None values (missing data, zero-divisor quarters) are rendered as narrow
    grey placeholder bars so quarter spacing is preserved and the viewer can
    see where data is missing.
    """
    div = var_data.get("divergence_pct_per_q") or []
    x = np.arange(1, len(div) + 1)

    if not div:
        ax.text(0.5, 0.5, "(no divergence data)",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes)
        ax.set_title(f"{display_name} — divergence", loc="left",
                     fontweight="bold", pad=6)
        return

    # Separate valid bars from missing-data placeholders
    valid_mask = [v is not None for v in div]
    missing_x = [xi for xi, valid in zip(x, valid_mask) if not valid]
    valid_x = [xi for xi, valid in zip(x, valid_mask) if valid]
    valid_heights = [v for v in div if v is not None]
    valid_colors = [COLOR_DIVERGENCE_POS if v >= 0 else COLOR_DIVERGENCE_NEG
                    for v in valid_heights]

    # Draw valid bars
    if valid_x:
        ax.bar(valid_x, valid_heights, color=valid_colors, edgecolor="white",
               linewidth=0.6, width=0.7, zorder=3)

        # Annotate each valid bar with its value
        for xi, h in zip(valid_x, valid_heights):
            va = "bottom" if h >= 0 else "top"
            y_offset = 0.5 if h >= 0 else -0.5
            ax.text(
                xi, h + y_offset, f"{h:+.1f}",
                ha="center", va=va, fontsize=7, color="#1a1a1a",
                zorder=4,
            )

    # Draw placeholder "below threshold" markers at filtered quarters
    if missing_x:
        # Small grey dot at zero line — shows "quarter exists, divergence below threshold"
        ax.scatter(
            missing_x, [0] * len(missing_x),
            marker="_", color="#999", s=60, linewidths=1.2,
            zorder=3,
        )
        # Add small annotation if only 1-3 missing quarters
        if len(missing_x) <= 3:
            for xi in missing_x:
                ax.annotate(
                    "below\nthreshold",
                    xy=(xi, 0), xytext=(0, -10), textcoords="offset points",
                    ha="center", va="top", fontsize=6.2, color="#999",
                    fontstyle="italic", linespacing=0.9,
                )

    ax.axhline(0, color="#333", linewidth=0.8, zorder=2)

    ax.set_title(f"{display_name} — divergence", loc="left",
                 fontweight="bold", pad=6)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Causal vs Fed (%)")
    ax.set_xlim(0.5, len(div) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    format_pct_axis(ax)

    # Headline annotation: min and max (ignoring filtered quarters)
    vals = [v for v in div if v is not None]
    if vals:
        lo, hi = min(vals), max(vals)
        note = f"Range: {lo:+.1f}% to {hi:+.1f}%"
        if missing_x:
            note += f"  ({len(vals)}/{len(div)} quarters above threshold)"
        ax.text(
            0.98, 0.02, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color=COLOR_ACCENT, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3", linewidth=0.5),
        )


def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    variables_data = data["variables"]

    # Filter to plotting variables that actually have data
    available = [(k, name) for k, name in PLOT_VARIABLES if k in variables_data]
    if len(available) < 1:
        raise SystemExit("ERROR: no headline variables found in data")

    n_vars = len(available)

    fig, axes = plt.subplots(2, n_vars, figsize=(12.5, 8.0))
    # Handle the 1-variable edge case
    if n_vars == 1:
        axes = axes.reshape(2, 1)

    for col, (var_key, display_name) in enumerate(available):
        var_data = variables_data[var_key]
        plot_projection_overlay(axes[0, col], var_data, display_name)
        plot_divergence_bars(axes[1, col], var_data, display_name)

    # Figure-level title and subtitle
    scenario_name = data.get("scenario_name", "DFAST 2026 Severely Adverse")
    source = data.get("scenario_source", "")
    horizon = data.get("horizon_quarters", 13)

    fig.suptitle(
        "DFAST 2026: our causal model vs. Federal Reserve projections",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.975,
    )
    fig.text(
        0.015, 0.945,
        f"{scenario_name}  \u00b7  {source}  \u00b7  "
        f"{horizon} quarterly projections  \u00b7  "
        f"Positive divergence = our model projects more stress than the Fed",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.92], h_pad=2.5, w_pad=2.5)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "dfast_figure_data.json",
        script_dir.parent / "ml_pipeline" / "dfast_figure_data.json",
        Path.cwd() / "dfast_figure_data.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: dfast_figure_data.json not found.")
        print("Expected locations tried:")
        for p in candidates:
            print(f"  {p}")
        print("\nRun dfast_figure_extract.py first to generate the JSON.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_9_dfast_divergence"

    try:
        pdf_path, png_path = render_figure(json_path, output_stem)
    except SystemExit as e:
        print(e)
        return

    print(f"\nFigure 9 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
