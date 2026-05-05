"""
Figure 8: Per-Event x Per-Variable Coverage Heatmap
=====================================================
Reads per_variable_coverage_matrix.json (from per_variable_coverage_matrix.py)
and renders an 11-events x 6-variables heatmap of in-range coverage rates.

Layout:
  - Rows: 11 canonical events (chronological; VAL in dark, TEST in normal)
  - Columns: 6 key variables (S&P, VIX, XLF, DGS10, Oil, HY Spread)
  - Cell color: fraction of 5 seeds where actual fell in 5-95% band
    - Green: high coverage (>= 0.8)
    - Yellow: moderate (0.5-0.8)
    - Red:   low (< 0.5)
  - Right margin: row % (event total coverage)
  - Bottom margin: column averages (variable total coverage)
  - Cell annotation: numeric value (0.00 - 1.00)

Outputs:
  - figure_8_per_event_heatmap.pdf  (vector, for paper)
  - figure_8_per_event_heatmap.png  (raster, 300 DPI)

Design:
  - Diverging color scale (RdYlGn) is intuitive: red = bad, green = good
  - Cells annotated with numeric values for precision
  - VAL/TEST split labeled on the left to show held-out events
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


# ============================================================
# RC PARAMS (matches Figure 6 for consistency across paper)
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# DISPLAY CONFIG
# ============================================================

VARIABLE_DISPLAY = {
    "^GSPC":        "S&P 500",
    "^VIX":         "VIX",
    "DGS10":        "10Y Treas.",
    "CL=F":         "Oil",
    "XLF":          "Financials",
    "BAMLH0A0HYM2": "HY OAS",
}

# Preferred column order (matches Figure 6)
COLUMN_ORDER = ["^GSPC", "^VIX", "XLF", "DGS10", "CL=F", "BAMLH0A0HYM2"]

# Event order: chronological (matches canonical EVENTS list)
EVENT_ORDER = [
    "2008 GFC",
    "2010 Flash Crash",
    "2011 US Debt Downgrade",
    "2015 China/Oil Crash",
    "2016 Brexit",
    "2018 Volmageddon",
    "2018 Q4 Selloff",
    "2020 COVID",
    "2020 Tech Selloff",
    "2022 Rate Hike",
    "2023 SVB Crisis",
]


# ============================================================
# COLOR SCALE
# ============================================================

def make_coverage_cmap():
    """
    Diverging colormap: red (low) -> yellow (mid) -> green (high).
    Paper-friendly muted shades, colorblind-safe enough.
    """
    colors = [
        (0.0,  "#b22222"),  # deep red
        (0.35, "#de9b6b"),  # light red-orange
        (0.55, "#f2d98f"),  # pale yellow
        (0.75, "#a6d49f"),  # pale green
        (1.0,  "#3a7d44"),  # deep green
    ]
    return LinearSegmentedColormap.from_list(
        "coverage", colors, N=256
    )


COVERAGE_CMAP = make_coverage_cmap()


# ============================================================
# MAIN
# ============================================================

def build_matrix(data: Dict, metric: str = "in_range_rate"):
    """
    Construct a (n_events x n_variables) numpy array of coverage values.
    Returns (matrix, row_events_in_order, col_vars_in_order, event_splits).
    metric: "in_range_rate" or "direction_rate"
    """
    matrix_data = data["matrix"]

    # Filter to events that actually appear in the data, in EVENT_ORDER
    events_available = [e for e in EVENT_ORDER if e in matrix_data]
    variables_available = [v for v in COLUMN_ORDER if v in data.get("key_variables", COLUMN_ORDER)]

    # Defensive: if key_variables wasn't populated, grab from first event
    if not variables_available and events_available:
        first_vars = list(matrix_data[events_available[0]]["variables"].keys())
        variables_available = [v for v in COLUMN_ORDER if v in first_vars]

    n_events = len(events_available)
    n_vars = len(variables_available)
    matrix = np.full((n_events, n_vars), np.nan)
    splits = []

    for i, event in enumerate(events_available):
        event_row = matrix_data[event]
        splits.append(event_row.get("split", ""))
        for j, var in enumerate(variables_available):
            if var in event_row["variables"]:
                matrix[i, j] = event_row["variables"][var].get(metric, np.nan)

    return matrix, events_available, variables_available, splits


def plot_heatmap(ax, matrix, events, variables, splits, title: str):
    """
    Render an extended heatmap that includes an extra column (row averages)
    on the right and an extra row (column averages) at the bottom. This is
    cleaner than overlaying text on an axis because everything lives in the
    same coordinate system.
    """
    n_events, n_vars = matrix.shape

    # Row averages and column averages (ignoring NaN)
    row_avg = np.nanmean(matrix, axis=1)   # shape: (n_events,)
    col_avg = np.nanmean(matrix, axis=0)   # shape: (n_vars,)
    overall_avg = np.nanmean(matrix)

    # Build extended matrix: main data + right col (row avgs) + bottom row (col avgs)
    ext = np.full((n_events + 1, n_vars + 1), np.nan)
    ext[:n_events, :n_vars] = matrix
    ext[:n_events, n_vars] = row_avg        # rightmost column
    ext[n_events, :n_vars] = col_avg        # bottom row
    ext[n_events, n_vars] = overall_avg

    # Plot
    im = ax.imshow(
        ext, cmap=COVERAGE_CMAP, vmin=0.0, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )

    # Column labels (top): variable names + "Avg"
    col_labels = [VARIABLE_DISPLAY.get(v, v) for v in variables] + ["Avg"]
    ax.set_xticks(np.arange(n_vars + 1))
    ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Emphasize the "Avg" column header
    for tick in ax.get_xticklabels():
        if tick.get_text() == "Avg":
            tick.set_fontweight("bold")
            tick.set_fontstyle("italic")
            tick.set_color("#555555")

    # Row labels (left): event names + "Avg"
    row_labels = []
    for name, split in zip(events, splits):
        split_marker = "  (V)" if split == "VAL" else "  (T)" if split == "TEST" else ""
        row_labels.append(f"{name}{split_marker}")
    row_labels.append("Avg")

    ax.set_yticks(np.arange(n_events + 1))
    ax.set_yticklabels(row_labels, fontsize=8.5)

    # Emphasize the "Avg" row label
    ytick_labels = ax.get_yticklabels()
    if ytick_labels:
        ytick_labels[-1].set_fontweight("bold")
        ytick_labels[-1].set_fontstyle("italic")
        ytick_labels[-1].set_color("#555555")

    # Cell annotations
    for i in range(n_events + 1):
        for j in range(n_vars + 1):
            val = ext[i, j]
            if np.isnan(val):
                continue
            # Text color: white on dark cells, dark on light cells
            text_color = "white" if (val < 0.3 or val > 0.92) else "#1a1a1a"
            # Avg cells get italic styling
            is_avg = (i == n_events) or (j == n_vars)
            fontstyle = "italic" if is_avg else "normal"
            fontweight = "bold" if (val == 1.0 or val == 0.0 or is_avg) else "normal"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, color=text_color,
                fontweight=fontweight, fontstyle=fontstyle,
            )

    # Gridlines between cells (white separators)
    ax.set_xticks(np.arange(n_vars + 2) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_events + 2) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    # Thicker separator between data and "Avg" column/row
    ax.axvline(n_vars - 0.5, color="#333333", linewidth=1.2, zorder=5)
    ax.axhline(n_events - 0.5, color="#333333", linewidth=1.2, zorder=5)

    # VAL/TEST divider (in the main data area only)
    try:
        first_test_idx = splits.index("TEST")
        ax.plot(
            [-0.5, n_vars - 0.5], [first_test_idx - 0.5, first_test_idx - 0.5],
            color="#b22222", linewidth=1.5, linestyle="-", zorder=6,
        )
    except ValueError:
        pass

    ax.set_title(title, loc="left", fontweight="bold", pad=10)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    return im, row_avg, col_avg


def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    cov_matrix, events, variables, splits = build_matrix(data, "in_range_rate")
    dir_matrix, _, _, _ = build_matrix(data, "direction_rate")

    n_events = len(events)
    n_vars = len(variables)

    fig = plt.figure(figsize=(14.0, 7.8))

    # GridSpec: two heatmaps on top, colorbar on bottom
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1, 1], height_ratios=[40, 1],
        hspace=0.12, wspace=0.45,
    )

    ax_cov = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[1, :])

    im1, row_avg_cov, col_avg_cov = plot_heatmap(
        ax_cov, cov_matrix, events, variables, splits,
        "(a) Coverage: actual value inside 5\u201395% band",
    )

    im2, row_avg_dir, col_avg_dir = plot_heatmap(
        ax_dir, dir_matrix, events, variables, splits,
        "(b) Direction: median scenario has correct sign",
    )

    # Shared colorbar at bottom
    cbar = fig.colorbar(
        im1, cax=ax_cbar, orientation="horizontal",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    cbar.ax.set_xticklabels(
        ["0.00\n(never)", "0.25", "0.50", "0.75", "1.00\n(always)"],
        fontsize=8,
    )
    cbar.ax.set_xlabel(
        "Rate across 5 random seeds  \u00b7  "
        "Red VAL/TEST line separates in-sample (top 4 events) from held-out (bottom 7 events)  \u00b7  "
        "(V) = validation, (T) = test",
        fontsize=8.5, labelpad=8,
    )
    cbar.outline.set_linewidth(0.5)

    # Figure title
    fig.suptitle(
        "Per-event per-variable coverage and direction accuracy",
        fontsize=12, fontweight="bold", x=0.015, ha="left", y=0.975,
    )
    fig.text(
        0.015, 0.945,
        f"Canonical model (Student-t, data-fit df)  \u00b7  "
        f"{data.get('n_seeds', 5)} seeds \u00d7 {n_events} events \u00d7 {n_vars} variables = "
        f"{data.get('n_seeds', 5) * n_events * n_vars} evaluations",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.99)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path, row_avg_cov, col_avg_cov


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "per_variable_coverage_matrix.json",
        script_dir.parent / "ml_pipeline" / "per_variable_coverage_matrix.json",
        Path.cwd() / "per_variable_coverage_matrix.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: per_variable_coverage_matrix.json not found.")
        print("Expected locations tried:")
        for p in candidates:
            print(f"  {p}")
        print("\nRun per_variable_coverage_matrix.py first.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_8_per_event_heatmap"

    pdf_path, png_path, row_avgs, col_avgs = render_figure(json_path, output_stem)

    print(f"\nFigure 8 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"\n  Row averages (event-level coverage %):")
    events_for_display = [e for e in EVENT_ORDER]
    for event, avg in zip(events_for_display[:len(row_avgs)], row_avgs):
        print(f"    {event:<30} {avg * 100:>5.1f}%")
    print(f"\n  Column averages (variable-level coverage):")
    vars_for_display = COLUMN_ORDER
    for var, avg in zip(vars_for_display[:len(col_avgs)], col_avgs):
        display = VARIABLE_DISPLAY.get(var, var)
        print(f"    {display:<14} {avg:.2f}")


if __name__ == "__main__":
    main()
