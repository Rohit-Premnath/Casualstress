"""
Figure 2: HMM Regime Timeline (2005-2026)
===========================================
Reads regime_timeline.json (from regime_timeline_extract.py) and renders a
publication-quality timeline visualization showing:

  - A wide horizontal band colored by the HMM-classified regime each day
  - The 11 canonical crisis events marked as vertical bands with labels
  - A summary strip showing regime distribution stats

Layout (top to bottom):
  Panel A (tall): the timeline strip
    - X-axis: date (2005 to 2026)
    - Colored bands: each day's HMM regime (green=calm -> red=crisis)
    - Vertical markers: crisis event windows with name labels
  Panel B (short): horizontal bar showing regime day counts + percentages

Outputs:
  - figure_2_regime_timeline.pdf
  - figure_2_regime_timeline.png
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


# ============================================================
# RC PARAMS (consistent with Figures 5, 6, 8, 9)
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Regime display order (calm to crisis)
REGIME_ORDER = ["calm", "normal", "elevated", "stressed", "crisis"]

# Fallback colors if JSON doesn't provide them
DEFAULT_COLORS = {
    "calm":     "#3a7d44",
    "normal":   "#a6d49f",
    "elevated": "#f2d98f",
    "stressed": "#de9b6b",
    "crisis":   "#b22222",
}


def parse_date(s: str) -> datetime:
    """Parse ISO date with a few fallbacks."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt.replace('%', ''))], fmt)
        except ValueError:
            continue
    return datetime.strptime(s[:10], "%Y-%m-%d")


def build_regime_runs(timeline: List[Dict]) -> List[Dict]:
    """
    Compress day-by-day timeline into runs of consecutive same-regime days.
    Returns list of {start_date, end_date, regime_name, n_days}.
    """
    if not timeline:
        return []

    runs = []
    current = {
        "start_date": parse_date(timeline[0]["date"]),
        "end_date": parse_date(timeline[0]["date"]),
        "regime_name": timeline[0]["regime_name"],
        "n_days": 1,
    }

    for entry in timeline[1:]:
        date = parse_date(entry["date"])
        regime = entry["regime_name"]
        if regime == current["regime_name"]:
            current["end_date"] = date
            current["n_days"] += 1
        else:
            runs.append(current)
            current = {
                "start_date": date,
                "end_date": date,
                "regime_name": regime,
                "n_days": 1,
            }
    runs.append(current)
    return runs


def plot_regime_timeline(ax, runs: List[Dict], regime_colors: Dict[str, str],
                         start_date, end_date):
    """Panel A: colored horizontal band per regime day."""
    for run in runs:
        color = regime_colors.get(run["regime_name"],
                                   DEFAULT_COLORS.get(run["regime_name"], "#999"))
        width = (run["end_date"] - run["start_date"]).days + 1
        rect = Rectangle(
            (mdates.date2num(run["start_date"]), 0.0),
            width, 1.0,
            facecolor=color, edgecolor="none", zorder=2,
        )
        ax.add_patch(rect)

    # X-axis formatting
    ax.set_xlim(mdates.date2num(start_date), mdates.date2num(end_date))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="major", length=4, labelsize=8.5)
    ax.tick_params(axis="x", which="minor", length=2)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)


def overlay_events(ax, events: List[Dict], timeline_top_y: float = 1.0):
    """
    Mark each event as a vertical band on the regime timeline.
    Labels placed above, colored by detected/missed status.
    """
    for i, event in enumerate(events):
        start = parse_date(event["start"])
        end = parse_date(event["end"])
        width = (end - start).days + 1

        # Color border by detection success
        border_color = "#1a1a1a" if event["detected"] else "#b22222"
        border_width = 1.2 if event["detected"] else 1.8

        # Vertical event band (semi-transparent)
        rect = Rectangle(
            (mdates.date2num(start), 0.0),
            width, timeline_top_y,
            facecolor="none", edgecolor=border_color,
            linewidth=border_width, zorder=5, linestyle="-",
        )
        ax.add_patch(rect)

        # Label above the timeline
        mid = start + (end - start) / 2
        # Shorten name for display
        short_name = event["name"].replace("US Debt Downgrade", "US Debt")
        short_name = short_name.replace("China/Oil Crash", "China/Oil")
        short_name = short_name.replace("Volmageddon", "Volmag.")
        short_name = short_name.replace("Q4 Selloff", "Q4 Sell")
        short_name = short_name.replace("Tech Selloff", "Tech Sell")
        short_name = short_name.replace("Rate Hike", "Rate Hike")
        short_name = short_name.replace("SVB Crisis", "SVB")
        short_name = short_name.replace("Flash Crash", "Flash")

        # Stagger labels vertically to avoid overlap
        y_positions = [1.18, 1.33, 1.48]
        y_label = y_positions[i % 3]
        label_color = "#1a1a1a" if event["detected"] else "#b22222"
        fontweight = "normal" if event["detected"] else "bold"

        ax.annotate(
            short_name,
            xy=(mdates.date2num(mid), timeline_top_y),
            xytext=(mdates.date2num(mid), y_label),
            textcoords="data",
            ha="center", va="bottom", fontsize=7.3,
            color=label_color, fontweight=fontweight,
            arrowprops=dict(arrowstyle="-", color=border_color,
                            linewidth=0.5, alpha=0.6),
        )


def plot_regime_distribution(ax, regime_stats: Dict, total_days: int):
    """Panel B: horizontal bar showing regime day counts."""
    # Sort by regime order (calm -> crisis), not by count
    ordered = [(r, regime_stats[r]) for r in REGIME_ORDER if r in regime_stats]

    x_cursor = 0.0
    for regime_name, stats in ordered:
        width = stats["days"]
        color = stats.get("color", DEFAULT_COLORS.get(regime_name, "#999"))
        rect = Rectangle(
            (x_cursor, 0), width, 1.0,
            facecolor=color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.add_patch(rect)

        # Label inside bar (if wide enough) or above
        mid = x_cursor + width / 2
        if width > total_days * 0.06:  # bar wide enough for label
            ax.text(
                mid, 0.5, f"{regime_name}\n{stats['days']:,} ({stats['pct']:.1f}%)",
                ha="center", va="center", fontsize=8.5,
                color="white" if regime_name in ("crisis", "calm") else "#1a1a1a",
                fontweight="bold",
            )
        else:
            # Narrow bars get a label above the bar
            ax.text(
                mid, 1.12, f"{regime_name}",
                ha="center", va="bottom", fontsize=7.5, color="#1a1a1a",
                fontweight="bold",
            )
            ax.text(
                mid, 1.04, f"{stats['days']:,} ({stats['pct']:.1f}%)",
                ha="center", va="bottom", fontsize=7,
                color="#555555",
            )

        x_cursor += width

    ax.set_xlim(0, total_days)
    ax.set_ylim(0, 1.5)   # extra room for labels above narrow bars
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(
        f"Regime distribution (total: {total_days:,} days)",
        loc="left", fontweight="bold", pad=6,
    )


def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    timeline = data["timeline"]
    events = data["events"]
    regime_stats = data["regime_stats"]
    metadata = data["metadata"]

    # Build runs for efficient rendering
    runs = build_regime_runs(timeline)
    start_date = parse_date(metadata["start_date"])
    end_date = parse_date(metadata["end_date"])

    regime_colors = {
        name: stats.get("color", DEFAULT_COLORS.get(name, "#999"))
        for name, stats in regime_stats.items()
    }

    n_detected = sum(1 for e in events if e["detected"])

    # --- Layout: 3 rows
    # Row 0: timeline (with event labels above)
    # Row 1: legend strip
    # Row 2: regime distribution bar
    fig = plt.figure(figsize=(14.0, 6.4))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[3.0, 0.35, 1.2],
        hspace=0.9,
    )
    ax_timeline = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])

    # --- Panel A: regime timeline ---
    plot_regime_timeline(ax_timeline, runs, regime_colors, start_date, end_date)
    overlay_events(ax_timeline, events, timeline_top_y=1.0)
    ax_timeline.set_ylim(0, 1.70)

    ax_timeline.set_title(
        "(a) HMM regime classification: daily assignments, 2005\u20132026",
        loc="left", fontweight="bold", pad=8,
    )

    # --- Legend strip: regime colors + event box key ---
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    # Regime color swatches (left side)
    ax_legend.text(0.0, 0.5, "Regime:", fontsize=9, fontweight="bold",
                   color="#555555", ha="left", va="center",
                   transform=ax_legend.transAxes)

    swatch_x = 0.075
    for regime_name in REGIME_ORDER:
        if regime_name not in regime_stats:
            continue
        color = regime_colors[regime_name]
        ax_legend.add_patch(Rectangle(
            (swatch_x, 0.35), 0.018, 0.30,
            facecolor=color, edgecolor="#555555", linewidth=0.5,
            transform=ax_legend.transAxes,
        ))
        ax_legend.text(swatch_x + 0.022, 0.5, regime_name,
                       fontsize=8.5, color="#1a1a1a", ha="left", va="center",
                       transform=ax_legend.transAxes)
        swatch_x += 0.11

    # Event box key (right side)
    key_x = 0.68
    ax_legend.text(key_x - 0.085, 0.5, "Event box:",
                   fontsize=9, fontweight="bold", color="#555555",
                   ha="left", va="center", transform=ax_legend.transAxes)
    ax_legend.add_patch(Rectangle(
        (key_x, 0.35), 0.018, 0.30,
        facecolor="none", edgecolor="#1a1a1a", linewidth=1.2,
        transform=ax_legend.transAxes,
    ))
    ax_legend.text(key_x + 0.022, 0.5, "detected (dominant = stressed/crisis)",
                   fontsize=8.5, color="#1a1a1a",
                   ha="left", va="center", transform=ax_legend.transAxes)

    ax_legend.add_patch(Rectangle(
        (key_x + 0.26, 0.35), 0.018, 0.30,
        facecolor="none", edgecolor="#b22222", linewidth=1.8,
        transform=ax_legend.transAxes,
    ))
    ax_legend.text(key_x + 0.282, 0.5, "missed",
                   fontsize=8.5, color="#b22222", fontweight="bold",
                   ha="left", va="center", transform=ax_legend.transAxes)

    # --- Panel B: regime distribution bar ---
    plot_regime_distribution(ax_dist, regime_stats, metadata["n_days"])

    # --- Figure title ---
    fig.suptitle(
        "HMM regime classification and crisis-event detection",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.985,
    )
    fig.text(
        0.015, 0.948,
        f"5-state Hidden Markov Model over {metadata['n_days']:,} trading days  \u00b7  "
        f"{n_detected}/{len(events)} crisis events dominantly classified as "
        f"stressed or crisis regime",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    # Note: tight_layout with suptitle — use rect to reserve top space for title
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path, n_detected, len(events)


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "regime_timeline.json",
        script_dir.parent / "ml_pipeline" / "regime_timeline.json",
        Path.cwd() / "regime_timeline.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: regime_timeline.json not found.")
        print("Run regime_timeline_extract.py first.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_2_regime_timeline"

    pdf_path, png_path, n_detected, n_total = render_figure(json_path, output_stem)

    print(f"\nFigure 2 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  Detection rate: {n_detected}/{n_total} "
          f"({n_detected / n_total * 100:.1f}%)")


if __name__ == "__main__":
    main()