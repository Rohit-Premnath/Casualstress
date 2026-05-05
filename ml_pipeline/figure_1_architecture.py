"""
Figure 1: CausalStress System Architecture
============================================
Horizontal pipeline diagram showing data flow through four processing layers:
  1. DATA LAYER      — ingestion, preprocessing, regime labels
  2. DISCOVERY LAYER — parallel DYNOTEARS + PCMCI, ensemble merge, regime split
  3. INFERENCE LAYER — regime-VAR, Student-t Monte Carlo, scenario ensemble
  4. VALIDATION LAYER— event backtesting, DFAST comparison, paper outputs

No DB queries. No JSON. Pure schematic — all numbers are locked constants from
the paper (canonical_paper_numbers.py).

Outputs:
  - figure_1_architecture.pdf
  - figure_1_architecture.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# COLORS — one accent per swim lane
# ============================================================

LANE_COLORS = {
    "data":       {"bg": "#eaf3fc", "border": "#4a7fb5", "accent": "#1f3d7a"},
    "discovery":  {"bg": "#f2e8f5", "border": "#8a5aa3", "accent": "#5a3278"},
    "inference":  {"bg": "#fef0e6", "border": "#d97706", "accent": "#a4510b"},
    "validation": {"bg": "#e8f3ed", "border": "#3a7d44", "accent": "#1f5229"},
}

COLOR_BOX_BG = "white"
COLOR_BOX_TEXT = "#1a1a1a"
COLOR_BOX_SUBTEXT = "#555555"
COLOR_ARROW = "#666666"


# ============================================================
# LAYOUT CONSTANTS
# ============================================================

FIG_W, FIG_H = 15.5, 9.0

# Coordinate space: 0-100 both axes
# Swim lanes are horizontal strips
LANE_LAYOUT = [
    # (key, label, y_bottom, y_top)
    ("data",       "DATA LAYER",       76, 96),
    ("discovery",  "DISCOVERY LAYER",  53, 74),
    ("inference",  "INFERENCE LAYER",  30, 51),
    ("validation", "VALIDATION LAYER",  7, 28),
]

# Lane label column (left side)
LABEL_COL_X = 3
LANE_START_X = 10     # where box columns begin

# ============================================================
# BOX DEFINITIONS — the content of each pipeline box
# ============================================================
# Each box: (lane, x_left, x_right, y_bottom, y_top, title, subtitle_lines)

BOXES = [
    # ---------- DATA LAYER (top row) ----------
    ("data", 10, 26, 81, 91,
     "Raw Data Sources",
     ["FRED macro/rates/credit (35)",
      "Yahoo Finance market series (21)",
      "2005 \u2013 2026"]),

    ("data", 30, 46, 81, 91,
     "Data Processing",
     ["Stationarity transforms",
      "Missing-data handling",
      "56 variables \u00d7 5,528 days"]),

    ("data", 50, 66, 81, 91,
     "HMM Regime Detection",
     ["5-state Gaussian HMM",
      "calm / normal / elevated /",
      "stressed / crisis"]),

    ("data", 70, 95, 81, 91,
     "Regime-Labeled Panel",
     ["Daily regime assignment",
      "72.7% event detection",
      "Feeds downstream discovery"]),

    # ---------- DISCOVERY LAYER ----------
    # Two parallel boxes (DYNOTEARS + PCMCI) that merge into Ensemble
    ("discovery", 10, 24, 63, 71,
     "DYNOTEARS",
     ["LASSO VAR(1)",
      "weight thresholding"]),

    ("discovery", 10, 24, 55, 62,
     "PCMCI",
     ["conditional independence",
      "bootstrap p-values"]),

    ("discovery", 32, 50, 58, 69,
     "Ensemble Merge",
     ["255 consensus edges",
      "(both methods agree)",
      "994 PCMCI-only retained",
      "PR-AUC = 0.168"]),

    ("discovery", 56, 78, 58, 69,
     "Regime-Conditional Graphs",
     ["Split data by regime",
      "Re-fit per regime",
      "+211 stress-only edges",
      "\u221297 calm-only edges"]),

    ("discovery", 83, 95, 58, 69,
     "Amplification",
     ["9.2\u00d7 inflation spiral",
      "6.9\u00d7 lending standards contagion",
      "4.2\u00d7 credit spread cascade"]),

    # ---------- INFERENCE LAYER ----------
    ("inference", 10, 30, 35, 46,
     "Regime-Conditional VAR",
     ["Per-regime coefficients",
      "Student-t innovations:",
      "normal df=5.97, mid=4.79,",
      "crisis df=3.84"]),

    ("inference", 34, 54, 35, 46,
     "Monte Carlo Engine",
     ["400 candidate paths",
      "\u2192 200 weighted/displayed scenarios",
      "(2\u00d7 oversample + weighting)"]),

    ("inference", 58, 78, 35, 46,
     "Scenario Ensemble",
     ["Fan-chart projections",
      "Soft causal filtering",
      "Plausibility scoring"]),

    ("inference", 82, 95, 35, 46,
     "AI Risk Narratives",
     ["LLM-assisted explanations",
      "Regulatory-style narratives",
      "DFAST/EBA alignment"]),

    # ---------- VALIDATION LAYER ----------
    ("validation", 10, 28, 12, 23,
     "Event Backtesting",
     ["7 held-out test events",
      "90.0% test coverage",
      "100% pairwise",
      "77.6% direction"]),

    ("validation", 32, 50, 12, 23,
     "Baseline Comparison",
     ["Historical replay (+29 pts, p=0.008)",
      "Uncond. VAR  (+28 pts, p=0.031)",
      "Gaussian MC / Regime VAR"]),

    ("validation", 54, 72, 12, 23,
     "DFAST Validation",
     ["Fed 2026 Severely Adverse",
      "34 divergence cells",
      "BBB +16.8\u201326.3%",
      "10Y +10.4\u201323.9%"]),

    ("validation", 76, 95, 12, 23,
     "Paper Outputs",
     ["Figures 2\u20139 + tables",
      "Canonical model + metrics locked",
      "Reproducibility verified"]),
]


# ============================================================
# ARROW DEFINITIONS — (from_box_idx, to_box_idx, style)
# ============================================================
# Box indices in BOXES list (0-indexed). style: "straight" or "curved"

ARROWS = [
    # DATA LAYER horizontal flow
    (0, 1, "h"), (1, 2, "h"), (2, 3, "h"),

    # DATA -> DISCOVERY (downward)
    (1, 4, "v"),  # Processed data -> DYNOTEARS
    (1, 5, "v"),  # Processed data -> PCMCI
    (3, 7, "v"),  # Regime-Labeled Panel -> Regime-Conditional Graphs

    # DISCOVERY internal: DYNOTEARS + PCMCI -> Ensemble
    (4, 6, "h"),  # DYNOTEARS -> Ensemble
    (5, 6, "h"),  # PCMCI -> Ensemble (curves up)

    # DISCOVERY horizontal
    (6, 7, "h"),  # Ensemble -> Regime-Conditional Graphs
    (7, 8, "h"),  # Regime-Conditional -> Amplification

    # DISCOVERY -> INFERENCE (downward)
    (7, 9,  "v"),  # Regime graphs -> Regime-VAR

    # INFERENCE horizontal flow
    (9, 10, "h"), (10, 11, "h"), (11, 12, "h"),

    # INFERENCE -> VALIDATION (downward)
    (11, 13, "v"), (11, 15, "v"),

    # VALIDATION horizontal
    (13, 14, "h"), (14, 15, "h"), (15, 16, "h"),
]


# ============================================================
# DRAWING
# ============================================================

def draw_swim_lanes(ax):
    """Draw the four colored swim-lane backgrounds with left-side labels."""
    for key, label, y_bot, y_top in LANE_LAYOUT:
        colors = LANE_COLORS[key]
        # Lane background
        lane_bg = Rectangle(
            (LANE_START_X - 2, y_bot - 1),
            100 - LANE_START_X + 2, (y_top - y_bot) + 2,
            facecolor=colors["bg"], edgecolor=colors["border"],
            linewidth=0.9, alpha=0.45, zorder=1,
        )
        ax.add_patch(lane_bg)
        # Left-side label (vertical text)
        y_mid = (y_bot + y_top) / 2
        ax.text(
            LABEL_COL_X, y_mid, label,
            ha="center", va="center", rotation=90,
            fontsize=10, fontweight="bold",
            color=colors["accent"], zorder=3,
        )


def draw_box(ax, lane, x_left, x_right, y_bot, y_top, title, subtitle_lines):
    """Draw one rounded pipeline box with title and subtitle."""
    colors = LANE_COLORS[lane]
    width = x_right - x_left
    height = y_top - y_bot

    box = FancyBboxPatch(
        (x_left, y_bot), width, height,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        facecolor=COLOR_BOX_BG,
        edgecolor=colors["border"],
        linewidth=1.3,
        zorder=5,
    )
    ax.add_patch(box)

    # Title
    ax.text(
        x_left + width / 2, y_top - 1.3, title,
        ha="center", va="top",
        fontsize=9.0, fontweight="bold",
        color=colors["accent"], zorder=6,
    )

    # Subtitle lines
    line_spacing = 1.55
    start_y = y_top - 3.3
    for i, line in enumerate(subtitle_lines):
        ax.text(
            x_left + width / 2, start_y - i * line_spacing, line,
            ha="center", va="top",
            fontsize=7.3, color=COLOR_BOX_SUBTEXT, zorder=6,
        )


def box_anchor(box_tuple, side):
    """Return (x, y) anchor point on one side of the box."""
    _, x_left, x_right, y_bot, y_top, _, _ = box_tuple
    if side == "right":
        return (x_right, (y_bot + y_top) / 2)
    elif side == "left":
        return (x_left, (y_bot + y_top) / 2)
    elif side == "top":
        return ((x_left + x_right) / 2, y_top)
    elif side == "bottom":
        return ((x_left + x_right) / 2, y_bot)
    raise ValueError(side)


def draw_arrow(ax, from_box, to_box, style):
    """Draw an arrow from one box to another, horizontal or vertical."""
    if style == "h":
        # Horizontal: left edge of right box -> right edge of left box
        # Determine which box is on the left
        _, f_xl, f_xr, f_yb, f_yt, _, _ = from_box
        _, t_xl, t_xr, t_yb, t_yt, _, _ = to_box

        # If same y-range, simple horizontal arrow
        if abs(((f_yb + f_yt) / 2) - ((t_yb + t_yt) / 2)) < 2:
            start = (f_xr + 0.3, (f_yb + f_yt) / 2)
            end = (t_xl - 0.3, (t_yb + t_yt) / 2)
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle="-|>", mutation_scale=10,
                color=COLOR_ARROW, linewidth=1.0, zorder=4,
            )
        else:
            # Curved horizontal (different rows in same lane, e.g., PCMCI -> Ensemble)
            start = (f_xr + 0.3, (f_yb + f_yt) / 2)
            end = (t_xl - 0.3, (t_yb + t_yt) / 2)
            arrow = FancyArrowPatch(
                start, end,
                connectionstyle="arc3,rad=-0.3",
                arrowstyle="-|>", mutation_scale=10,
                color=COLOR_ARROW, linewidth=1.0, zorder=4,
            )
        ax.add_patch(arrow)

    elif style == "v":
        # Vertical: bottom of top box -> top of bottom box
        start = box_anchor(from_box, "bottom")
        end = box_anchor(to_box, "top")
        # Shift start/end slightly off the edge to avoid overlap with box border
        start = (start[0], start[1] - 0.3)
        end = (end[0], end[1] + 0.3)
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=10,
            color=COLOR_ARROW, linewidth=1.0,
            linestyle="-", zorder=4,
        )
        ax.add_patch(arrow)


def render(output_stem: Path):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_aspect("auto")

    # Draw swim lanes
    draw_swim_lanes(ax)

    # Draw boxes
    for box in BOXES:
        lane, x_left, x_right, y_bot, y_top, title, subtitle_lines = box
        draw_box(ax, lane, x_left, x_right, y_bot, y_top, title, subtitle_lines)

    # Draw arrows
    for from_idx, to_idx, style in ARROWS:
        from_box = BOXES[from_idx]
        to_box = BOXES[to_idx]
        draw_arrow(ax, from_box, to_box, style)

    # Figure title
    fig.suptitle(
        "CausalStress system architecture",
        fontsize=13, fontweight="bold", x=0.015, ha="left", y=0.98,
    )
    fig.text(
        0.015, 0.948,
        "End-to-end pipeline: 56-variable market panel \u2192 "
        "HMM regime detection \u2192 causal discovery \u2192 "
        "regime-conditioned Monte Carlo \u2192 validation",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main():
    script_dir = Path(__file__).parent
    output_stem = script_dir / "figure_1_architecture"
    pdf_path, png_path = render(output_stem)
    print(f"\nFigure 1 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
