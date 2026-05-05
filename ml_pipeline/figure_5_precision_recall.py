"""
Figure 5: Precision-Recall Curve for Causal Discovery
=======================================================
Reads precision_at_k_v2.json (from precision_at_k_v2.py) and renders a
publication-quality precision-recall curve showing how the consensus-product
ranking identifies ground-truth economic relationships.

What it shows:
  - X-axis: recall (fraction of 25 ground-truth edges recovered)
  - Y-axis: precision (fraction of selected edges that are ground truth)
  - Main curve: consensus_product (the winning strategy, 0.1677 PR-AUC)
  - Dashed: random baseline (25/1249 = 0.02)
  - Marked points: k=10, 25, 50, 100, 200, 500, k_full
  - Lift-over-baseline shaded region

Outputs:
  - figure_5_precision_recall.pdf  (vector)
  - figure_5_precision_recall.png  (raster, 300 DPI)

The figure complements Figure 8 (scenario-level evidence) by showing
mechanism-level validation: the ensemble finds the right relationships, not
just noise.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


# ============================================================
# RC PARAMS (matching Figures 6 & 8 for visual consistency)
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
    "axes.labelcolor": "#1a1a1a",
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
# COLORS (consistent with Figures 6 & 8)
# ============================================================

COLOR_MAIN_CURVE = "#1f3d7a"          # deep blue for winner
COLOR_MAIN_FILL = "#7a9dd4"           # lighter blue for lift shading
COLOR_SECONDARY = "#8a5aa3"           # muted purple for runner-up
COLOR_SECONDARY_LIGHT = "#c9a3e0"
COLOR_BASELINE = "#b22222"            # red for random baseline
COLOR_ANNOTATION = "#1a1a1a"
COLOR_GROUND_TRUTH = "#3a7d44"        # green for GT markers


# Strategies to plot (by priority)
STRATEGIES = [
    {
        "key": "consensus_product",
        "display": "Consensus (DYNOTEARS \u00d7 PCMCI)",
        "color": COLOR_MAIN_CURVE,
        "fill": COLOR_MAIN_FILL,
        "linewidth": 2.0,
        "zorder": 10,
        "is_main": True,
    },
    {
        "key": "dynotears_only",
        "display": "DYNOTEARS only",
        "color": COLOR_SECONDARY,
        "fill": COLOR_SECONDARY_LIGHT,
        "linewidth": 1.3,
        "zorder": 6,
        "is_main": False,
    },
    {
        "key": "pcmci_only",
        "display": "PCMCI only",
        "color": "#d97706",            # orange
        "fill": "#f5c088",
        "linewidth": 1.3,
        "zorder": 5,
        "is_main": False,
    },
]

# k values to mark on the main curve
K_MARKERS = [10, 25, 50, 100, 200, 500]


# ============================================================
# DATA LOADING
# ============================================================

def build_pr_curve_from_trajectory(trajectory: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of {k, precision, recall, ...} dicts, sort by recall ascending
    and return (recall_array, precision_array).
    """
    sorted_traj = sorted(trajectory, key=lambda r: r["recall"])
    recall = np.array([r["recall"] for r in sorted_traj])
    precision = np.array([r["precision"] for r in sorted_traj])
    return recall, precision


def load_strategy_curves(data: Dict) -> Dict[str, Dict]:
    """
    Extract PR-curve data for each strategy from the JSON blob.

    Schema (from precision_at_k_v2.py):
      data["strategies"][strategy_name] = {
          "per_k": [{"k": N, "tp": N, "precision": float, "recall": float, "f1": float}, ...],
          "k_full_recovery": int,
          "pr_auc": float,
          "top_10_edges": [...],
      }
    """
    strategies_raw = data.get("strategies", {})
    out = {}

    for s in STRATEGIES:
        key = s["key"]
        if key not in strategies_raw:
            available = list(strategies_raw.keys())
            print(f"  WARNING: strategy '{key}' not found. Available: {available}")
            continue

        raw = strategies_raw[key]
        per_k = raw.get("per_k", [])

        if not per_k:
            print(f"  WARNING: strategy '{key}' has empty 'per_k' list")
            continue

        # Sort by k ascending to build a proper PR trajectory
        per_k_sorted = sorted(per_k, key=lambda r: r["k"])

        recall = np.array([r["recall"] for r in per_k_sorted])
        precision = np.array([r["precision"] for r in per_k_sorted])

        # Extract k-marker points for the main curve (exact precision/recall at each k)
        k_markers = {}
        for record in per_k_sorted:
            k = record["k"]
            if k in K_MARKERS:
                k_markers[k] = (record["recall"], record["precision"])

        out[key] = {
            "recall": recall,
            "precision": precision,
            "pr_auc": raw.get("pr_auc"),
            "k_markers": k_markers,
            "k_full_recovery": raw.get("k_full_recovery"),
            "per_k_raw": per_k_sorted,
        }

    return out


# ============================================================
# RENDER
# ============================================================

def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    curves = load_strategy_curves(data)
    if "consensus_product" not in curves:
        raise SystemExit("ERROR: consensus_product strategy not found — cannot render Figure 5")

    # Ground truth info from metadata (support multiple key names)
    n_gt = (
        data.get("n_ground_truth")
        or data.get("n_ground_truth_edges")
        or data.get("ground_truth_size")
        or 25
    )
    n_edges = (
        data.get("n_discovered_total")
        or data.get("total_edges")
        or data.get("n_edges")
        or 1249
    )
    random_baseline = n_gt / n_edges

    # Figure layout
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # ----- Plot each strategy curve -----
    for s in STRATEGIES:
        if s["key"] not in curves:
            continue
        c = curves[s["key"]]
        recall, precision = c["recall"], c["precision"]

        # Include origin and (1, random_baseline) for complete curve
        # Prepend (0, precision[0]) if recall doesn't start at 0
        r_plot = np.concatenate([[0.0], recall])
        p_plot = np.concatenate([[precision[0]], precision])

        # Main strategy gets filled area
        if s["is_main"]:
            ax.fill_between(
                r_plot, random_baseline, p_plot,
                where=(p_plot > random_baseline),
                color=s["fill"], alpha=0.25, zorder=s["zorder"] - 1,
                label=f"Lift over random baseline",
            )

        ax.plot(
            r_plot, p_plot,
            color=s["color"], linewidth=s["linewidth"],
            label=f"{s['display']}"
                  + (f"  (PR-AUC = {c['pr_auc']:.3f})" if c.get("pr_auc") else ""),
            zorder=s["zorder"],
            solid_joinstyle="round",
        )

    # ----- Random baseline -----
    ax.axhline(
        random_baseline, color=COLOR_BASELINE, linewidth=1.2,
        linestyle=(0, (5, 3)), zorder=3,
        label=f"Random baseline ({n_gt}/{n_edges} = {random_baseline:.3f})",
    )

    # ----- k-marker annotations on main curve -----
    main_curve = curves["consensus_product"]
    for k, (r, p) in main_curve["k_markers"].items():
        ax.scatter(
            [r], [p], s=35, color=COLOR_MAIN_CURVE,
            edgecolor="white", linewidth=1.0, zorder=11,
        )
        # Smart offsetting to avoid overlap
        if k == 10:
            xy_offset = (8, 8)
        elif k == 25:
            xy_offset = (8, 6)
        elif k == 50:
            xy_offset = (10, 0)
        elif k == 100:
            xy_offset = (8, -10)
        elif k == 200:
            xy_offset = (8, -8)
        else:  # 500 — push up and left to avoid full-recovery marker
            xy_offset = (-8, 10)
        ax.annotate(
            f"k = {k}",
            xy=(r, p), xytext=xy_offset, textcoords="offset points",
            fontsize=7.8, color=COLOR_ANNOTATION,
            ha="left" if xy_offset[0] > 0 else "right",
            va="center",
        )

    # ----- Full-recovery marker -----
    k_full = main_curve.get("k_full_recovery")
    if k_full:
        # At recall=1.0, precision = n_gt / k_full
        p_at_full = n_gt / k_full
        ax.scatter(
            [1.0], [p_at_full], s=80, marker="*",
            color=COLOR_GROUND_TRUTH, edgecolor="white", linewidth=1.2, zorder=12,
            label=f"Full recovery (k = {k_full})",
        )
        # Annotation positioned up-left with a short connector line
        ax.annotate(
            f"100% recall\nat k = {k_full}",
            xy=(1.0, p_at_full), xytext=(0.86, 0.14),
            textcoords="data",
            fontsize=7.8, color=COLOR_GROUND_TRUTH, fontweight="bold",
            ha="center", va="bottom",
            arrowprops=dict(
                arrowstyle="-", color=COLOR_GROUND_TRUTH,
                linewidth=0.6, alpha=0.7,
            ),
        )

    # ----- Axes -----
    ax.set_xlabel("Recall (fraction of 25 ground-truth edges recovered)")
    ax.set_ylabel("Precision (fraction of selected edges that are ground truth)")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0, max(0.45, main_curve["precision"].max() * 1.15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    # ----- Legend -----
    # Reorder legend so main strategy appears first
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles, labels,
        loc="upper right", ncol=1, fontsize=8.3,
        borderpad=0.6, handlelength=2.2, handletextpad=0.7,
    )
    leg.get_frame().set_linewidth(0.5)

    # ----- Title -----
    fig.suptitle(
        "Precision-recall curve for causal discovery",
        fontsize=12, fontweight="bold", x=0.045, ha="left", y=0.978,
    )
    top10_precision = main_curve["k_markers"].get(10, (None, main_curve["precision"][0]))[1]
    top10_lift = top10_precision / random_baseline

    fig.text(
        0.045, 0.938,
        f"Ensemble (DYNOTEARS + PCMCI) recovers all {n_gt} textbook economic edges; "
        f"top-10 precision is {top10_lift:.0f}\u00d7 the random baseline",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path, main_curve


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "precision_at_k_v2.json",
        script_dir / "precision_at_k.json",
        script_dir.parent / "ml_pipeline" / "precision_at_k_v2.json",
        Path.cwd() / "precision_at_k_v2.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: precision_at_k_v2.json not found.")
        print("Expected locations tried:")
        for p in candidates:
            print(f"  {p}")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_5_precision_recall"

    try:
        pdf_path, png_path, main_curve = render_figure(json_path, output_stem)
    except SystemExit as e:
        print(e)
        return

    print(f"\nFigure 5 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    if main_curve.get("pr_auc"):
        print(f"  PR-AUC: {main_curve['pr_auc']:.4f}")
    if main_curve.get("k_full_recovery"):
        print(f"  Full recall at k = {main_curve['k_full_recovery']}")


if __name__ == "__main__":
    main()
