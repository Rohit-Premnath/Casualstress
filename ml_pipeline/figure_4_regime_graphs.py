"""
Figure 4: Regime-Conditional Causal Graphs (Calm vs Stressed)
==============================================================
Side-by-side network visualization showing how the causal structure of
financial markets REWIRES between calm and stressed regimes.

Layout (2 columns + 1 legend column):
  Left:  CALM regime graph    — shared edges + calm-only edges
  Right: STRESSED regime graph — shared edges + stress-only edges
  Right column: legend + diff stats

Edge coloring:
  - Shared edges (in both regimes): neutral gray, low opacity
  - Regime-specific edges: highlighted in contrast color
    - Calm-only: cool teal ("relationships that break under stress")
    - Stress-only: warm red-orange ("contagion paths")

The story: visually apparent that stressed graph has MORE edges, and the
new edges crisscross the center (cross-category contagion paths).

Outputs:
  - figure_4_regime_graphs.pdf
  - figure_4_regime_graphs.png
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# LAYOUT CONSTANTS (aligned with Figure 3)
# ============================================================

NODE_RADIUS = 1.0

CATEGORY_ORDER = [
    "equity_index",
    "equity_sector",
    "volatility",
    "rates",
    "credit",
    "commodity",
    "fx",
    "macro",
    "other",
]

CATEGORY_GAP = 0.12

NODE_MIN_SIZE = 12
NODE_MAX_SIZE = 140

# Edge styling
EDGE_SHARED_COLOR = "#a0a0a0"       # muted gray for edges present in both
EDGE_SHARED_ALPHA_MIN = 0.10
EDGE_SHARED_ALPHA_MAX = 0.40

EDGE_CALM_ONLY_COLOR = "#14919b"    # teal — "broken" relationships
EDGE_STRESS_ONLY_COLOR = "#d62728"  # red — contagion paths

EDGE_HIGHLIGHT_ALPHA_MIN = 0.35
EDGE_HIGHLIGHT_ALPHA_MAX = 0.80

EDGE_BASE_WIDTH = 0.35
EDGE_MAX_WIDTH = 1.2


# ============================================================
# LAYOUT HELPERS (shared between panels)
# ============================================================

def assign_node_positions(nodes: List[Dict]):
    """Place nodes around a unit circle, grouped by category."""
    by_cat = {cat: [] for cat in CATEGORY_ORDER}
    for node in nodes:
        cat = node["category"]
        if cat not in by_cat:
            by_cat.setdefault("other", []).append(node)
        else:
            by_cat[cat].append(node)

    for cat in by_cat:
        by_cat[cat].sort(key=lambda n: n["id"])

    active_cats = [cat for cat in CATEGORY_ORDER if by_cat[cat]]
    total_nodes = sum(len(by_cat[c]) for c in active_cats)
    n_gaps = len(active_cats)
    usable_angle = 2 * math.pi - n_gaps * CATEGORY_GAP

    positions = {}
    category_arcs = {}
    current_angle = math.pi / 2  # top

    for cat in active_cats:
        cat_nodes = by_cat[cat]
        n = len(cat_nodes)
        arc_angle = usable_angle * n / total_nodes
        arc_mid = current_angle - arc_angle / 2
        category_arcs[cat] = {
            "start_angle": current_angle,
            "end_angle": current_angle - arc_angle,
            "mid_angle": arc_mid,
            "node_count": n,
        }
        for i, node in enumerate(cat_nodes):
            frac = (i + 0.5) / n if n > 1 else 0.5
            angle = current_angle - frac * arc_angle
            x = NODE_RADIUS * math.cos(angle)
            y = NODE_RADIUS * math.sin(angle)
            positions[node["id"]] = (x, y, angle)
        current_angle -= arc_angle + CATEGORY_GAP

    return positions, category_arcs


def compute_node_sizes(nodes, degree_field):
    """Size by a specific degree field (calm_degree or stressed_degree)."""
    degrees = [n[degree_field] for n in nodes]
    if not degrees or max(degrees) == 0:
        return {n["id"]: NODE_MIN_SIZE for n in nodes}
    max_d = max(degrees)
    min_d = min(degrees)
    sizes = {}
    for n in nodes:
        if max_d == min_d:
            sizes[n["id"]] = (NODE_MIN_SIZE + NODE_MAX_SIZE) / 2
        else:
            frac = (n[degree_field] - min_d) / (max_d - min_d)
            sizes[n["id"]] = NODE_MIN_SIZE + (NODE_MAX_SIZE - NODE_MIN_SIZE) * math.sqrt(frac)
    return sizes


def draw_curved_edge(ax, src_pos, tgt_pos, color, alpha, width):
    x0, y0, _ = src_pos
    x1, y1, _ = tgt_pos
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    pull = 0.35
    cx, cy = mx * (1 - pull), my * (1 - pull)
    path = MplPath(
        [(x0, y0), (cx, cy), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
    )
    patch = PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=width, alpha=alpha, zorder=2,
    )
    ax.add_patch(patch)


def edge_alpha_from_weight(weight, min_a, max_a, scale=0.5):
    """Map |weight| to alpha in [min_a, max_a]. scale controls saturation point."""
    a = abs(weight) / scale
    frac = min(1.0, a)
    return min_a + (max_a - min_a) * frac


def edge_width_from_weight(weight, scale=0.5):
    a = abs(weight) / scale
    frac = min(1.0, a)
    return EDGE_BASE_WIDTH + (EDGE_MAX_WIDTH - EDGE_BASE_WIDTH) * frac


# ============================================================
# DRAW ONE PANEL (calm or stressed)
# ============================================================

def draw_panel(ax, nodes, positions, category_arcs, category_stats,
               shared_edges, unique_edges, node_sizes,
               panel_label: str, unique_edge_color: str, unique_edge_label: str,
               n_edges_total: int):
    """
    Draw a single regime graph panel.

    shared_edges: edges that exist in BOTH regimes (drawn gray)
    unique_edges: edges UNIQUE to this regime (drawn in highlight color)
    """
    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.60, 1.60)
    ax.set_aspect("equal")
    ax.axis("off")

    node_by_id = {n["id"]: n for n in nodes}

    # Draw shared edges first (background layer)
    for edge in shared_edges:
        src, tgt = edge["source"], edge["target"]
        if src not in positions or tgt not in positions:
            continue
        # Use the regime-specific weight for this panel
        weight = edge.get("stressed_weight" if panel_label == "Stressed" else "calm_weight",
                          edge.get("weight", 0))
        alpha = edge_alpha_from_weight(weight, EDGE_SHARED_ALPHA_MIN, EDGE_SHARED_ALPHA_MAX)
        width = edge_width_from_weight(weight)
        draw_curved_edge(ax, positions[src], positions[tgt],
                         EDGE_SHARED_COLOR, alpha, width)

    # Draw unique (regime-specific) edges on top
    for edge in unique_edges:
        src, tgt = edge["source"], edge["target"]
        if src not in positions or tgt not in positions:
            continue
        weight = edge["weight"]
        alpha = edge_alpha_from_weight(weight, EDGE_HIGHLIGHT_ALPHA_MIN,
                                       EDGE_HIGHLIGHT_ALPHA_MAX)
        width = edge_width_from_weight(weight) + 0.2  # slightly thicker
        draw_curved_edge(ax, positions[src], positions[tgt],
                         unique_edge_color, alpha, width)

    # Draw nodes
    for node in nodes:
        x, y, angle = positions[node["id"]]
        cat = node["category"]
        color = category_stats.get(cat, {}).get("color", "#999")
        size = node_sizes[node["id"]]
        ax.scatter(x, y, s=size, color=color,
                   edgecolor="white", linewidth=0.6, zorder=5)

        # Label
        label_r = NODE_RADIUS + 0.055
        lx = label_r * math.cos(angle)
        ly = label_r * math.sin(angle)
        rotation_deg = math.degrees(angle)
        if rotation_deg < -90 or rotation_deg > 90:
            rotation_deg += 180
            ha = "right"
        else:
            ha = "left"
        display_label = node["label"].replace("^", "").replace("=F", "").replace("=X", "")
        if len(display_label) > 11:
            display_label = display_label[:10] + "\u2026"
        ax.text(lx, ly, display_label,
                ha=ha, va="center", rotation=rotation_deg,
                fontsize=5.5, color="#1a1a1a",
                rotation_mode="anchor", zorder=6)

    # Category labels (outer ring)
    for cat, arc_info in category_arcs.items():
        if cat not in category_stats or category_stats[cat]["node_count"] == 0:
            continue
        mid = arc_info["mid_angle"]
        label_r = NODE_RADIUS + 0.42
        lx = label_r * math.cos(mid)
        ly = label_r * math.sin(mid)
        rotation_deg = math.degrees(mid)
        if rotation_deg < -90 or rotation_deg > 90:
            rotation_deg += 180
        ax.text(lx, ly, category_stats[cat]["label"].upper(),
                ha="center", va="center", rotation=rotation_deg,
                fontsize=8.5, fontweight="bold",
                color=category_stats[cat]["color"],
                rotation_mode="anchor", zorder=7)

    # Panel title
    panel_color = "#1f3d7a" if panel_label == "Calm" else "#b22222"
    ax.set_title(
        f"{panel_label} regime  \u00b7  {n_edges_total} total edges  "
        f"(+{len(unique_edges)} {unique_edge_label})",
        loc="center", fontsize=11, fontweight="bold",
        color=panel_color, pad=6,
    )


# ============================================================
# LEGEND PANEL
# ============================================================

def draw_legend(ax, metadata, category_stats):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "Regime diff", ha="center", va="top",
            fontsize=10.5, fontweight="bold", transform=ax.transAxes)

    y = 0.90
    dy = 0.028

    # Summary stats
    summary_lines = [
        (f"Calm graph:",       f"{metadata['calm_edge_count']} edges"),
        (f"Stressed graph:",   f"{metadata['stress_edge_count']} edges"),
        (f"Shared (in both):", f"{metadata['shared_count']}"),
        (f"Stress-only:",      f"+{metadata['stress_only_count']}"),
        (f"Calm-only:",        f"+{metadata['calm_only_count']}"),
    ]
    for label, val in summary_lines:
        ax.text(0.04, y, label, fontsize=8.3, color="#1a1a1a",
                transform=ax.transAxes)
        ax.text(0.98, y, val, fontsize=8.3, color="#1a1a1a",
                fontweight="bold", ha="right", transform=ax.transAxes)
        y -= dy

    y -= 0.015
    ax.text(0.04, y, "Edge color key", fontsize=9.5, fontweight="bold",
            color="#1a1a1a", transform=ax.transAxes)
    y -= dy * 1.3

    # Shared
    ax.plot([0.05, 0.12], [y + 0.005, y + 0.005],
            color=EDGE_SHARED_COLOR, linewidth=1.3, alpha=0.6,
            transform=ax.transAxes)
    ax.text(0.15, y, "shared (both regimes)", fontsize=8.2,
            color="#1a1a1a", va="center", transform=ax.transAxes)
    y -= dy

    # Calm-only
    ax.plot([0.05, 0.12], [y + 0.005, y + 0.005],
            color=EDGE_CALM_ONLY_COLOR, linewidth=1.5,
            transform=ax.transAxes)
    ax.text(0.15, y, "calm-only (breaks)", fontsize=8.2,
            color="#1a1a1a", va="center", transform=ax.transAxes)
    y -= dy

    # Stress-only
    ax.plot([0.05, 0.12], [y + 0.005, y + 0.005],
            color=EDGE_STRESS_ONLY_COLOR, linewidth=1.5,
            transform=ax.transAxes)
    ax.text(0.15, y, "stress-only (contagion)", fontsize=8.2,
            color="#1a1a1a", va="center", transform=ax.transAxes)
    y -= dy

    y -= 0.02
    ax.text(0.04, y, "Node categories", fontsize=9.5, fontweight="bold",
            color="#1a1a1a", transform=ax.transAxes)
    y -= dy * 1.2

    for cat_key in CATEGORY_ORDER:
        if cat_key not in category_stats:
            continue
        stats = category_stats[cat_key]
        if stats["node_count"] == 0:
            continue
        ax.add_patch(Circle(
            (0.06, y + 0.005), 0.010,
            facecolor=stats["color"], edgecolor="white", linewidth=0.5,
            transform=ax.transAxes,
        ))
        ax.text(0.11, y, stats["label"], fontsize=8,
                color="#1a1a1a", va="center", transform=ax.transAxes)
        ax.text(0.98, y, f"{stats['node_count']}", fontsize=7.8,
                color="#555", va="center", ha="right", transform=ax.transAxes)
        y -= dy * 0.95

    y -= 0.02
    ax.text(0.04, y,
            "Node size = degree within that regime",
            fontsize=7.5, color="#555", fontstyle="italic",
            transform=ax.transAxes)
    y -= dy
    ax.text(0.04, y,
            "Edge opacity = |weight|",
            fontsize=7.5, color="#555", fontstyle="italic",
            transform=ax.transAxes)


# ============================================================
# MAIN RENDER
# ============================================================

def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    shared_edges = data["shared_edges"]
    stress_only_edges = data["stress_only_edges"]
    calm_only_edges = data["calm_only_edges"]
    category_stats = data["category_stats"]
    metadata = data["metadata"]

    # Compute layout (shared between panels)
    positions, category_arcs = assign_node_positions(nodes)
    calm_sizes = compute_node_sizes(nodes, "calm_degree")
    stress_sizes = compute_node_sizes(nodes, "stressed_degree")

    # Layout: 2 graph panels + 1 legend panel
    fig = plt.figure(figsize=(17.0, 8.5))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[4.5, 4.5, 1.4],
        wspace=0.03,
    )
    ax_calm = fig.add_subplot(gs[0])
    ax_stress = fig.add_subplot(gs[1])
    ax_legend = fig.add_subplot(gs[2])

    # Draw calm panel (shared + calm-only edges)
    draw_panel(
        ax_calm, nodes, positions, category_arcs, category_stats,
        shared_edges=shared_edges,
        unique_edges=calm_only_edges,
        node_sizes=calm_sizes,
        panel_label="Calm",
        unique_edge_color=EDGE_CALM_ONLY_COLOR,
        unique_edge_label="calm-only",
        n_edges_total=metadata["calm_edge_count"],
    )

    # Draw stressed panel (shared + stress-only edges)
    draw_panel(
        ax_stress, nodes, positions, category_arcs, category_stats,
        shared_edges=shared_edges,
        unique_edges=stress_only_edges,
        node_sizes=stress_sizes,
        panel_label="Stressed",
        unique_edge_color=EDGE_STRESS_ONLY_COLOR,
        unique_edge_label="stress-only",
        n_edges_total=metadata["stress_edge_count"],
    )

    # Draw legend
    draw_legend(ax_legend, metadata, category_stats)

    # Figure title
    fig.suptitle(
        "The causal graph rewires under stress: "
        f"+{metadata['stress_only_count']} contagion edges appear, "
        f"{metadata['calm_only_count']} relationships break",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.985,
    )
    fig.text(
        0.015, 0.952,
        f"Regime-conditional causal graphs across 56 financial variables  \u00b7  "
        f"shared edges shown in gray  \u00b7  "
        f"regime-specific edges highlighted in color",
        fontsize=9, color="#555555", ha="left", fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return pdf_path, png_path


def main():
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "regime_graphs_data.json",
        script_dir.parent / "ml_pipeline" / "regime_graphs_data.json",
        Path.cwd() / "regime_graphs_data.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: regime_graphs_data.json not found.")
        print("Run regime_graphs_extract.py first.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_4_regime_graphs"

    pdf_path, png_path = render_figure(json_path, output_stem)

    print(f"\nFigure 4 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()