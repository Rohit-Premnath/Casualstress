"""
Figure 3: Causal Graph (Consensus Edges)
==========================================
Reads causal_graph_data.json and renders a publication-quality network
visualization of the 255 consensus edges (appearing in BOTH DYNOTEARS
and PCMCI) across 56 financial variables.

Layout:
  Grouped circular — nodes placed around a circle with same-category nodes
  arranged in adjacent arcs. Categories (equity, rates, credit, commodity,
  volatility, macro, FX) each get a labeled sector of the circle.

Edge styling:
  - Opacity proportional to consensus score (dyno * pcmci)
  - Thin bezier curves to show direction without clutter
  - Warm-tone edges for same-category (within-cluster)
  - Cool-tone edges for cross-category (contagion paths)

Node styling:
  - Size proportional to total degree (in + out connections)
  - Color from category palette
  - Label positioned radially outside the node

Outputs:
  - figure_3_causal_graph.pdf
  - figure_3_causal_graph.png
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ============================================================
# RC PARAMS (consistent with other figures)
# ============================================================

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
# LAYOUT CONSTANTS
# ============================================================

# Main circle radius where nodes sit
NODE_RADIUS = 1.0

# Category order around the circle (clockwise from 12 o'clock)
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

# Gap between category arcs (radians)
CATEGORY_GAP = 0.12

# Node sizing
NODE_MIN_SIZE = 15
NODE_MAX_SIZE = 200

# Edge styling
EDGE_MIN_ALPHA = 0.08
EDGE_MAX_ALPHA = 0.55
EDGE_BASE_WIDTH = 0.4
EDGE_MAX_WIDTH = 1.3

COLOR_EDGE_WITHIN = "#d97706"   # warm orange: within-category
COLOR_EDGE_CROSS = "#4b6cb7"    # cool blue: cross-category


# ============================================================
# HELPERS
# ============================================================

def assign_node_positions(nodes: List[Dict], category_stats: Dict) -> Dict[str, Tuple[float, float]]:
    """
    Place nodes around a unit circle, grouped by category.
    Returns {node_id: (x, y)}.
    """
    # Group nodes by category
    by_cat = {cat: [] for cat in CATEGORY_ORDER}
    for node in nodes:
        cat = node["category"]
        if cat not in by_cat:
            by_cat["other"].append(node)
        else:
            by_cat[cat].append(node)

    # Sort each category's nodes alphabetically for stable ordering
    for cat in by_cat:
        by_cat[cat].sort(key=lambda n: n["id"])

    # Only include categories with at least one node
    active_cats = [cat for cat in CATEGORY_ORDER if by_cat[cat]]

    # Each category gets an angular arc proportional to its node count
    total_nodes = sum(len(by_cat[c]) for c in active_cats)
    n_gaps = len(active_cats)
    total_gap = n_gaps * CATEGORY_GAP
    usable_angle = 2 * math.pi - total_gap

    positions = {}
    # Start at top (12 o'clock), go clockwise
    current_angle = math.pi / 2  # 90 degrees = top

    category_arc_info = {}  # for legend/label placement

    for cat in active_cats:
        cat_nodes = by_cat[cat]
        n = len(cat_nodes)
        # Angular width of this category's arc
        arc_angle = usable_angle * n / total_nodes

        # Mid-angle of this arc (for category label)
        arc_mid = current_angle - arc_angle / 2
        category_arc_info[cat] = {
            "start_angle": current_angle,
            "end_angle": current_angle - arc_angle,
            "mid_angle": arc_mid,
            "node_count": n,
        }

        # Distribute nodes evenly within the arc
        for i, node in enumerate(cat_nodes):
            if n == 1:
                angle = arc_mid
            else:
                # Place at fractional positions within the arc
                frac = (i + 0.5) / n
                angle = current_angle - frac * arc_angle
            x = NODE_RADIUS * math.cos(angle)
            y = NODE_RADIUS * math.sin(angle)
            positions[node["id"]] = (x, y, angle)

        # Move to next category (with gap)
        current_angle -= arc_angle + CATEGORY_GAP

    return positions, category_arc_info


def compute_node_sizes(nodes: List[Dict]) -> Dict[str, float]:
    """Map total_degree to a size between NODE_MIN_SIZE and NODE_MAX_SIZE."""
    degrees = [n["total_degree"] for n in nodes]
    if not degrees:
        return {}
    max_d = max(degrees) or 1
    min_d = min(degrees)
    sizes = {}
    for n in nodes:
        if max_d == min_d:
            sizes[n["id"]] = (NODE_MIN_SIZE + NODE_MAX_SIZE) / 2
        else:
            # Non-linear scaling (sqrt) so high-degree nodes don't dominate
            frac = (n["total_degree"] - min_d) / (max_d - min_d)
            sizes[n["id"]] = NODE_MIN_SIZE + (NODE_MAX_SIZE - NODE_MIN_SIZE) * math.sqrt(frac)
    return sizes


def draw_curved_edge(ax, src_pos, tgt_pos, color, alpha, width):
    """Draw a curved edge from src to tgt, curving toward the center."""
    x0, y0, _ = src_pos
    x1, y1, _ = tgt_pos

    # Control point: midpoint pulled toward center
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2
    # Pull toward origin
    pull = 0.35
    cx = mx * (1 - pull)
    cy = my * (1 - pull)

    # Quadratic Bezier path
    path = MplPath(
        [(x0, y0), (cx, cy), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
    )
    patch = PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=width, alpha=alpha, zorder=2,
    )
    ax.add_patch(patch)


def draw_directed_edge_with_arrow(ax, src_pos, tgt_pos, node_radius_data,
                                    color, alpha, width):
    """
    Draw a curved arrow from src to tgt, with the arrowhead stopping at
    the edge of the target node rather than at its center.
    """
    x0, y0, _ = src_pos
    x1, y1, _ = tgt_pos

    # Shorten by node radius
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return
    # Scale arrow endpoint back by the target node's radius
    shrink = node_radius_data / length
    x1_adj = x1 - dx * shrink
    y1_adj = y1 - dy * shrink

    # Use FancyArrowPatch with a curved connection style for aesthetic
    arrow = FancyArrowPatch(
        (x0, y0), (x1_adj, y1_adj),
        connectionstyle="arc3,rad=0.18",   # curve amount
        arrowstyle="-|>", mutation_scale=5,
        linewidth=width, color=color, alpha=alpha,
        zorder=2,
    )
    ax.add_patch(arrow)


def render_figure(json_path: Path, output_stem: Path):
    with open(json_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]
    category_stats = data["category_stats"]
    metadata = data["metadata"]

    # Compute layout
    positions, category_arcs = assign_node_positions(nodes, category_stats)
    node_sizes = compute_node_sizes(nodes)

    # Build ID -> node lookup
    node_by_id = {n["id"]: n for n in nodes}

    # Figure layout: main graph panel + legend/stats panel
    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[4.5, 1.2],
        wspace=0.08,
    )
    ax_graph = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # ---------------- Main graph panel ----------------
    ax_graph.set_xlim(-1.70, 1.70)
    ax_graph.set_ylim(-1.70, 1.70)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    # Draw edges (bezier curves)
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in positions or tgt not in positions:
            continue

        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)
        if not src_node or not tgt_node:
            continue

        # Color by within- vs cross-category
        same_cat = src_node["category"] == tgt_node["category"]
        color = COLOR_EDGE_WITHIN if same_cat else COLOR_EDGE_CROSS

        # Scale alpha and width by consensus score
        score = edge.get("consensus_score", 0)
        alpha = EDGE_MIN_ALPHA + (EDGE_MAX_ALPHA - EDGE_MIN_ALPHA) * min(1.0, score)
        width = EDGE_BASE_WIDTH + (EDGE_MAX_WIDTH - EDGE_BASE_WIDTH) * min(1.0, score)

        draw_curved_edge(ax_graph, positions[src], positions[tgt],
                         color, alpha, width)

    # Draw nodes
    for node in nodes:
        x, y, angle = positions[node["id"]]
        cat = node["category"]
        color = category_stats.get(cat, {}).get("color", "#999")
        size = node_sizes[node["id"]]

        ax_graph.scatter(
            x, y, s=size, color=color,
            edgecolor="white", linewidth=0.8, zorder=5,
        )

        # Label placement: radially outside the node, pushed out
        label_r = NODE_RADIUS + 0.055
        lx = label_r * math.cos(angle)
        ly = label_r * math.sin(angle)

        # Text rotation for readability (tangent to circle)
        rotation_deg = math.degrees(angle)
        # Keep text reading left-to-right (never upside down)
        if rotation_deg < -90 or rotation_deg > 90:
            rotation_deg += 180
            ha = "right"
        else:
            ha = "left"
        # Simplify special chars for display
        label = node["label"]
        display_label = label.replace("^", "").replace("=F", "").replace("=X", "")
        # Truncate extremely long labels
        if len(display_label) > 12:
            display_label = display_label[:11] + "\u2026"

        ax_graph.text(
            lx, ly, display_label,
            ha=ha, va="center", rotation=rotation_deg,
            fontsize=6.8, color="#1a1a1a",
            rotation_mode="anchor", zorder=6,
        )

    # Draw category arc labels (pushed further out beyond node labels)
    for cat, arc_info in category_arcs.items():
        if cat not in category_stats:
            continue
        stats = category_stats[cat]
        if stats["node_count"] == 0:
            continue
        mid = arc_info["mid_angle"]
        # Push category label well outside node labels
        label_r = NODE_RADIUS + 0.48
        lx = label_r * math.cos(mid)
        ly = label_r * math.sin(mid)
        # Keep category text upright
        rotation_deg = math.degrees(mid)
        if rotation_deg < -90 or rotation_deg > 90:
            rotation_deg += 180
        ax_graph.text(
            lx, ly,
            f"{stats['label'].upper()}",
            ha="center", va="center", rotation=rotation_deg,
            fontsize=10.5, fontweight="bold",
            color=stats["color"],
            rotation_mode="anchor", zorder=7,
        )

    # ---------------- Legend / stats panel ----------------
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    # Title
    ax_legend.text(
        0.5, 0.97, "Graph statistics",
        ha="center", va="top", fontsize=10, fontweight="bold",
        transform=ax_legend.transAxes,
    )

    y_pos = 0.90
    line_h = 0.03

    # Overview stats
    overview = [
        f"Method: {metadata['method']}",
        f"Nodes:  {metadata['node_count']}",
        f"Total edges (ensemble):  {metadata['total_edges']:,}",
        f"Consensus edges shown:  {metadata['consensus_edges']:,}",
        f"",
        f"Edge = in BOTH DYNOTEARS and PCMCI",
        f"Opacity = consensus score (dyno \u00d7 pcmci)",
    ]
    for line in overview:
        if line:
            ax_legend.text(0.04, y_pos, line, fontsize=8.3, color="#1a1a1a",
                          transform=ax_legend.transAxes)
        y_pos -= line_h

    y_pos -= 0.02

    # Category legend
    ax_legend.text(0.04, y_pos, "Categories", fontweight="bold",
                   fontsize=9, color="#1a1a1a", transform=ax_legend.transAxes)
    y_pos -= line_h * 1.3

    for cat_key in CATEGORY_ORDER:
        if cat_key not in category_stats:
            continue
        stats = category_stats[cat_key]
        if stats["node_count"] == 0:
            continue

        # Color swatch
        ax_legend.add_patch(Circle(
            (0.06, y_pos + 0.005), 0.012,
            facecolor=stats["color"], edgecolor="white", linewidth=0.6,
            transform=ax_legend.transAxes,
        ))
        # Label
        ax_legend.text(
            0.10, y_pos, f"{stats['label']}",
            fontsize=8.3, color="#1a1a1a", va="center",
            transform=ax_legend.transAxes,
        )
        # Node count
        ax_legend.text(
            0.62, y_pos, f"{stats['node_count']} nodes",
            fontsize=7.8, color="#555", va="center",
            transform=ax_legend.transAxes,
        )
        # Total degree
        ax_legend.text(
            0.86, y_pos, f"{stats['total_degree']}",
            fontsize=7.8, color="#555", va="center",
            transform=ax_legend.transAxes,
        )
        y_pos -= line_h

    y_pos -= 0.02
    ax_legend.text(0.04, y_pos, "Edges", fontweight="bold",
                   fontsize=9, color="#1a1a1a", transform=ax_legend.transAxes)
    y_pos -= line_h * 1.3

    # Edge legend: within-category vs cross-category
    for label, color in [("within-category", COLOR_EDGE_WITHIN),
                         ("cross-category (contagion)", COLOR_EDGE_CROSS)]:
        ax_legend.plot(
            [0.05, 0.09], [y_pos + 0.005, y_pos + 0.005],
            color=color, linewidth=1.2, alpha=0.7,
            transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.12, y_pos, label,
            fontsize=8.3, color="#1a1a1a", va="center",
            transform=ax_legend.transAxes,
        )
        y_pos -= line_h

    y_pos -= 0.02
    ax_legend.text(0.04, y_pos, "Node size = total degree",
                   fontsize=8, color="#555", fontstyle="italic",
                   transform=ax_legend.transAxes)

    # ---------------- Figure title ----------------
    # Count within vs cross edges
    within_count = sum(
        1 for e in edges
        if node_by_id.get(e["source"], {}).get("category") ==
           node_by_id.get(e["target"], {}).get("category")
    )
    cross_count = len(edges) - within_count

    fig.suptitle(
        "Consensus causal graph across 56 financial variables",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.985,
    )
    fig.text(
        0.015, 0.955,
        f"{metadata['consensus_edges']} edges identified by BOTH DYNOTEARS and PCMCI "
        f"({within_count} within-category, {cross_count} cross-category contagion paths)",
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
        script_dir / "causal_graph_data.json",
        script_dir.parent / "ml_pipeline" / "causal_graph_data.json",
        Path.cwd() / "causal_graph_data.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("ERROR: causal_graph_data.json not found.")
        print("Run causal_graph_extract.py first.")
        return

    print(f"Loading: {json_path}")
    output_stem = script_dir / "figure_3_causal_graph"

    pdf_path, png_path = render_figure(json_path, output_stem)

    print(f"\nFigure 3 complete.")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
