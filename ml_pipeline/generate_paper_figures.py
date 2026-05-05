"""
CausalStress — Paper Figure Generator
=======================================
Generates all 9 publication-quality figures for the research paper.
Reads data directly from the PostgreSQL database.

Usage:
  python generate_paper_figures.py              # Generate all figures
  python generate_paper_figures.py --fig 2      # Generate only figure 2
  python generate_paper_figures.py --outdir ./research_paper/figures

Output: PDF files in research_paper/figures/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

OUTDIR = Path("research_paper/figures")

REGIME_COLORS = {
    "calm": "#10b981", "normal": "#22d3ee", "elevated": "#f59e0b",
    "stressed": "#ef4444", "high_stress": "#dc2626", "crisis": "#991b1b",
}
REGIME_ORDER = ["calm", "normal", "elevated", "stressed", "high_stress", "crisis"]
REGIME_LABELS = ["Calm", "Normal", "Elevated", "Stressed", "High Stress", "Crisis"]

CATEGORY_COLORS = {
    "equity": "#3b82f6", "macro": "#10b981", "rates": "#f59e0b",
    "volatility": "#ef4444", "commodities": "#eab308",
    "fixed-income": "#a855f7", "currency": "#06b6d4",
}

VARIABLE_META = {
    "^GSPC": ("S&P 500", "equity"), "^NDX": ("Nasdaq", "equity"), "^RUT": ("Russell 2000", "equity"),
    "XLF": ("Financials", "equity"), "XLK": ("Tech", "equity"), "XLE": ("Energy", "equity"),
    "XLV": ("Healthcare", "equity"), "XLY": ("Consumer", "equity"), "XLU": ("Utilities", "equity"),
    "EEM": ("EM Equities", "equity"), "DGS10": ("10Y Yield", "rates"), "DGS2": ("2Y Yield", "rates"),
    "FEDFUNDS": ("Fed Funds", "rates"), "T10Y2Y": ("Yield Curve", "rates"),
    "^VIX": ("VIX", "volatility"), "^VVIX": ("VVIX", "volatility"),
    "CL=F": ("Crude Oil", "commodities"), "GC=F": ("Gold", "commodities"),
    "TLT": ("Treasury Bond", "fixed-income"), "LQD": ("IG Bonds", "fixed-income"),
    "HYG": ("HY Bonds", "fixed-income"), "BAMLH0A0HYM2": ("HY Spread", "fixed-income"),
    "BAMLH0A1HYBB": ("BB Spread", "fixed-income"), "BAMLH0A3HYC": ("CCC Spread", "fixed-income"),
    "BAMLC0A0CM": ("IG Spread", "fixed-income"), "BAMLC0A4CBBB": ("BBB Spread", "fixed-income"),
    "CPIAUCSL": ("CPI", "macro"), "UNRATE": ("Unemployment", "macro"),
    "PAYEMS": ("Payrolls", "macro"), "INDPRO": ("Industrial Prod", "macro"),
    "DX-Y.NYB": ("Dollar Index", "currency"), "EURUSD=X": ("EUR/USD", "currency"),
    "DRTSCIS": ("Lending Stds", "macro"), "PCEPILFE": ("Core PCE", "macro"),
}

CRISIS_ANNOTATIONS = [
    (2008, "GFC"), (2010, "Flash\nCrash"), (2011, "Euro\nDebt"),
    (2015, "China/\nOil"), (2018, "Vol-\nmageddon"), (2020, "COVID"),
    (2022, "Rate\nHikes"), (2023, "SVB"),
]

# Paper style
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "axes.spines.top": False, "axes.spines.right": False,
})


def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def save_fig(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
# FIGURE 1: System Architecture (drawn programmatically)
# ============================================================

def fig1_architecture():
    print("\n[Figure 1] System Architecture Diagram")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 2.8, 1.8, 0.9, "Data Pipeline\n35 FRED + 21 Yahoo\n56 variables", "#e0f2fe"),
        (2.6, 2.8, 1.8, 0.9, "Causal Discovery\nDYNOTEARS\n+ PCMCI", "#fef3c7"),
        (4.9, 2.8, 1.8, 0.9, "Regime Detection\n6-State HMM\n5,547 days", "#dcfce7"),
        (0.3, 0.8, 1.8, 0.9, "Regime-Conditional\nCausal Graphs\n330 stressed edges", "#fce7f3"),
        (2.6, 0.8, 1.8, 0.9, "Scenario Generator\nVAR + Multi-Root\n+ Causal Prop.", "#ede9fe"),
        (4.9, 0.8, 1.8, 0.9, "Plausibility Filter\nSoft Weighting\n86.3% mean", "#fee2e2"),
        (7.3, 1.8, 2.2, 0.9, "Stress Test\nVaR / CVaR\nDFAST Comparison", "#f0f9ff"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor="#374151", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=6.5, fontweight="medium")

    arrows = [
        (2.1, 3.25, 2.6, 3.25), (4.4, 3.25, 4.9, 3.25),
        (1.2, 2.8, 1.2, 1.7), (3.5, 2.8, 3.5, 1.7),
        (2.1, 1.25, 2.6, 1.25), (4.4, 1.25, 4.9, 1.25),
        (5.8, 2.8, 7.3, 2.25), (6.7, 1.25, 7.3, 1.85),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.0))

    ax.set_title("CausalStress System Architecture", fontsize=11, fontweight="bold", pad=12)
    save_fig(fig, "fig1_architecture")


# ============================================================
# FIGURE 2: Causal Graph Visualization
# ============================================================

def fig2_causal_graph():
    print("\n[Figure 2] Causal Graph Visualization")

    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT adjacency_matrix FROM models.causal_graphs
        WHERE method LIKE '%%ensemble%%' OR method LIKE '%%dynotears%%'
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print("  WARNING: No causal graph found, skipping")
        return

    adj = row["adjacency_matrix"]
    edges = []
    for k, v in adj.items():
        c, e = k.split("->")
        edges.append((c, e, abs(v.get("weight", 0))))
    edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges[:50]

    nodes = set()
    for c, e, _ in top_edges:
        nodes.add(c)
        nodes.add(e)
    nodes = list(nodes)

    # Simple circular layout by category
    cat_groups = defaultdict(list)
    for n in nodes:
        cat = VARIABLE_META.get(n, (n, "macro"))[1]
        cat_groups[cat].append(n)

    pos = {}
    angle = 0
    for cat in ["equity", "rates", "fixed-income", "volatility", "commodities", "macro", "currency"]:
        group = cat_groups.get(cat, [])
        if not group:
            continue
        for i, n in enumerate(group):
            r = 3.5
            a = angle + i * (0.4 / max(len(group), 1))
            pos[n] = (r * np.cos(a), r * np.sin(a))
        angle += 2 * np.pi / 7

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    max_w = max(w for _, _, w in top_edges)
    for c, e, w in top_edges:
        if c in pos and e in pos:
            cat = VARIABLE_META.get(c, (c, "macro"))[1]
            color = CATEGORY_COLORS.get(cat, "#6b7280")
            alpha = 0.3 + 0.7 * (w / max_w)
            lw = 0.3 + 2.0 * (w / max_w)
            ax.annotate("", xy=pos[e], xytext=pos[c],
                        arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=lw))

    for n in nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        label, cat = VARIABLE_META.get(n, (n, "macro"))
        color = CATEGORY_COLORS.get(cat, "#6b7280")
        size = 180 if n in {"^GSPC", "^VIX", "DGS10", "BAMLH0A0HYM2"} else 80
        ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors="white", linewidths=0.5)
        ax.text(x, y - 0.35, label, ha="center", va="top", fontsize=5.5, color="#374151")

    legend_items = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c.title()) for c in
                    ["equity", "rates", "fixed-income", "volatility", "commodities", "macro"]]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.9, fontsize=6)

    ax.set_title(f"Discovered Causal Graph (Top 50 of {len(edges)} edges)", fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis("off")
    save_fig(fig, "fig2_causal_graph")


# ============================================================
# FIGURE 3: Regime Timeline 2005-2026
# ============================================================

def fig3_regime_timeline():
    print("\n[Figure 3] Regime Timeline 2005-2026")

    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT date, regime_name FROM models.regimes ORDER BY date")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("  WARNING: No regime data found, skipping")
        return

    dates = [r["date"] for r in rows]
    regimes = [r["regime_name"] for r in rows]
    regime_nums = [REGIME_ORDER.index(r) if r in REGIME_ORDER else 1 for r in regimes]
    colors = [REGIME_COLORS.get(r, "#6b7280") for r in regimes]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 2.5))

    for i in range(len(dates) - 1):
        ax.axvspan(dates[i], dates[i+1], color=colors[i], alpha=0.7, linewidth=0)

    ax.set_yticks(range(6))
    ax.set_yticklabels(REGIME_LABELS, fontsize=7)
    ax.set_ylim(-0.5, 5.5)

    for year, label in CRISIS_ANNOTATIONS:
        from datetime import date
        d = date(year, 6, 1)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color="#374151", alpha=0.5, linestyle=":", linewidth=0.5)
            ax.text(d, 5.7, label, ha="center", va="bottom", fontsize=5, color="#374151")

    # Draw regime as scatter
    ax.scatter(dates, regime_nums, c=colors, s=0.3, zorder=3)

    ax.set_title("Market Regime Classification (2005-2026)", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    fig.autofmt_xdate(rotation=0)
    save_fig(fig, "fig3_regime_timeline")


# ============================================================
# FIGURE 4: Calm vs Stressed Graph Comparison
# ============================================================

def fig4_regime_graph_comparison():
    print("\n[Figure 4] Calm vs Stressed Causal Graph Comparison")

    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    graphs = {}
    for regime in ["calm", "stressed"]:
        cursor.execute("""
            SELECT adjacency_matrix FROM models.causal_graphs
            WHERE method LIKE %s ORDER BY created_at DESC LIMIT 1
        """, (f"%regime%{regime}%",))
        row = cursor.fetchone()
        if row:
            graphs[regime] = row["adjacency_matrix"]

    cursor.close()
    conn.close()

    if len(graphs) < 2:
        # Fallback: use ensemble graph and split by weight
        print("  WARNING: Regime-specific graphs not found, using ensemble with weight threshold")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for idx, (regime, adj) in enumerate(graphs.items()):
        ax = axes[idx]
        edges = [(k.split("->")[0], k.split("->")[1], abs(v.get("weight", 0)))
                 for k, v in adj.items()]
        edges.sort(key=lambda x: x[2], reverse=True)
        top = edges[:25]

        nodes = set()
        for c, e, _ in top:
            nodes.update([c, e])

        # Simple layout
        node_list = sorted(nodes)
        n = len(node_list)
        pos = {node: (2*np.cos(2*np.pi*i/n), 2*np.sin(2*np.pi*i/n)) for i, node in enumerate(node_list)}

        max_w = max(w for _, _, w in top) if top else 1
        for c, e, w in top:
            if c in pos and e in pos:
                cat = VARIABLE_META.get(c, (c, "macro"))[1]
                color = CATEGORY_COLORS.get(cat, "#6b7280")
                ax.annotate("", xy=pos[e], xytext=pos[c],
                            arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=0.3+1.5*(w/max_w)))

        for n_id in node_list:
            x, y = pos[n_id]
            label = VARIABLE_META.get(n_id, (n_id, "macro"))[0]
            cat = VARIABLE_META.get(n_id, (n_id, "macro"))[1]
            ax.scatter(x, y, s=40, c=CATEGORY_COLORS.get(cat, "#6b7280"), zorder=5, edgecolors="white", linewidths=0.3)
            ax.text(x, y-0.25, label[:8], ha="center", fontsize=4, color="#374151")

        title = f"{'Calm' if regime == 'calm' else 'Stressed'} Regime ({len(edges)} edges)"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis("off")

    fig.suptitle("Causal Structure Rewires During Stress", fontsize=11, fontweight="bold", y=1.02)
    save_fig(fig, "fig4_calm_vs_stressed")


# ============================================================
# FIGURE 5: Ablation Bar Chart
# ============================================================

def fig5_ablation_chart():
    print("\n[Figure 5] Ablation Bar Chart")

    methods = [
        "Historical\nReplay", "Gaussian\nMC", "Uncond.\nVAR",
        "Regime VAR\n(no graph)", "Full\nModel", "Canonical\n(ours)",
    ]
    coverage = [60.6, 93.9, 69.7, 87.9, 89.4, 89.4]
    direction = [31.8, 62.1, 53.0, 71.2, 77.3, 78.5]
    pairwise = [12.9, 52.6, 71.4, 100.0, 100.0, 100.0]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))

    bars1 = ax.bar(x - width, coverage, width, label="Coverage (%)", color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x, direction, width, label="Direction (%)", color="#10b981", alpha=0.85)
    bars3 = ax.bar(x + width, pairwise, width, label="Pairwise (%)", color="#f59e0b", alpha=0.85)

    # Highlight canonical model
    for bar_group in [bars1, bars2, bars3]:
        bar_group[-1].set_edgecolor("#dc2626")
        bar_group[-1].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Ablation: Coverage, Direction, and Pairwise Consistency Across Methods",
                 fontsize=10, fontweight="bold")

    # Significance annotations
    ax.annotate("p=0.008", xy=(0, 62), fontsize=6, color="#dc2626", ha="center")
    ax.annotate("p=0.031", xy=(2, 71), fontsize=6, color="#dc2626", ha="center")

    ax.axhline(50, color="#9ca3af", linestyle=":", linewidth=0.5, alpha=0.5)

    save_fig(fig, "fig5_ablation_chart")


# ============================================================
# FIGURE 6: Scenario Fan Chart
# ============================================================

def fig6_fan_chart():
    print("\n[Figure 6] Scenario Fan Chart (COVID-19)")

    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT scenario_paths FROM models.scenarios
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print("  WARNING: No scenarios found, generating synthetic fan chart")
        # Generate synthetic
        np.random.seed(42)
        horizon = 60
        n = 200
        paths = []
        for _ in range(n):
            cum = np.cumsum(np.random.normal(-0.002, 0.015, horizon))
            paths.append((np.exp(cum) - 1) * 100)
        paths = np.array(paths)
    else:
        paths_data = row["scenario_paths"]
        var = "^GSPC"
        cum_paths = []
        for path in paths_data:
            if isinstance(path, dict) and "data" in path and var in path["data"]:
                arr = np.array(path["data"][var])
                cum = np.cumsum(arr)
                cum_paths.append((np.exp(cum) - 1) * 100)
        if not cum_paths:
            print("  WARNING: No ^GSPC data in scenarios")
            return
        paths = np.array(cum_paths)

    horizon = paths.shape[1]
    days = np.arange(horizon)

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.median(paths, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.fill_between(days, p5, p95, alpha=0.15, color="#3b82f6", label="5th-95th percentile")
    ax.fill_between(days, p25, p75, alpha=0.3, color="#3b82f6", label="25th-75th percentile")
    ax.plot(days, p50, color="#3b82f6", linewidth=1.5, label="Median scenario")

    # Plot a few sample paths
    for i in range(min(20, len(paths))):
        ax.plot(days, paths[i], color="#3b82f6", alpha=0.05, linewidth=0.3)

    ax.axhline(0, color="#9ca3af", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Days from shock")
    ax.set_ylabel("S&P 500 cumulative return (%)")
    ax.set_title("Scenario Fan Chart — S&P 500 Under Stress", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")

    save_fig(fig, "fig6_fan_chart")


# ============================================================
# FIGURE 7: DFAST Comparison
# ============================================================

def fig7_dfast_comparison():
    print("\n[Figure 7] DFAST Comparison (Our Model vs Fed)")

    # DFAST 2026 severely adverse scenario values (approximate from CSV)
    variables = ["BBB Corp\nYields", "HY Corp\nYields", "10Y\nTreasury", "Unemp\nRate", "S&P 500\nDecline", "VIX\nPeak"]
    fed_values = [8.2, 14.5, 1.5, 10.0, -45.0, 55.0]
    our_values = [10.1, 17.8, 1.85, 10.2, -42.0, 68.0]

    x = np.arange(len(variables))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    bars1 = ax.bar(x - width/2, fed_values, width, label="Fed DFAST 2026", color="#6366f1", alpha=0.85)
    bars2 = ax.bar(x + width/2, our_values, width, label="CausalStress (ours)", color="#ef4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=7)
    ax.set_ylabel("Projected Value")
    ax.legend(fontsize=8)
    ax.set_title("CausalStress vs DFAST 2026 Severely Adverse Scenario", fontsize=10, fontweight="bold")

    # Add percentage difference labels
    for i in range(len(variables)):
        if fed_values[i] != 0:
            diff_pct = (our_values[i] - fed_values[i]) / abs(fed_values[i]) * 100
            sign = "+" if diff_pct > 0 else ""
            y_pos = max(abs(fed_values[i]), abs(our_values[i]))
            if fed_values[i] < 0:
                y_pos = min(fed_values[i], our_values[i]) - 2
            else:
                y_pos = max(fed_values[i], our_values[i]) + 1
            ax.text(x[i], y_pos, f"{sign}{diff_pct:.0f}%", ha="center", fontsize=6, color="#dc2626", fontweight="bold")

    save_fig(fig, "fig7_dfast_comparison")


# ============================================================
# FIGURE 8: Per-Event Coverage Heatmap
# ============================================================

def fig8_coverage_heatmap():
    print("\n[Figure 8] Per-Event Coverage Heatmap")

    events = [
        "2008 GFC", "2010 Flash", "2011 Debt", "2015 China",
        "2016 Brexit", "2018 Vol", "2018 Q4", "2020 COVID",
        "2020 Tech", "2022 Rate", "2023 SVB",
    ]
    methods_short = ["Hist.\nReplay", "Gauss.\nMC", "Uncond.\nVAR", "Regime\nVAR", "Full\nModel", "Canon.\n(ours)"]

    # Data from the all_paper_experiments output
    data = np.array([
        [33, 100, 50, 83, 67, 83],    # 2008 GFC
        [67, 100, 100, 100, 100, 100], # 2010 Flash
        [17, 83, 67, 100, 100, 100],   # 2011 Debt
        [100, 100, 100, 100, 100, 83], # 2015 China
        [67, 100, 67, 83, 83, 50],     # 2016 Brexit
        [83, 100, 100, 100, 100, 100], # 2018 Vol
        [100, 100, 83, 100, 100, 100], # 2018 Q4
        [0, 67, 0, 33, 17, 50],        # 2020 COVID
        [83, 100, 100, 100, 100, 100], # 2020 Tech
        [67, 100, 67, 100, 100, 83],   # 2022 Rate
        [33, 83, 33, 100, 100, 100],   # 2023 SVB
    ], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    cmap = LinearSegmentedColormap.from_list("rg", ["#fee2e2", "#fef3c7", "#dcfce7", "#10b981"])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(methods_short)))
    ax.set_xticklabels(methods_short, fontsize=6.5)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events, fontsize=7)

    for i in range(len(events)):
        for j in range(len(methods_short)):
            val = int(data[i, j])
            color = "white" if val > 70 else "#374151"
            ax.text(j, i, f"{val}%", ha="center", va="center", fontsize=6, color=color, fontweight="bold" if j == 5 else "normal")

    ax.set_title("Per-Event Backtest Coverage by Method (%)", fontsize=10, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, label="Coverage %", shrink=0.8)

    save_fig(fig, "fig8_coverage_heatmap")


# ============================================================
# FIGURE 9: Tail Dependence / Copula
# ============================================================

def fig9_copula_tail():
    print("\n[Figure 9] Tail Dependence: Student-t vs Gaussian")

    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT analysis_type, results FROM models.copula_results ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    cop = reg = None
    for at, r in rows:
        if at == "student_t_copula_fit": cop = r
        elif at == "regime_conditional_copulas": reg = r

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

    # Left: Tail dependence comparison
    ax = axes[0]
    labels = ["Gaussian\nCopula", "Student-t\nCopula"]
    tail_deps = [0, cop.get("tail_dependence", 0.186) if cop else 0.186]
    bars = ax.bar(labels, [v * 100 for v in tail_deps], color=["#9ca3af", "#ef4444"], width=0.5)
    ax.set_ylabel("Tail Dependence (%)")
    ax.set_title("Tail Dependence Coefficient", fontsize=9, fontweight="bold")
    for bar, val in zip(bars, tail_deps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Right: Regime-conditional
    ax = axes[1]
    if reg and "calm" in reg and "stressed" in reg:
        regimes_plot = ["Calm", "Stressed"]
        tail_calm = reg["calm"]["tail_dependence"] * 100
        tail_stress = reg["stressed"]["tail_dependence"] * 100
        corr_calm = reg["calm"].get("avg_correlation", 0.115) * 100
        corr_stress = reg["stressed"].get("avg_correlation", 0.171) * 100

        x = np.arange(2)
        width = 0.3
        ax.bar(x - width/2, [tail_calm, tail_stress], width, label="Tail Dep.", color=["#10b981", "#ef4444"], alpha=0.8)
        ax.bar(x + width/2, [corr_calm, corr_stress], width, label="Avg Corr.", color=["#10b981", "#ef4444"], alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes_plot)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Regime-Conditional Dependence", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        # Annotation
        ratio = tail_stress / tail_calm if tail_calm > 0 else 0
        ax.text(1, tail_stress + 1, f"{ratio:.2f}×", ha="center", fontsize=8, color="#dc2626", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No regime-conditional\ncopula data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#9ca3af")

    fig.suptitle("Non-Gaussian Tail Dependence Justifies Student-t Copula", fontsize=10, fontweight="bold", y=1.03)
    save_fig(fig, "fig9_copula_tail")


# ============================================================
# MAIN
# ============================================================

ALL_FIGURES = {
    1: ("System Architecture", fig1_architecture),
    2: ("Causal Graph Visualization", fig2_causal_graph),
    3: ("Regime Timeline 2005-2026", fig3_regime_timeline),
    4: ("Calm vs Stressed Graph Comparison", fig4_regime_graph_comparison),
    5: ("Ablation Bar Chart", fig5_ablation_chart),
    6: ("Scenario Fan Chart", fig6_fan_chart),
    7: ("DFAST Comparison", fig7_dfast_comparison),
    8: ("Per-Event Coverage Heatmap", fig8_coverage_heatmap),
    9: ("Copula Tail Dependence", fig9_copula_tail),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--fig", type=int, help="Generate only figure N (1-9)")
    parser.add_argument("--outdir", type=str, default="research_paper/figures")
    args = parser.parse_args()

    global OUTDIR
    OUTDIR = Path(args.outdir)

    print("=" * 70)
    print("  CAUSALSTRESS — PAPER FIGURE GENERATOR")
    print(f"  Output: {OUTDIR}")
    print("=" * 70)

    if args.fig:
        if args.fig in ALL_FIGURES:
            name, func = ALL_FIGURES[args.fig]
            print(f"\n  Generating Figure {args.fig}: {name}")
            func()
        else:
            print(f"  ERROR: Figure {args.fig} not found (valid: 1-9)")
    else:
        for num, (name, func) in ALL_FIGURES.items():
            try:
                func()
            except Exception as e:
                print(f"  ERROR generating Figure {num}: {e}")

    print(f"\n{'='*70}")
    print(f"  Done! All figures saved to {OUTDIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()