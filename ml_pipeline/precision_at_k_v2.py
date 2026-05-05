"""
Precision-at-k Analysis v2: Ensemble Ranking
==============================================
Tries multiple ranking strategies on the ensemble causal graph:
  1. consensus_product:  dyno_conf * pcmci_score (rewards agreement)
  2. consensus_min:      min(dyno_conf, pcmci_score) (strict AND gate)
  3. consensus_tier:     both-agree edges first, then single-method
  4. dynotears_only:     DYNOTEARS confidence alone (v1 baseline)
  5. pcmci_only:         PCMCI score alone
  6. weight:             |edge weight| (magnitude baseline)

For each strategy, computes precision@{10,25,50,100,200,500} and
reports which strategy produces the best paper-ready numbers.

The 208 "consensus" edges (confirmed by both DYNOTEARS and PCMCI) are
the gold standard. A good ranker should surface these FIRST.
"""

import os
import sys
import json
import uuid
import warnings
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

KNOWN_EDGES = [
    ("PAYEMS", "UNRATE"), ("DGS10", "T10Y2Y"), ("DGS10", "DGS2"),
    ("^GSPC", "XLF"), ("^GSPC", "XLK"), ("^GSPC", "XLE"),
    ("^GSPC", "XLV"), ("^GSPC", "XLY"), ("^GSPC", "XLU"),
    ("^GSPC", "^NDX"), ("^GSPC", "^RUT"), ("^GSPC", "EEM"),
    ("^VIX", "^VVIX"), ("DGS10", "TLT"), ("BAMLH0A0HYM2", "HYG"),
    ("CL=F", "XLE"), ("FEDFUNDS", "DGS2"), ("CPIAUCSL", "PCEPILFE"),
    ("BAMLH0A0HYM2", "BAMLH0A1HYBB"), ("BAMLH0A0HYM2", "BAMLH0A3HYC"),
    ("BAMLC0A0CM", "BAMLC0A4CBBB"), ("BAMLC0A0CM", "BAMLC0A3CA"),
    ("DRTSCIS", "DRTSCILM"), ("DX-Y.NYB", "EURUSD=X"), ("INDPRO", "PAYEMS"),
]
KNOWN_EDGE_SET = set(KNOWN_EDGES)

K_VALUES = [10, 25, 50, 100, 200, 500]


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def load_ensemble_graph():
    """Load the ensemble graph with its full adjacency matrix."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT adjacency_matrix FROM models.causal_graphs
        WHERE method LIKE '%%ensemble%%' OR method LIKE '%%dynotears%%'
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else None


def store_results(payload):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models.precision_at_k_analysis (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            analysis_name VARCHAR(200),
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    clean = json.loads(json.dumps(payload, default=lambda x:
        None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
        else bool(x) if isinstance(x, (np.bool_,))
        else float(x) if isinstance(x, (np.floating,))
        else int(x) if isinstance(x, (np.integer,))
        else x))
    cursor.execute(
        "INSERT INTO models.precision_at_k_analysis (id, analysis_name, results) VALUES (%s, %s, %s)",
        (str(uuid.uuid4()), "Ensemble precision-at-k v2", Json(clean)),
    )
    conn.commit()
    cursor.close()
    conn.close()


# ============================================================
# EDGE EXTRACTION
# ============================================================

def extract_edge_fields(edge_data: dict) -> dict:
    """
    Pull out every score-like field we might use for ranking.
    Returns a dict with best-effort normalized fields.
    """
    fields = {
        "dyno_confidence": None,
        "pcmci_score": None,
        "weight": None,
        "in_dyno": False,
        "in_pcmci": False,
    }

    if not isinstance(edge_data, dict):
        return fields

    # Try every possible field name the DB might use
    for key in ["dyno_confidence", "dynotears_confidence", "dyno_conf", "dynotears"]:
        if key in edge_data and edge_data[key] is not None:
            try:
                fields["dyno_confidence"] = float(edge_data[key])
                fields["in_dyno"] = fields["dyno_confidence"] > 0.0
                break
            except (TypeError, ValueError):
                continue

    for key in ["pcmci_score", "pcmci", "pcmci_confidence", "pcmci_strength"]:
        if key in edge_data and edge_data[key] is not None:
            try:
                fields["pcmci_score"] = float(edge_data[key])
                fields["in_pcmci"] = fields["pcmci_score"] > 0.0
                break
            except (TypeError, ValueError):
                continue

    for key in ["weight", "edge_weight", "coefficient"]:
        if key in edge_data and edge_data[key] is not None:
            try:
                fields["weight"] = abs(float(edge_data[key]))
                break
            except (TypeError, ValueError):
                continue

    # Fallback: also check "methods" or "method_list" field for membership
    for key in ["methods", "method_list", "algorithms"]:
        if key in edge_data:
            val = edge_data[key]
            if isinstance(val, (list, tuple)):
                txt = " ".join(str(m).lower() for m in val)
            elif isinstance(val, str):
                txt = val.lower()
            else:
                continue
            if "dyno" in txt or "dynotears" in txt:
                fields["in_dyno"] = True
            if "pcmci" in txt:
                fields["in_pcmci"] = True

    return fields


def parse_edges(adjacency_matrix: dict) -> list:
    """Parse all edges into unified records with all scoring fields."""
    edges = []
    for edge_key, edge_data in adjacency_matrix.items():
        if "->" not in edge_key:
            continue
        source, target = edge_key.split("->", 1)
        fields = extract_edge_fields(edge_data if isinstance(edge_data, dict) else {})
        edges.append({
            "source": source,
            "target": target,
            **fields,
        })
    return edges


# ============================================================
# RANKING STRATEGIES
# ============================================================

def rank_consensus_product(edges: list) -> list:
    """Rank by dyno * pcmci. Edges missing one score get 0."""
    scored = []
    for e in edges:
        d = e["dyno_confidence"] or 0.0
        p = e["pcmci_score"] or 0.0
        scored.append((e["source"], e["target"], d * p, e))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def rank_consensus_min(edges: list) -> list:
    """Rank by min(dyno, pcmci). Strict AND gate."""
    scored = []
    for e in edges:
        d = e["dyno_confidence"] or 0.0
        p = e["pcmci_score"] or 0.0
        scored.append((e["source"], e["target"], min(d, p), e))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def rank_consensus_tier(edges: list) -> list:
    """
    Consensus edges first (both algorithms), then DYNOTEARS-only, then PCMCI-only.
    Within each tier, rank by the primary score.
    """
    both, dyno_only, pcmci_only, neither = [], [], [], []
    for e in edges:
        d = e["dyno_confidence"] or 0.0
        p = e["pcmci_score"] or 0.0
        in_d = e["in_dyno"] or d > 0.0
        in_p = e["in_pcmci"] or p > 0.0
        if in_d and in_p:
            # Sort consensus by geometric mean (balances both)
            score = (d * p) ** 0.5 if (d > 0 and p > 0) else max(d, p)
            both.append((e["source"], e["target"], score, e))
        elif in_d:
            dyno_only.append((e["source"], e["target"], d, e))
        elif in_p:
            pcmci_only.append((e["source"], e["target"], p, e))
        else:
            neither.append((e["source"], e["target"], 0.0, e))

    # Sort each tier internally
    both.sort(key=lambda x: x[2], reverse=True)
    dyno_only.sort(key=lambda x: x[2], reverse=True)
    pcmci_only.sort(key=lambda x: x[2], reverse=True)

    return both + dyno_only + pcmci_only + neither


def rank_dynotears_only(edges: list) -> list:
    """Baseline: DYNOTEARS confidence alone (v1's ranking)."""
    scored = [
        (e["source"], e["target"], e["dyno_confidence"] or 0.0, e)
        for e in edges
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def rank_pcmci_only(edges: list) -> list:
    """Baseline: PCMCI score alone."""
    scored = [
        (e["source"], e["target"], e["pcmci_score"] or 0.0, e)
        for e in edges
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def rank_by_weight(edges: list) -> list:
    """Baseline: absolute edge weight (magnitude)."""
    scored = [
        (e["source"], e["target"], e["weight"] or 0.0, e)
        for e in edges
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


RANKING_STRATEGIES = [
    ("consensus_product",   rank_consensus_product),
    ("consensus_min",       rank_consensus_min),
    ("consensus_tier",      rank_consensus_tier),
    ("dynotears_only",      rank_dynotears_only),
    ("pcmci_only",          rank_pcmci_only),
    ("weight_magnitude",    rank_by_weight),
]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(ranked: list, k: int, known_set: set) -> dict:
    k = min(k, len(ranked))
    top = ranked[:k]
    top_edges = {(s, t) for s, t, _, _ in top}
    tp = len(top_edges & known_set)
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(known_set) if known_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "k": k, "tp": tp, "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
    }


def k_full_recovery(ranked: list, known_set: set) -> int | None:
    """Smallest k at which all known edges are in top-k."""
    found = 0
    for i, (s, t, _, _) in enumerate(ranked, start=1):
        if (s, t) in known_set:
            found += 1
            if found >= len(known_set):
                return i
    return None


def pr_auc(ranked: list, known_set: set) -> float:
    n_known = len(known_set)
    if n_known == 0:
        return 0.0
    tp = 0
    prev_recall = 0.0
    auc = 0.0
    for i, (s, t, _, _) in enumerate(ranked, start=1):
        if (s, t) in known_set:
            tp += 1
        p = tp / i
        r = tp / n_known
        auc += p * (r - prev_recall)
        prev_recall = r
        if r >= 1.0:
            break
    return round(float(auc), 4)


# ============================================================
# MAIN
# ============================================================

def run_analysis():
    print("=" * 90)
    print("  ENSEMBLE PRECISION-AT-k ANALYSIS (v2)")
    print(f"  Ground-truth edges: {len(KNOWN_EDGES)}")
    print("=" * 90)

    adj = load_ensemble_graph()
    if adj is None:
        print("  ERROR: no ensemble graph found in models.causal_graphs")
        sys.exit(1)

    edges = parse_edges(adj)
    print(f"  Total edges parsed: {len(edges)}")

    # Diagnostic: what fields do we actually have populated?
    n_with_dyno = sum(1 for e in edges if e["dyno_confidence"] is not None)
    n_with_pcmci = sum(1 for e in edges if e["pcmci_score"] is not None)
    n_with_weight = sum(1 for e in edges if e["weight"] is not None)
    n_in_both = sum(1 for e in edges if e["in_dyno"] and e["in_pcmci"])
    n_dyno_only = sum(1 for e in edges if e["in_dyno"] and not e["in_pcmci"])
    n_pcmci_only = sum(1 for e in edges if e["in_pcmci"] and not e["in_dyno"])

    print()
    print(f"  Edge membership diagnostic:")
    print(f"    Have dyno_confidence:   {n_with_dyno:>5}")
    print(f"    Have pcmci_score:       {n_with_pcmci:>5}")
    print(f"    Have weight:            {n_with_weight:>5}")
    print(f"    In BOTH algorithms:     {n_in_both:>5}  <-- consensus edges")
    print(f"    In DYNOTEARS only:      {n_dyno_only:>5}")
    print(f"    In PCMCI only:          {n_pcmci_only:>5}")

    if n_in_both == 0 and n_with_pcmci == 0:
        print()
        print("  WARNING: No PCMCI scores found in the graph. Falling back to")
        print("  DYNOTEARS-only comparison — results will match v1.")
        print("  Check that pcmci_score/pcmci field is populated in adjacency_matrix.")

    # ---------------------------------------------------------
    # Run every ranking strategy
    # ---------------------------------------------------------
    all_results = {}

    for strategy_name, strategy_fn in RANKING_STRATEGIES:
        ranked = strategy_fn(edges)

        strat_result = {
            "per_k": [],
            "k_full_recovery": k_full_recovery(ranked, KNOWN_EDGE_SET),
            "pr_auc": pr_auc(ranked, KNOWN_EDGE_SET),
            "top_10_edges": [
                {"source": s, "target": t, "score": round(score, 4),
                 "is_ground_truth": (s, t) in KNOWN_EDGE_SET}
                for s, t, score, _ in ranked[:10]
            ],
        }
        for k in K_VALUES:
            if k <= len(ranked):
                strat_result["per_k"].append(precision_at_k(ranked, k, KNOWN_EDGE_SET))
        all_results[strategy_name] = strat_result

    # ---------------------------------------------------------
    # COMPARISON TABLE
    # ---------------------------------------------------------
    print()
    print("=" * 90)
    print("  PRECISION@k COMPARISON ACROSS RANKING STRATEGIES")
    print("=" * 90)

    header = f"  {'Strategy':<22}" + "".join(f"{'p@'+str(k):>9}" for k in K_VALUES) + f"  {'k_full':>8} {'PR-AUC':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_p50 = (None, 0.0)
    best_p100 = (None, 0.0)

    for strategy_name, result in all_results.items():
        line = f"  {strategy_name:<22}"
        for k in K_VALUES:
            pk = next((r for r in result["per_k"] if r["k"] == k), None)
            if pk:
                line += f"{pk['precision']:>9.3f}"
                if k == 50 and pk["precision"] > best_p50[1]:
                    best_p50 = (strategy_name, pk["precision"])
                if k == 100 and pk["precision"] > best_p100[1]:
                    best_p100 = (strategy_name, pk["precision"])
            else:
                line += f"{'--':>9}"
        kfr = result["k_full_recovery"] or "-"
        line += f"  {str(kfr):>8} {result['pr_auc']:>8.4f}"
        print(line)

    # ---------------------------------------------------------
    # RECALL@k FOR BEST STRATEGY
    # ---------------------------------------------------------
    best_strategy = best_p50[0] or best_p100[0] or "consensus_product"
    print()
    print("=" * 90)
    print(f"  BEST STRATEGY: {best_strategy}")
    print("=" * 90)

    best_result = all_results[best_strategy]
    print(f"\n  Precision / Recall / F1 at each k:")
    print(f"  {'k':>6} {'TP':>5} {'Precision':>11} {'Recall':>9} {'F1':>8}")
    print(f"  {'-'*48}")
    for pk in best_result["per_k"]:
        print(f"  {pk['k']:>6} {pk['tp']:>5} {pk['precision']:>10.4f}  {pk['recall']:>8.4f} {pk['f1']:>7.4f}")

    print(f"\n  k for full recall (all 25 edges): {best_result['k_full_recovery']}")
    print(f"  PR-AUC: {best_result['pr_auc']:.4f}")

    # Top-10 peek
    print(f"\n  Top 10 edges in this strategy:")
    for i, edge in enumerate(best_result["top_10_edges"], start=1):
        marker = "  *** GROUND TRUTH" if edge["is_ground_truth"] else ""
        print(f"    {i:>2}. {edge['source']} -> {edge['target']}  (score={edge['score']:.4f}){marker}")

    # ---------------------------------------------------------
    # PAPER HEADLINE
    # ---------------------------------------------------------
    print()
    print("=" * 90)
    print("  PAPER-READY HEADLINE (using best strategy)")
    print("=" * 90)

    p25 = next((r for r in best_result["per_k"] if r["k"] == 25), None)
    p50 = next((r for r in best_result["per_k"] if r["k"] == 50), None)
    p100 = next((r for r in best_result["per_k"] if r["k"] == 100), None)

    if p25:
        lift_25 = p25["precision"] / (len(KNOWN_EDGES) / len(edges))
        print(f"\n  precision@25  = {p25['precision']:.2%}  "
              f"({p25['tp']} of 25 are ground truth, {lift_25:.1f}x random baseline)")
    if p50:
        lift_50 = p50["precision"] / (len(KNOWN_EDGES) / len(edges))
        print(f"  precision@50  = {p50['precision']:.2%}  "
              f"({p50['tp']} of 50 are ground truth, {lift_50:.1f}x random baseline)")
    if p100:
        lift_100 = p100["precision"] / (len(KNOWN_EDGES) / len(edges))
        print(f"  precision@100 = {p100['precision']:.2%}  "
              f"({p100['tp']} of 100 are ground truth, {lift_100:.1f}x random baseline)")

    # ---------------------------------------------------------
    # STORE & PASTE BLOCK
    # ---------------------------------------------------------
    payload = {
        "n_ground_truth": len(KNOWN_EDGES),
        "n_discovered_total": len(edges),
        "n_consensus_edges": n_in_both,
        "strategies": all_results,
        "best_strategy": best_strategy,
        "ground_truth_edges": [list(e) for e in KNOWN_EDGES],
    }

    try:
        store_results(payload)
        print(f"\n  Stored in models.precision_at_k_analysis")
    except Exception as e:
        print(f"\n  WARNING: DB store failed: {e}")

    out_path = Path(__file__).parent / "precision_at_k_v2.json"
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  JSON export: {out_path}")
    except Exception as e:
        print(f"  WARNING: JSON write failed: {e}")

    print()
    print("=" * 90)
    print(f"  PASTE INTO canonical_paper_numbers.py (Experiment 1 section):")
    print("=" * 90)
    print()
    print(f"CAUSAL_RANKING_STRATEGY = '{best_strategy}'")
    if p25:
        print(f"CAUSAL_PRECISION_AT_25  = {p25['precision']:.4f}   # {p25['tp']} of top 25 are ground truth")
    if p50:
        print(f"CAUSAL_PRECISION_AT_50  = {p50['precision']:.4f}   # {p50['tp']} of top 50 are ground truth")
    if p100:
        print(f"CAUSAL_PRECISION_AT_100 = {p100['precision']:.4f}   # {p100['tp']} of top 100 are ground truth")
    if best_result["k_full_recovery"]:
        print(f"CAUSAL_K_FULL_RECOVERY  = {best_result['k_full_recovery']}   # edges needed for 100% recall")
    print(f"CAUSAL_PR_AUC           = {best_result['pr_auc']}   # PR-AUC")
    print(f"CAUSAL_CONSENSUS_EDGES  = {n_in_both}   # edges found by BOTH algorithms")
    print(f"CAUSAL_ENSEMBLE_METHODS = ['DYNOTEARS', 'PCMCI']")
    print()
    print("=" * 90)

    return payload


if __name__ == "__main__":
    run_analysis()