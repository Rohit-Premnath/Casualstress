"""
adversarial_serve.py
====================
Serving wrapper for BanditRewardNet adversarial inference.

Pre-load once at FastAPI startup (via lifespan), reuse across requests.
Each profile gets its own AdversarialEngine: model + catalog + env.

Model resolution order per profile:
    1. runs/bandit_v2_{profile}/bandit.pt  (2-step, preferred)
    2. runs/bandit_v1_{profile}/bandit.pt  (1-step, fallback)

Usage (from FastAPI lifespan):
    from ml_pipeline.generative_engine_rl.adversarial_serve import load_all_engines
    load_all_engines()   # call in thread pool executor — blocking

Usage (per request):
    engine = get_engine("balanced")
    result = engine.find_worst_case(n_seeds=4, ucb_beta=0.5)
    scenarios = engine.find_ranked_scenarios(n_seeds=20, top_k=5)
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent   # ml_pipeline/
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

logger = logging.getLogger("causalstress.adversarial")

RUNS_DIR = ML_ROOT / "runs"
PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]

_engines: Dict[str, "AdversarialEngine"] = {}
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdversarialResult:
    profile: str
    model_version: str          # "v1" or "v2"
    worst_sequence: List[Dict]  # [{"target_var": str, "family_name": str, "magnitude": float}]
    portfolio_loss: float
    causal_fidelity: float
    diversity: float
    dfast_breach: float
    sampled_state: Optional[Dict]   # {"date": "2008-10-10", "regime_name": "crisis"}
    seeds_tried: int
    seed_used: int
    vs_beam_pct: Optional[float] = None   # from stored benchmark JSON


@dataclass
class RankedScenario:
    rank: int
    sequence: List[Dict]        # [{"target_var": str, "family_name": str, "magnitude": float}]
    portfolio_loss: float
    causal_fidelity: float
    diversity: float
    dfast_breach: float
    reward: float
    sampled_state: Optional[Dict]
    seed_used: int
    causal_pathway: str         # human-readable: "Technology sector shock → Nasdaq-100 contagion"


# ---------------------------------------------------------------------------
# Asset classification for portfolio fingerprinting
# ---------------------------------------------------------------------------

_TECH_ASSETS = frozenset({
    "XLK", "QQQ", "ARKK", "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG",
    "META", "AMZN", "AMD", "TSM", "AVGO", "^NDX", "SOXX", "SMH",
})
_DURATION_ASSETS = frozenset({
    "TLT", "IEF", "AGG", "BND", "EDV", "VGLT", "TLH", "GOVT", "ZROZ",
    "LTPZ", "SHY", "IEI", "VGIT", "TIP", "SCHZ",
})
_CREDIT_ASSETS = frozenset({
    "HYG", "JNK", "LQD", "VCIT", "VCSH", "ANGL", "USHY", "FALN",
    "SJNK", "BKLN", "SHYG", "PHB",
})

_ASSET_LABELS: Dict[str, str] = {
    "TLT": "Long-dated Treasuries",
    "IEF": "Intermediate Treasuries",
    "AGG": "Aggregate Bonds",
    "BND": "Total Bond Market",
    "XLK": "Technology sector",
    "QQQ": "Nasdaq-100",
    "^NDX": "Nasdaq-100",
    "ARKK": "Innovation/Tech ETF",
    "HYG": "High-yield credit",
    "JNK": "High-yield bonds",
    "LQD": "Investment-grade credit",
    "SPY": "S&P 500 equities",
    "IWM": "Small-cap equities",
    "VTI": "Total US market",
    "GLD": "Gold",
    "XLE": "Energy sector",
    "XLF": "Financials sector",
    "XLV": "Healthcare sector",
    "XLY": "Consumer discretionary",
    "XLU": "Utilities sector",
    "XLI": "Industrials sector",
    "XLP": "Consumer staples",
    "XLB": "Materials sector",
    "EEM": "Emerging markets",
    "VNQ": "Real estate",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment rate",
    "VIX": "Volatility index",
    "NVDA": "Nvidia (semiconductors)",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet",
}


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class AdversarialEngine:
    """
    Pre-loaded model + environment for one portfolio profile.
    Thread-safe: a lock serialises env.reset/step so concurrent requests
    don't corrupt internal env state.
    """

    def __init__(self, profile: str, model_path: Path, model_version: str):
        import torch
        from generative_engine_rl.neural_bandit import BanditRewardNet, build_catalog_tensor
        from generative_engine_rl.env_factory import make_env

        self.profile = profile
        self.model_version = model_version
        self._lock = threading.Lock()

        logger.info("Loading bandit %s  profile=%s  path=%s", model_version, profile, model_path)
        self.net = BanditRewardNet.load(str(model_path))
        self.net.eval()

        n_steps = 2 if model_version == "v2" else 1
        self.env = make_env(
            mode="real",
            portfolio_profile=profile,
            n_magnitude_bins=21,
            reward_mode="portfolio_adversarial",
            actions_per_episode=n_steps,
        )
        self.catalog = build_catalog_tensor(self.env)
        self.vs_beam_pct = _load_vs_beam_pct(profile, model_version)
        logger.info(
            "Engine ready  profile=%s  version=%s  n_actions=%d  vs_beam=%.1f%%",
            profile, model_version, len(self.catalog),
            self.vs_beam_pct if self.vs_beam_pct is not None else float("nan"),
        )

    def find_worst_case(
        self,
        n_seeds: int = 4,
        seed_start: int = 30000,
        ucb_beta: float = 0.5,
    ) -> AdversarialResult:
        """
        Run bandit inference over n_seeds historical starting states.
        Returns the state/action pair that maximises portfolio_loss.
        seed_start=30000 is clear of training (1000-2049),
        warm-start (10000-10011), and held-out eval (20000-20015) seeds.
        """
        from generative_engine_rl.neural_bandit import bandit_sequence

        best = None
        with self._lock:
            for seed in range(seed_start, seed_start + n_seeds):
                result = bandit_sequence(
                    self.net, self.catalog, self.env,
                    seed, ucb_beta=ucb_beta,
                )
                if best is None or result["portfolio_loss"] > best["portfolio_loss"]:
                    best = result

        return AdversarialResult(
            profile=self.profile,
            model_version=self.model_version,
            worst_sequence=best["sequence"],
            portfolio_loss=float(best["portfolio_loss"]),
            causal_fidelity=float(best["causal_fidelity"]),
            diversity=float(best["diversity"]),
            dfast_breach=float(best["dfast_breach"]),
            sampled_state=best.get("sampled_state"),
            seeds_tried=n_seeds,
            seed_used=int(best["seed"]),
            vs_beam_pct=self.vs_beam_pct,
        )

    def find_ranked_scenarios(
        self,
        n_seeds: int = 20,
        seed_start: int = 40000,
        ucb_beta: float = 0.5,
        top_k: int = 5,
    ) -> List[RankedScenario]:
        """
        Run bandit over n_seeds starting states, deduplicate by shock-sequence
        type, rank by portfolio_loss, return top_k distinct adversarial scenarios.

        Deduplication key: tuple of target_var names in the sequence.
        Within each key, the highest portfolio_loss result is kept so each
        returned scenario represents a distinct causal pathway, not just a
        magnitude variation of the same shock.

        seed_start=40000 is clear of all other seed ranges:
          training 1000-2049, warm-start 10000-10011,
          held-out eval 20000-20015, worst-case inference 30000+.
        """
        from generative_engine_rl.neural_bandit import bandit_sequence

        all_results: List[Dict] = []
        with self._lock:
            for seed in range(seed_start, seed_start + n_seeds):
                result = bandit_sequence(
                    self.net, self.catalog, self.env,
                    seed, ucb_beta=ucb_beta,
                )
                all_results.append(result)

        # Deduplicate: keep highest-loss result per distinct (target_var, ...) key
        seen: Dict[tuple, Dict] = {}
        for r in all_results:
            key = tuple(step["target_var"] for step in r["sequence"])
            if key not in seen or r["portfolio_loss"] > seen[key]["portfolio_loss"]:
                seen[key] = r

        # Rank by portfolio_loss descending, take top_k
        ranked = sorted(seen.values(), key=lambda x: x["portfolio_loss"], reverse=True)[:top_k]

        return [
            RankedScenario(
                rank=i + 1,
                sequence=r["sequence"],
                portfolio_loss=float(r["portfolio_loss"]),
                causal_fidelity=float(r["causal_fidelity"]),
                diversity=float(r["diversity"]),
                dfast_breach=float(r["dfast_breach"]),
                reward=float(r["reward"]),
                sampled_state=r.get("sampled_state"),
                seed_used=int(r["seed"]),
                causal_pathway=_describe_pathway(r["sequence"]),
            )
            for i, r in enumerate(ranked)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_path(profile: str) -> Tuple[Path, str]:
    """Return (Path, version_str) for the best available bandit model."""
    v2 = RUNS_DIR / f"bandit_v2_{profile}" / "bandit.pt"
    v1 = RUNS_DIR / f"bandit_v1_{profile}" / "bandit.pt"
    if v2.exists():
        return v2, "v2"
    if v1.exists():
        return v1, "v1"
    raise FileNotFoundError(
        f"No bandit model found for profile '{profile}'. "
        f"Expected runs/bandit_v1_{profile}/bandit.pt or runs/bandit_v2_{profile}/bandit.pt."
    )


def _load_vs_beam_pct(profile: str, version: str) -> Optional[float]:
    """Read the best greedy/UCB vs-beam % from stored benchmark JSON."""
    import json
    for fname in ("heldout_results_1step.json", "heldout_results.json"):
        path = RUNS_DIR / f"bandit_{version}_{profile}" / fname
        if path.exists():
            try:
                d = json.loads(path.read_text())
                candidates = [
                    d.get("bandit_greedy_vs_beam_pct"),
                    d.get("bandit_ucb_vs_beam_pct"),
                ]
                values = [v for v in candidates if v is not None]
                return round(max(values), 1) if values else None
            except Exception:
                pass
    return None


def _describe_pathway(sequence: List[Dict]) -> str:
    """Convert a shock sequence to a human-readable causal pathway description."""
    if not sequence:
        return "Unknown pathway"
    parts = [_ASSET_LABELS.get(s["target_var"], s["target_var"]) for s in sequence]
    if len(parts) == 1:
        return f"{parts[0]} shock"
    if len(parts) == 2:
        return f"{parts[0]} shock → {parts[1]} contagion"
    return " → ".join(parts)


_DURATION_CATS = frozenset(("bond", "bonds", "fixed income", "fixed-income", "treasury", "rates"))
_DURATION_KEYWORDS = ("treasury", "bond", "fixed income", "gilt", "sovereign")
# Credit checked before duration: "high yield bonds" has "bond" but should be credit
_CREDIT_KEYWORDS = ("high yield", "junk", "clo", "leveraged loan")
_TECH_KEYWORDS = ("tech", "semiconductor", "software", "hardware", "innovation", "nasdaq")


def compute_exposure_fingerprint(holdings: List[Dict]) -> Dict[str, float]:
    """
    Compute portfolio exposure features from a list of holding dicts.

    Expected keys per holding: "asset" (str), "weight" (float), "category" (str).
    When asset is a human-readable name (e.g. "20Y Treasury Bonds") rather than
    a ticker, keyword matching on the lowercased asset string is used as fallback.
    Credit is matched before duration so "High Yield Bonds" routes correctly
    even when category is the generic "fixed-income".

    Returns {"tech": float, "duration": float, "credit": float, "equity": float}
    where all values are percent weights. equity is the residual (100 - others).
    """
    tech = duration = credit = 0.0

    for h in holdings:
        asset_raw = str(h.get("asset", "")).strip()
        asset_upper = asset_raw.upper()
        asset_lower = asset_raw.lower()
        weight = float(h.get("weight", 0) or 0)
        category = str(h.get("category", "")).lower().strip()

        if (
            asset_upper in _TECH_ASSETS
            or category in ("tech", "technology")
            or any(k in asset_lower for k in _TECH_KEYWORDS)
        ):
            tech += weight
        elif (
            asset_upper in _CREDIT_ASSETS
            or category in ("credit", "high yield", "hy")
            or any(k in asset_lower for k in _CREDIT_KEYWORDS)
        ):
            # Credit before duration: catches "High Yield Bonds" (category='fixed-income')
            credit += weight
        elif (
            asset_upper in _DURATION_ASSETS
            or category in _DURATION_CATS
            or any(k in asset_lower for k in _DURATION_KEYWORDS)
        ):
            duration += weight
        # else: general equity, real assets, commodities — absorbed into residual

    equity = max(0.0, 100.0 - tech - duration - credit)
    return {
        "tech": round(tech, 1),
        "duration": round(duration, 1),
        "credit": round(credit, 1),
        "equity": round(equity, 1),
    }


def classify_profile(holdings: List[Dict]) -> Tuple[str, float]:
    """
    Map portfolio holdings to the nearest vulnerability profile.

    Returns (profile_name, confidence_pct).
    confidence_pct is the dominant exposure weight (0-100).
    Falls back to "balanced" when no single exposure exceeds 25%.
    """
    fp = compute_exposure_fingerprint(holdings)

    candidates = {
        "tech_heavy": fp["tech"],
        "bond_heavy": fp["duration"],
        "credit_heavy": fp["credit"],
    }

    best_profile = max(candidates, key=lambda k: candidates[k])
    best_score = candidates[best_profile]

    if best_score < 25.0:
        return "balanced", round(fp["equity"], 1)

    return best_profile, round(best_score, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_engines(profiles: List[str] = PROFILES) -> None:
    """Load models for all available profiles. Called once at startup."""
    with _load_lock:
        for profile in profiles:
            if profile in _engines:
                continue
            try:
                path, version = _resolve_model_path(profile)
                _engines[profile] = AdversarialEngine(profile, path, version)
            except FileNotFoundError as e:
                logger.warning("Skipping profile %s: %s", profile, e)
            except Exception as e:
                logger.error("Failed to load engine for %s: %s", profile, e, exc_info=True)


def get_engine(profile: str) -> AdversarialEngine:
    engine = _engines.get(profile)
    if engine is None:
        raise KeyError(
            f"No engine loaded for profile '{profile}'. "
            f"Available: {list(_engines.keys())}. "
            f"Check that bandit.pt exists and training completed."
        )
    return engine


def loaded_profiles() -> List[str]:
    return list(_engines.keys())
