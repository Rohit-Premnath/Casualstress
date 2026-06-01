"""
Adversarial Worst-Case Router
==============================
POST /api/v1/adversarial/worst-case         — single worst-case (original, unchanged)
POST /api/v1/adversarial/ranked-scenarios   — portfolio-specific ranked search (new)
GET  /api/v1/adversarial/status

The ranked-scenarios endpoint accepts raw portfolio holdings, infers the
nearest vulnerability profile via exposure fingerprinting, runs the bandit
across multiple market starting states, deduplicates by shock-pathway type,
and returns the top-K ranked adversarial scenarios with causal explanations.

Models are pre-loaded at startup (main.py lifespan). All blocking
env/model calls run in a thread-pool executor so the event loop is free.
"""

import asyncio
import logging
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("causalstress.adversarial")

router = APIRouter(prefix="/api/v1/adversarial", tags=["adversarial"])

PROFILES = ["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class ShockStep(BaseModel):
    target_var: str
    family_name: str
    magnitude: float


# ---------------------------------------------------------------------------
# Request / response models — original worst-case endpoint
# ---------------------------------------------------------------------------

class AdversarialRequest(BaseModel):
    portfolio_profile: Literal["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]
    n_seeds: int = Field(
        default=4, ge=1, le=16,
        description="Number of random market states to evaluate (more = better coverage, slower)",
    )
    ucb_beta: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="UCB exploration: 0=greedy, higher=more exploration via MC-Dropout uncertainty",
    )


class SampledState(BaseModel):
    date: Optional[str] = None
    regime_name: Optional[str] = None
    row_index: Optional[int] = None


class AdversarialResponse(BaseModel):
    profile: str
    model_version: str
    worst_sequence: List[ShockStep]
    portfolio_loss: float
    causal_fidelity: float
    diversity: float
    dfast_breach: float
    sampled_state: Optional[SampledState] = None
    seeds_tried: int
    seed_used: int
    vs_beam_pct: Optional[float] = None
    quality_note: str


# ---------------------------------------------------------------------------
# Request / response models — ranked-scenarios endpoint
# ---------------------------------------------------------------------------

class HoldingItem(BaseModel):
    asset: str
    weight: float
    amount: float = 0.0
    category: str = "equity"


class AdversarialRankedRequest(BaseModel):
    holdings: List[HoldingItem]
    portfolio_profile: Optional[Literal["balanced", "tech_heavy", "bond_heavy", "credit_heavy"]] = Field(
        default=None,
        description="Override auto-inferred profile. If omitted, profile is inferred from holdings.",
    )
    n_seeds: int = Field(
        default=20, ge=4, le=50,
        description="Market starting states to search (more = richer scenario diversity)",
    )
    top_k: int = Field(
        default=5, ge=1, le=10,
        description="Number of distinct ranked scenarios to return",
    )
    ucb_beta: float = Field(default=0.5, ge=0.0, le=2.0)


class RankedScenarioItem(BaseModel):
    rank: int
    sequence: List[ShockStep]
    portfolio_loss: float
    causal_fidelity: float
    dfast_breach: float
    reward: float
    causal_pathway: str


class ProfileFingerprint(BaseModel):
    tech: float
    duration: float
    credit: float
    equity: float


class AdversarialRankedResponse(BaseModel):
    profile: str
    inferred_profile: bool
    profile_confidence: float
    profile_fingerprint: ProfileFingerprint
    model_version: str
    scenarios: List[RankedScenarioItem]
    seeds_tried: int
    vs_beam_pct: Optional[float] = None
    vulnerability_summary: str
    quality_note: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/worst-case", response_model=AdversarialResponse)
async def find_worst_case(req: AdversarialRequest):
    """
    Find the adversarial worst-case shock sequence for a portfolio profile.

    Scores all candidate shocks for each of n_seeds historical starting states,
    returns the combination that maximises portfolio loss under the causal graph.
    """
    from ml_pipeline.generative_engine_rl.adversarial_serve import get_engine

    try:
        engine = get_engine(req.portfolio_profile)
    except KeyError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model for '{req.portfolio_profile}' not loaded. "
                f"Training may still be running or bandit.pt is missing. "
                f"Check GET /api/v1/adversarial/status for available profiles."
            ),
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: engine.find_worst_case(n_seeds=req.n_seeds, ucb_beta=req.ucb_beta),
        )
    except Exception as e:
        logger.exception("Inference failed for profile=%s", req.portfolio_profile)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    sampled = None
    if result.sampled_state:
        sampled = SampledState(
            date=str(result.sampled_state.get("date", "")),
            regime_name=result.sampled_state.get("regime_name"),
            row_index=result.sampled_state.get("row_index"),
        )

    return AdversarialResponse(
        profile=result.profile,
        model_version=result.model_version,
        worst_sequence=[ShockStep(**s) for s in result.worst_sequence],
        portfolio_loss=round(result.portfolio_loss, 4),
        causal_fidelity=round(result.causal_fidelity, 4),
        diversity=round(result.diversity, 4),
        dfast_breach=round(result.dfast_breach, 4),
        sampled_state=sampled,
        seeds_tried=result.seeds_tried,
        seed_used=result.seed_used,
        vs_beam_pct=result.vs_beam_pct,
        quality_note=_quality_note(result.model_version, result.vs_beam_pct),
    )


@router.post("/ranked-scenarios", response_model=AdversarialRankedResponse)
async def find_ranked_scenarios(req: AdversarialRankedRequest):
    """
    Ranked adversarial scenario search for a user's portfolio.

    Accepts raw holdings, auto-infers the vulnerability profile via exposure
    fingerprinting (tech / duration / credit weights), runs the bandit across
    n_seeds market starting states, deduplicates by causal-pathway type, and
    returns the top_k distinct worst-case scenarios ranked by portfolio loss.

    Set portfolio_profile to override the auto-inferred profile.
    """
    from ml_pipeline.generative_engine_rl.adversarial_serve import (
        get_engine,
        compute_exposure_fingerprint,
        classify_profile,
    )

    holdings_dicts = [h.dict() for h in req.holdings]

    # Classify profile — either auto-infer or use the caller's override
    inferred = req.portfolio_profile is None
    if inferred:
        profile, confidence = classify_profile(holdings_dicts)
    else:
        profile = req.portfolio_profile
        fp_tmp = compute_exposure_fingerprint(holdings_dicts)
        _key_map = {
            "tech_heavy": "tech",
            "bond_heavy": "duration",
            "credit_heavy": "credit",
            "balanced": "equity",
        }
        confidence = fp_tmp.get(_key_map.get(profile, "equity"), 50.0)

    fingerprint = compute_exposure_fingerprint(holdings_dicts)

    try:
        engine = get_engine(profile)
    except KeyError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model for '{profile}' not loaded. "
                f"Check GET /api/v1/adversarial/status for available profiles."
            ),
        )

    loop = asyncio.get_event_loop()
    try:
        scenarios = await loop.run_in_executor(
            None,
            lambda: engine.find_ranked_scenarios(
                n_seeds=req.n_seeds,
                ucb_beta=req.ucb_beta,
                top_k=req.top_k,
            ),
        )
    except Exception as e:
        logger.exception("Ranked inference failed for profile=%s", profile)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    top = scenarios[0] if scenarios else None
    vuln_summary = (
        f"This portfolio is most vulnerable through the {top.causal_pathway} channel."
        if top else "No adversarial scenarios identified."
    )

    return AdversarialRankedResponse(
        profile=profile,
        inferred_profile=inferred,
        profile_confidence=round(confidence, 1),
        profile_fingerprint=ProfileFingerprint(**fingerprint),
        model_version=engine.model_version,
        scenarios=[
            RankedScenarioItem(
                rank=s.rank,
                sequence=[ShockStep(**step) for step in s.sequence],
                portfolio_loss=round(s.portfolio_loss, 4),
                causal_fidelity=round(s.causal_fidelity, 4),
                dfast_breach=round(s.dfast_breach, 4),
                reward=round(s.reward, 4),
                causal_pathway=s.causal_pathway,
            )
            for s in scenarios
        ],
        seeds_tried=req.n_seeds,
        vs_beam_pct=engine.vs_beam_pct,
        vulnerability_summary=vuln_summary,
        quality_note=_quality_note(engine.model_version, engine.vs_beam_pct),
    )


@router.get("/status")
async def adversarial_status():
    """List which portfolio profiles have a loaded bandit model."""
    from ml_pipeline.generative_engine_rl.adversarial_serve import _engines

    profiles_info: Dict[str, dict] = {}
    for profile in PROFILES:
        if profile in _engines:
            eng = _engines[profile]
            profiles_info[profile] = {
                "loaded": True,
                "model_version": eng.model_version,
                "vs_beam_pct": eng.vs_beam_pct,
                "n_actions": int(len(eng.catalog)),
            }
        else:
            profiles_info[profile] = {"loaded": False}

    return {
        "profiles": profiles_info,
        "loaded_count": sum(1 for v in profiles_info.values() if v["loaded"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality_note(version: str, vs_beam_pct: Optional[float]) -> str:
    if vs_beam_pct is None:
        return f"Bandit {version} — benchmark quality not yet evaluated."
    pct = round(vs_beam_pct)
    label = "2-step" if version == "v2" else "1-step"
    gate = "Exceeds 85% gate." if pct >= 85 else "Gate threshold is 85%."
    return f"Bandit {version} ({label}): {pct}% of exhaustive beam search quality. {gate}"
