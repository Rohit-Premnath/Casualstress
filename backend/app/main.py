"""
CausalStress API
=================
FastAPI backend serving real data from the ML pipeline to the frontend.
All data comes from PostgreSQL — the same database ml_pipeline writes to.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import dashboard, causal, regimes, scenarios, stress_test, advisor, adversarial

logger = logging.getLogger("causalstress.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load bandit models once at startup — eliminates cold-start latency per request."""
    loop = asyncio.get_event_loop()
    try:
        from ml_pipeline.generative_engine_rl.adversarial_serve import load_all_engines
        logger.info("Pre-loading adversarial bandit models...")
        await loop.run_in_executor(None, load_all_engines)
        logger.info("Adversarial models ready.")
    except Exception as e:
        logger.warning("Could not pre-load adversarial models (non-fatal): %s", e)
    yield


app = FastAPI(
    title="CausalStress API",
    description="AI-powered financial stress testing with causal discovery, "
                "regime detection, and generative scenarios",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS - allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(dashboard.router)
app.include_router(causal.router)
app.include_router(regimes.router)
app.include_router(scenarios.router)
app.include_router(stress_test.router)
app.include_router(advisor.router)
app.include_router(adversarial.router)


@app.middleware("http")
async def log_requests(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.get("/")
async def root():
    return {
        "name": "CausalStress API",
        "version": "0.2.0",
        "status": "running",
        "environment": settings.ENV,
        "endpoints": {
            "dashboard": "/api/v1/dashboard/summary",
            "causal_graph": "/api/v1/causal/graph",
            "regimes": "/api/v1/regimes/current",
            "scenarios": "/api/v1/scenarios/latest",
            "stress_test": "/api/v1/stress-test/run",
            "adversarial_worst_case": "/api/v1/adversarial/worst-case",
            "adversarial_status": "/api/v1/adversarial/status",
            "advisor": "/api/v1/advisor/chat",
            "docs": "/docs",
        },
    }


@app.get("/live")
async def live_check():
    """Lightweight liveness probe that does not depend on the database."""
    return {
        "status": "alive",
        "service": "causalstress-api",
        "environment": settings.ENV,
    }


@app.get("/health")
async def health_check():
    """Health check with database connectivity test."""
    import psycopg
    db_status = "unknown"
    try:
        conn = psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "host": settings.POSTGRES_HOST,
        "port": settings.POSTGRES_PORT,
    }
