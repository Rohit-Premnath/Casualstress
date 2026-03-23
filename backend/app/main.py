from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title="CausalStress API",
    description="AI-powered financial stress testing with causal discovery, regime detection, and generative scenarios",
    version="0.1.0",
)

# CORS - allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "CausalStress API",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.ENV,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": settings.POSTGRES_HOST,
        "redis": settings.REDIS_HOST,
    }