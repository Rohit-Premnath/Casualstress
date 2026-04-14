"""
AI Advisor API Router
Bridges the frontend chat UI to the real narrative advisor engine.
"""

from functools import lru_cache
from typing import Any, List
import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if "/ml_pipeline" not in sys.path:
    sys.path.append("/ml_pipeline")

from app.config import settings

router = APIRouter(prefix="/api/v1/advisor", tags=["advisor"])


def get_conn():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@lru_cache(maxsize=1)
def get_advisor() -> Any:
    """Reuse the real advisor engine across requests."""
    from ml_pipeline.narrative.advisor_engine import AdvisorChat
    return AdvisorChat()


def reset_conversation(advisor: Any, history: List[ChatMessage]) -> None:
    """Seed the advisor engine from frontend history before processing a turn."""
    advisor.conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history[-10:]
        if msg.role in {"user", "assistant"} and msg.content
    ]


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI Financial Risk Advisor using the real tool-enabled engine."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "role": "assistant",
            "content": "I'm the CausalStress AI Advisor. To enable the real narrative engine, please set the ANTHROPIC_API_KEY environment variable.",
        }

    try:
        advisor = get_advisor()
        reset_conversation(advisor, request.history)
        response = advisor.chat(request.message)
        return {"role": "assistant", "content": response}
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Advisor engine error: {str(e)}",
        }


@router.get("/suggested-prompts")
async def get_suggested_prompts():
    """Return context-aware suggested prompts."""
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT regime_name FROM models.regimes ORDER BY date DESC LIMIT 1
    """)
    regime = cursor.fetchone()
    cursor.close()
    conn.close()

    regime_name = regime["regime_name"] if regime else "unknown"

    return [
        f"What's the current portfolio risk under the {regime_name} regime?",
        "Simulate a 2008-style credit crisis scenario",
        "Explain the top causal links affecting financial sector (XLF)",
        "Compare our stress projections to the Fed's DFAST 2026 scenario",
        "What would happen if credit spreads blow out by 300bps?",
    ]
