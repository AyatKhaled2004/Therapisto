
import logging
import secrets
import threading
import uuid

from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)


# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from APP.agent import graph
from APP.config import settings


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------

app = FastAPI(
    title="Therapisto API",
    version="1.0"
)


# ---------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=settings.allowed_origins_list,

    allow_methods=["GET", "POST"],

    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------

bearer = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
):

    expected_token = (settings.api_token or "").strip()

    if not expected_token:

        raise HTTPException(
            status_code=500,
            detail="API_TOKEN is not configured on the server."
        )

    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(),
        expected_token.encode()
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------
# SQLite Memory + Graph Lock
# ---------------------------------------------------------------

# Local model and SQLite memory:
# One request at a time is the safe choice.

_graph_lock = threading.Lock()


# ---------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$"
    )


# ---------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------

class ChatResponse(BaseModel):

    answer: str

    session_id: str

    route: str

    sources: list[dict]


# ---------------------------------------------------------------
# Build Sources
# ---------------------------------------------------------------

def build_sources(result: dict) -> list[dict]:

    route = result.get("route", "")

    # Don't expose sources in crisis responses.

    if route == "crisis":
        return []

    sources = []

    seen = set()


    # Knowledge Base Sources

    for item in result.get("kb", []):

        key = (
            item.get("pmcid") or item.get("source"),
            item.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "type": "kb",

                "title": (
                    item.get("title")
                    or item.get("source")
                    or ""
                ),

                "source": item.get("source", ""),

                "page": item.get("page"),

                "pmcid": item.get("pmcid", ""),

                "url": item.get("source_url", ""),

                "score": round(
                    item.get("score", 0.0),
                    3
                ),
            }
        )


    # Web Sources

    if route == "web":

        for item in result.get("web", []):

            sources.append(
                {
                    "type": "web",

                    "title": item.get("title", ""),

                    "url": item.get("url", ""),
                }
            )

    return sources


# ---------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_token)]
)

def chat(req: ChatRequest):

    session_id = req.session_id or uuid.uuid4().hex

    try:

        with _graph_lock:

            result = graph.invoke(

                {
                    "question": req.question.strip(),

                    "image_description": ""
                },

                config={
                    "configurable": {
                        "thread_id": session_id
                    }
                }
            )

    except Exception:

        logger.exception(
            "Chat request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong while "
                "generating the answer."
            )
        )


    return ChatResponse(

        answer=result.get("answer", ""),

        session_id=session_id,

        route=result.get("route", ""),

        sources=build_sources(result),
    )
