import logging
import secrets
import threading
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from APP.agent import graph
from APP.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Therapisto API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

bearer = HTTPBearer(auto_error=False)

# SQLite memory + a local model: one request at a time is the safe choice
_graph_lock = threading.Lock()


# ---------------------------------------------------------------
# Auth
# ---------------------------------------------------------------

def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if not settings.api_token:
        raise HTTPException(status_code=500, detail="API_TOKEN is not configured on the server.")

    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), settings.api_token.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    route: str
    sources: list[dict]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def build_sources(result: dict) -> list[dict]:
    route = result.get("route", "")

    if route == "crisis":
        return []

    sources, seen = [], set()

    for item in result.get("kb", []):
        key = (item.get("pmcid") or item.get("source"), item.get("page"))
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "type": "kb",
            "title": item.get("title") or item.get("source") or "",
            "source": item.get("source", ""),
            "page": item.get("page"),
            "pmcid": item.get("pmcid", ""),
            "url": item.get("source_url", ""),
            "score": round(item.get("score", 0.0), 3),
        })

    if route == "web":
        for item in result.get("web", []):
            sources.append({
                "type": "web",
                "title": item.get("title", ""),
                "url": item.get("url", ""),
            })

    return sources


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex

    try:
        with _graph_lock:
            result = graph.invoke(
                {"question": req.question.strip(), "image_description": ""},
                config={"configurable": {"thread_id": session_id}},
            )
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Something went wrong while generating the answer.")

    return ChatResponse(
        answer=result.get("answer", ""),
        session_id=session_id,
        route=result.get("route", ""),
        sources=build_sources(result),
    )