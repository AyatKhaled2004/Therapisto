import time
import uuid
from collections import deque

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from APP.config import settings

EMBED_DIM = 768            # smaller vectors = faster upload, almost same quality
EMBED_BATCH = 20           # texts per Gemini request
UPSERT_BATCH = 5           # points per Qdrant request
RATE_LIMIT_PER_MIN = 90    # Gemini free tier = 100/min, keep a margin
MAX_RETRIES = 6

gemini = genai.Client(api_key=settings.google_api_key)
qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=120,
)

_calls: deque = deque()


def _throttle(n: int):
    """Wait until we can embed n more texts without passing the per-minute limit."""
    while True:
        now = time.time()
        while _calls and now - _calls[0] > 60:
            _calls.popleft()

        if len(_calls) + n <= RATE_LIMIT_PER_MIN:
            _calls.extend([now] * n)
            return

        wait = 60 - (now - _calls[0]) + 0.5
        print(f"   (waiting {wait:.0f}s for rate limit...)", flush=True)
        time.sleep(max(wait, 1))


def _with_retry(fn, what: str):
    """Run fn(); on 429 / timeout / connection errors wait and retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)

            # Daily quota exhausted: waiting won't help
            if "PerDay" in msg:
                raise

            retryable = any(
                k in msg for k in
                ("429", "RESOURCE_EXHAUSTED", "timed out", "timeout",
                 "Connection", "503", "502", "500")
            )
            if not retryable or attempt == MAX_RETRIES:
                raise
            wait = 20 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 3 * attempt
            print(f"   ({what} failed, retry {attempt}/{MAX_RETRIES} in {wait}s)", flush=True)
            time.sleep(wait)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """ONE embedding per text, rate-limited and retried."""
    vectors = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        _throttle(len(batch))

        def call():
            return gemini.models.embed_content(
                model=settings.gemini_embed_model,
                contents=[types.Content(parts=[types.Part(text=t)]) for t in batch],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )

        response = _with_retry(call, "embedding")
        batch_vectors = [e.values for e in response.embeddings]

        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding count mismatch: sent {len(batch)}, got {len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)

    return vectors


def ensure_collection(vector_size: int = EMBED_DIM):
    """Create the collection, or fail early if it has the wrong vector size."""
    name = settings.qdrant_collection

    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    current = qdrant.get_collection(name).config.params.vectors
    current_size = getattr(current, "size", None)

    if current_size is not None and current_size != vector_size:
        raise RuntimeError(
            f"Collection '{name}' has vector size {current_size}, but the code uses "
            f"{vector_size}. Delete the collection and run again."
        )

def make_id(key: str) -> str:
    """Stable ID: same key => same point, so re-running never duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def existing_ids(ids: list[str]) -> set[str]:
    """Which of these IDs are already stored? (lets us resume after a failure)"""
    if not qdrant.collection_exists(settings.qdrant_collection):
        return set()
    found = _with_retry(
        lambda: qdrant.retrieve(
            collection_name=settings.qdrant_collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        ),
        "check existing",
    )
    return {str(p.id) for p in found}


def store_items(items: list[dict]) -> int:
    """
    items: [{"key": str, "text": str, "payload": dict}, ...]
    Embeds and stores only the ones not already in Qdrant.
    Returns the number newly stored.
    """
    if not items:
        return 0

    ensure_collection(EMBED_DIM)      # check/create BEFORE spending embedding quota

    for it in items:
        it["id"] = make_id(it["key"])

    done = existing_ids([it["id"] for it in items])
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return 0

    vectors = embed_texts([it["text"] for it in todo])

    points = [
        PointStruct(
            id=it["id"],
            vector=vec,
            payload={"text": it["text"], **it["payload"]},
        )
        for it, vec in zip(todo, vectors)
    ]

    for j in range(0, len(points), UPSERT_BATCH):
        part = points[j:j + UPSERT_BATCH]
        _with_retry(
            lambda part=part: qdrant.upsert(
                collection_name=settings.qdrant_collection, points=part
            ),
            "upsert",
        )

    return len(points)
def search_chunks(query: str, limit: int = 5):
    query_vector = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    result = _with_retry(
        lambda: qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=limit,
        ),
        "search",
    )
    return result.points