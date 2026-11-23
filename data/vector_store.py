
import os
import re
import time
import uuid
import logging

from ollama import Client

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from APP.config import settings


# =====================================================
# 1. Configuration
# =====================================================

logger = logging.getLogger(__name__)

# Local Ollama configuration
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "embeddinggemma"
)

OLLAMA_TIMEOUT = 300

# Processing configuration
EMBED_BATCH = 8
UPSERT_BATCH = 20
MAX_RETRIES = 4

# Optional model-specific prefixes
# Nomic models use search_document: and search_query:
# EmbeddingGemma uses plain text by default.
DOCUMENT_PREFIX = os.getenv(
    "OLLAMA_DOCUMENT_PREFIX",
    ""
)

QUERY_PREFIX = os.getenv(
    "OLLAMA_QUERY_PREFIX",
    ""
)

if "nomic-embed-text" in OLLAMA_EMBED_MODEL.lower():

    if not DOCUMENT_PREFIX:
        DOCUMENT_PREFIX = "search_document: "

    if not QUERY_PREFIX:
        QUERY_PREFIX = "search_query: "


# =====================================================
# 2. Qdrant Collection Configuration
# =====================================================

# Keep Ollama vectors separate from old Gemini vectors.
# This prevents mixing embeddings from different models.

model_slug = re.sub(
    r"[^a-zA-Z0-9_-]",
    "_",
    OLLAMA_EMBED_MODEL
)

COLLECTION_NAME = os.getenv(
    "QDRANT_OLLAMA_COLLECTION",
    f"{settings.qdrant_collection}_ollama_{model_slug}"
)


# =====================================================
# 3. Initialize Clients
# =====================================================

ollama_client = Client(
    host=OLLAMA_HOST,
    timeout=OLLAMA_TIMEOUT
)

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=120,
)


# =====================================================
# 4. Retry Mechanism
# =====================================================

def _with_retry(fn, what: str):
    """
    Execute a function with retries for temporary errors.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return fn()

        except Exception as error:

            message = str(error).lower()

            status_code = getattr(
                error,
                "status_code",
                None
            )

            # Errors that should not be retried
            permanent_errors = [
                "model not found",
                "pull model",
                "invalid model",
                "dimension mismatch",
            ]

            if any(
                item in message
                for item in permanent_errors
            ):
                raise

            retryable = (

                status_code in (
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                )

                or isinstance(
                    error,
                    (TimeoutError, ConnectionError)
                )

                or any(
                    item in message
                    for item in [
                        "timeout",
                        "timed out",
                        "connection",
                        "temporarily unavailable",
                        "server disconnected",
                        "503",
                        "502",
                        "500",
                    ]
                )
            )

            if not retryable or attempt == MAX_RETRIES:

                logger.exception(
                    "%s failed permanently.",
                    what
                )

                raise

            wait_time = min(
                2 ** attempt,
                30
            )

            logger.warning(
                "%s failed: %s | Retry %d/%d in %d seconds",
                what,
                error,
                attempt,
                MAX_RETRIES,
                wait_time
            )

            time.sleep(wait_time)


# =====================================================
# 5. Prepare Embedding Input
# =====================================================

def _prepare_texts(
    texts: list[str],
    task_type: str
) -> list[str]:
    """
    Apply model-specific prefixes when configured.
    """

    if task_type in (
        "RETRIEVAL_QUERY",
        "QUERY"
    ):

        prefix = QUERY_PREFIX

    else:

        prefix = DOCUMENT_PREFIX

    return [
        f"{prefix}{text}"
        for text in texts
    ]


# =====================================================
# 6. Generate Embeddings Using Ollama
# =====================================================

def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """
    Generate embeddings locally using Ollama.

    Supports batches and returns one vector per text.
    """

    if not texts:
        return []

    if not OLLAMA_EMBED_MODEL:
        raise ValueError(
            "OLLAMA_EMBED_MODEL is not configured."
        )

    vectors = []

    prepared_texts = _prepare_texts(
        texts,
        task_type
    )

    # Process texts in smaller batches
    for start in range(
        0,
        len(prepared_texts),
        EMBED_BATCH
    ):

        batch = prepared_texts[
            start:start + EMBED_BATCH
        ]

        def call():

            return ollama_client.embed(
                model=OLLAMA_EMBED_MODEL,
                input=batch,
                truncate=True,
            )

        response = _with_retry(
            call,
            "Ollama embedding"
        )

        batch_vectors = response.embeddings

        if len(batch_vectors) != len(batch):

            raise ValueError(
                f"Embedding count mismatch: "
                f"sent {len(batch)}, "
                f"received {len(batch_vectors)}"
            )

        for vector in batch_vectors:

            if not vector:

                raise ValueError(
                    "Ollama returned an empty embedding."
                )

            vectors.append([
                float(value)
                for value in vector
            ])

    # Verify that every vector has the same dimension
    dimensions = {
        len(vector)
        for vector in vectors
    }

    if len(dimensions) != 1:

        raise ValueError(
            f"Inconsistent embedding dimensions: {dimensions}"
        )

    return vectors


# =====================================================
# 7. Ensure Qdrant Collection
# =====================================================

def ensure_collection(vector_size: int):
    """
    Create the collection if it does not exist.
    Validate the vector dimension if it already exists.
    """

    name = COLLECTION_NAME

    if not qdrant.collection_exists(name):

        logger.info(
            "Creating Qdrant collection: %s",
            name
        )

        qdrant.create_collection(

            collection_name=name,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        return

    collection = qdrant.get_collection(name)

    current = collection.config.params.vectors

    current_size = getattr(
        current,
        "size",
        None
    )

    if current_size is None:

        raise RuntimeError(
            "This code expects an unnamed Qdrant vector configuration."
        )

    if current_size != vector_size:

        raise RuntimeError(
            f"Collection '{name}' has dimension "
            f"{current_size}, but Ollama produces "
            f"{vector_size} dimensions. "
            "Use a separate collection for the new configuration."
        )


# =====================================================
# 8. Stable Chunk ID
# =====================================================

def make_id(key: str) -> str:
    """
    Generate a deterministic UUID.

    The same key always produces the same ID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            str(key)
        )
    )


# =====================================================
# 9. Check Existing IDs
# =====================================================

def existing_ids(ids: list[str]) -> set[str]:
    """
    Return IDs already stored in Qdrant.
    Allows ingestion to resume after failures.
    """

    if not ids:
        return set()

    if not qdrant.collection_exists(COLLECTION_NAME):
        return set()

    found = _with_retry(

        lambda: qdrant.retrieve(

            collection_name=COLLECTION_NAME,

            ids=ids,

            with_payload=False,

            with_vectors=False,
        ),

        "Check existing IDs"
    )

    return {
        str(point.id)
        for point in found
    }


# =====================================================
# 10. Store Items in Qdrant
# =====================================================

def store_items(items: list[dict]) -> int:
    """
    Expected input:

    [
        {
            "key": "PMC123456_chunk_0",
            "text": "Article content...",
            "payload": {
                "pmcid": "PMC123456",
                "title": "Article title",
                "chunk_index": 0,
                "source": "PMC"
            }
        }
    ]

    Generates embeddings and stores new items only.

    Returns the number of newly stored points.
    """

    if not items:
        return 0

    prepared_items = []

    for item in items:

        key = str(
            item["key"]
        ).strip()

        text = str(
            item["text"]
        ).strip()

        if not key or not text:
            continue

        payload = dict(
            item.get("payload") or {}
        )

        point_id = make_id(key)

        prepared_items.append({

            "id": point_id,

            "key": key,

            "text": text,

            "payload": payload,

        })

    if not prepared_items:
        return 0

    # Check which items already exist
    ids = [
        item["id"]
        for item in prepared_items
    ]

    done = existing_ids(ids)

    todo = [
        item
        for item in prepared_items
        if item["id"] not in done
    ]

    if not todo:

        logger.info(
            "All chunks already exist. Skipping."
        )

        return 0

    total_stored = 0

    # Embed and store in smaller batches
    for start in range(
        0,
        len(todo),
        EMBED_BATCH
    ):

        batch = todo[
            start:start + EMBED_BATCH
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        # Generate Ollama embeddings
        vectors = embed_texts(
            texts,
            task_type="RETRIEVAL_DOCUMENT"
        )

        if len(vectors) != len(batch):

            raise ValueError(
                "Number of embeddings does not match number of items."
            )

        vector_size = len(vectors[0])

        # Create collection or verify dimensions
        ensure_collection(vector_size)

        points = []

        for item, vector in zip(
            batch,
            vectors
        ):

            payload = {
                **item["payload"],

                "text": item["text"],

                "embedding_model": OLLAMA_EMBED_MODEL,
            }

            points.append(
                PointStruct(

                    id=item["id"],

                    vector=vector,

                    payload=payload,
                )
            )

        # Upload points in smaller batches
        for start_point in range(
            0,
            len(points),
            UPSERT_BATCH
        ):

            part = points[
                start_point:start_point + UPSERT_BATCH
            ]

            _with_retry(

                lambda part=part: qdrant.upsert(

                    collection_name=COLLECTION_NAME,

                    points=part,
                ),

                "Qdrant upsert"
            )

        total_stored += len(points)

        logger.info(
            "Stored %d/%d new chunks",
            total_stored,
            len(todo)
        )

    return total_stored


# =====================================================
# 11. Search in Qdrant
# =====================================================

def search_chunks(
    query: str,
    limit: int = 5
):
    """
    Embed the user query using the same Ollama model,
    then retrieve similar chunks from Qdrant.
    """

    if not query or not query.strip():
        return []

    # Query must use the same model as indexed documents
    query_vector = embed_texts(

        [query],

        task_type="RETRIEVAL_QUERY"
    )[0]

    # Validate collection dimension
    ensure_collection(
        len(query_vector)
    )

    result = _with_retry(

        lambda: qdrant.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            limit=limit,

            with_payload=True,

            with_vectors=False,
        ),

        "Qdrant search"
    )

    return result.points


# =====================================================
# 12. Collection Information
# =====================================================

def collection_info():
    """
    Return basic information about the active collection.
    """

    if not qdrant.collection_exists(COLLECTION_NAME):

        return {
            "exists": False,
            "collection": COLLECTION_NAME,
            "embedding_model": OLLAMA_EMBED_MODEL,
        }

    info = qdrant.get_collection(
        COLLECTION_NAME
    )

    return {

        "exists": True,

        "collection": COLLECTION_NAME,

        "embedding_model": OLLAMA_EMBED_MODEL,

        "points_count": info.points_count,

        "vectors_count": info.vectors_count,

    }
