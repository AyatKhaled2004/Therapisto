import time
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from APP.config import settings


SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

# Prefer review articles / meta-analyses: they summarize a whole topic
REVIEW_FILTER = (
    '(review[Title] OR "systematic review"[Title/Abstract] '
    'OR "meta-analysis"[Title/Abstract] OR guideline[Title])'
)

# NCBI allows ~3 requests/sec without a key, ~10 with a key
MIN_INTERVAL = 0.12 if settings.ncbi_api_key else 0.4
_last_call = 0.0


# ---------------------------------------------------------------
# HTTP helpers (retry + throttle)
# ---------------------------------------------------------------

def _build_session() -> requests.Session:
    """One shared session that retries on connection errors / 429 / 5xx."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,                       # waits 2s, 4s, 8s, ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "therapisto/1.0 (graduation project)"})
    return session


session = _build_session()


def _get(url: str, **kwargs) -> requests.Response:
    """GET with throttling so we never hammer the server."""
    global _last_call
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        return session.get(url, **kwargs)
    finally:
        _last_call = time.time()


def _ncbi_params(extra: dict) -> dict:
    params = {"tool": "therapisto", **extra}
    if getattr(settings, "ncbi_email", ""):
        params["email"] = settings.ncbi_email
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------

def search_pmc(topic: str, max_results: int = 3, reviews_only: bool = False):
    """Return PMC article IDs for a topic, best match first."""

    def run(term: str, retmax: int):
        params = _ncbi_params({
            "db": "pmc",
            "term": f"{term} AND open access[filter]",
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        })
        response = _get(SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()["esearchresult"]["idlist"]

    base = f'"{topic}"[Title/Abstract]'

    ids = []
    if reviews_only:
        ids = run(f"{base} AND {REVIEW_FILTER}", max_results)

    # not enough reviews? fill the rest with the best-matching articles
    if len(ids) < max_results:
        for aid in run(base, max_results * 2):
            if aid not in ids:
                ids.append(aid)

    return ids[:max_results]


# ---------------------------------------------------------------
# Fetch full text
# ---------------------------------------------------------------

def _fetch_bioc(pmcid: str):
    """Try BioC. Returns dict, or None if unavailable / blocked (caller falls back)."""
    url = f"{BIOC_URL}/BioC_json/{pmcid}/unicode"

    try:
        response = _get(url, timeout=60)
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    if "json" not in response.headers.get("Content-Type", "").lower():
        return None

    data = response.json()
    if isinstance(data, list):
        data = data[0] if data else None
    return data


def _fetch_efetch(pmcid: str):
    """Fallback: full text from PMC efetch XML, shaped like BioC."""
    params = _ncbi_params({
        "db": "pmc",
        "id": pmcid.replace("PMC", ""),
        "retmode": "xml",
    })
    response = _get(EFETCH_URL, params=params, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    def text_of(el):
        return " ".join("".join(el.itertext()).split())

    passages = []

    title = root.find(".//article-meta//article-title")
    if title is not None:
        passages.append({"text": text_of(title)})

    for p in root.findall(".//abstract//p"):
        passages.append({"text": text_of(p)})

    for p in root.findall(".//body//p"):
        passages.append({"text": text_of(p)})

    if not passages:
        return None

    return {"documents": [{"id": pmcid, "passages": passages}]}


def fetch_pmc_article(article_id: str):
    pmcid = (
        article_id
        if article_id.upper().startswith("PMC")
        else f"PMC{article_id}"
    )

    data = _fetch_bioc(pmcid)
    if data is None:
        data = _fetch_efetch(pmcid)

    if data is None:
        raise ValueError(f"No full text available for {pmcid}.")

    return data


def extract_article_text(article_data):
    text_parts = []

    for document in article_data.get("documents", []):
        for passage in document.get("passages", []):
            text = passage.get("text", "").strip()
            if text:
                text_parts.append(text)

    return "\n\n".join(text_parts)