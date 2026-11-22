import re
import unicodedata

# Sections that are not useful for RAG
SKIP_SECTIONS = {
    "REF", "AUTH_CONT", "COMP_INT", "ACK_FUND",
    "SUPPL", "TABLE", "REVIEW_INFO",
}
SKIP_TYPES = {"ref", "table", "table_caption", "table_footnote"}

MIN_PASSAGE_CHARS = 40  # drop tiny fragments


def clean_text(text: str) -> str:
    # Normalize unicode (curly quotes, non-breaking spaces, ligatures)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove numeric citations: [1], [1,2], [1-3], [1, 3-5]
    text = re.sub(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]", "", text)

    # Remove author-year citations: (Smith et al., 2020) / (Smith, 2020; Lee, 2019)
    text = re.sub(
        r"\((?:[A-Z][A-Za-z\-]+(?: et al\.?| and [A-Z][A-Za-z\-]+)?,? \d{4}[a-z]?;? ?)+\)",
        "",
        text,
    )

    # Fix words broken by hyphen + newline
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Clean spaces: collapse spaces, remove space before punctuation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)          # empty parentheses left over
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess_article(article_data: dict) -> str:
    """BioC dict -> clean text, skipping references and non-content sections."""
    parts = []

    for document in article_data.get("documents", []):
        for passage in document.get("passages", []):
            infons = passage.get("infons", {})
            section = str(infons.get("section_type", "")).upper()
            ptype = str(infons.get("type", "")).lower()

            if section in SKIP_SECTIONS or ptype in SKIP_TYPES:
                continue

            text = clean_text(passage.get("text", ""))

            if len(text) >= MIN_PASSAGE_CHARS:
                parts.append(text)

    return "\n\n".join(parts)