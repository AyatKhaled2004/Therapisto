CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # shared text between neighbouring chunks


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into chunks, preferring paragraph boundaries."""
    chunks, current = [], ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            while len(paragraph) > size:
                chunks.append(paragraph[:size])
                paragraph = paragraph[size - overlap:]
            current = paragraph

    if current:
        chunks.append(current)

    return chunks