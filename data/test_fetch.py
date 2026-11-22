import time

from data.first_source import search_pmc, fetch_pmc_article
from data.preprocessing import preprocess_article
from data.chunking import chunk_text
from data.vector_store import store_chunks, search_chunks

TOPIC = "anxiety"
MAX_RESULTS = 3


def main():
    print(f"Searching PMC for: '{TOPIC}'\n")
    article_ids = search_pmc(topic=TOPIC, max_results=MAX_RESULTS)

    if not article_ids:
        print("No articles found.")
        return

    print(f"Found {len(article_ids)} articles: {article_ids}\n")
    print("-" * 70)

    stored_total = 0

    for aid in article_ids:
        try:
            data = fetch_pmc_article(aid)
            text = preprocess_article(data)

            if not text:
                raise ValueError("empty text")

            title = text.split("\n\n")[0][:100]
            chunks = chunk_text(text)
            stored = store_chunks(aid, title, chunks)
            stored_total += stored

            print(f"✅ PMC{aid}")
            print(f"   Title : {title}")
            print(f"   Chars : {len(text):,}")
            print(f"   Chunks: {len(chunks)}")
            print(f"   Stored: {stored} chunks in Qdrant")

        except Exception as e:
            print(f"❌ PMC{aid} failed: {e}")

        time.sleep(1)
        print("-" * 70)

    print(f"\nTotal chunks stored: {stored_total}")

    # Quick retrieval test
    question = "What are the symptoms of anxiety?"
    print(f"\nTest search: '{question}'\n")

    for hit in search_chunks(question, limit=3):
        print(f"score={hit.score:.3f} | {hit.payload['pmcid']} | {hit.payload['title'][:60]}")
        print(hit.payload["text"][:250].replace("\n", " "), "\n")


if __name__ == "__main__":
    main()