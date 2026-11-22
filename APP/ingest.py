import json
import time
from pathlib import Path

from data.first_source import search_pmc, fetch_pmc_article
from data.preprocessing import preprocess_article
from data.chunking import chunk_text
from data.vector_store import store_items

TOPICS = [
    "anxiety disorders",
    "generalized anxiety disorder",
    "panic disorder",
    "social anxiety disorder",
    "major depressive disorder",
    "depression treatment",
    "cognitive behavioral therapy",
    "post-traumatic stress disorder",
    "insomnia",
    "eating disorders",
    "stress management",
    "obsessive-compulsive disorder",
    "bipolar disorder",
    "mindfulness",
]

PER_TOPIC = 4            # articles per topic
REVIEWS_ONLY = True      # prefer review articles / meta-analyses
MIN_CHARS = 2000         # skip articles with almost no text
STATE_FILE = Path("data/ingested_pmc.json")


def load_done() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_done(done: set[str]):
    STATE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")


def main():
    done = load_done()
    total_chunks = 0
    failed = []

    for topic in TOPICS:
        print(f"\n=== {topic} ===")

        try:
            ids = search_pmc(topic, max_results=PER_TOPIC, reviews_only=REVIEWS_ONLY)
        except Exception as e:
            print(f"  search failed: {e}")
            continue

        for aid in ids:
            if aid in done:
                print(f"  - PMC{aid}: already stored, skipping")
                continue

            try:
                text = preprocess_article(fetch_pmc_article(aid))

                if len(text) < MIN_CHARS:
                    raise ValueError("not enough text")

                title = text.split("\n\n")[0][:150]
                chunks = chunk_text(text)

                items = [
                    {
                        "key": f"PMC{aid}-{i}",
                        "text": chunk,
                        "payload": {
                            "pmcid": f"PMC{aid}",
                            "title": title,
                            "topic": topic,
                            "chunk_index": i,
                            "source": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{aid}/",
                        },
                    }
                    for i, chunk in enumerate(chunks)
                ]

                stored = store_items(items)
                total_chunks += stored
                done.add(aid)
                save_done(done)

                print(f"  ✅ PMC{aid}: {len(chunks)} chunks ({stored} new) | {title[:70]}")

            except Exception as e:
                failed.append(aid)
                print(f"  ❌ PMC{aid}: {e}")

            time.sleep(1)

    print(f"\nDone. New chunks stored: {total_chunks} | Failed: {len(failed)}")


if __name__ == "__main__":
    main()