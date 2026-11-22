"""
Shared helpers.

Every search method returns the same shape: a list of (doc_id, score), best first.
"""
import json
from pathlib import Path

CORPUS = Path("data/processed/corpus.jsonl")


def load_corpus(path=CORPUS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def doc_text(doc):
    """The text BM25 searches in: title + plot."""
    return f"{doc['title']}. {doc['plot']}"


def chunk_plot(doc, size=200, stride=150):
    """Split a plot into overlapping pieces of ~200 words (title added to each piece).
    Meaning models only read ~250 words, so chunks let them see the WHOLE plot."""
    words = doc["plot"].split()
    starts = range(0, max(len(words) - size + stride, 1), stride)
    return [f"{doc['title']}. " + " ".join(words[s:s + size]) for s in starts]


class Retriever:
    name = "base"

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError
