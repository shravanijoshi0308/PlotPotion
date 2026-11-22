"""
The common shape every search method must follow.

BM25, dense (meaning) search and the reranker all return the same thing:
a list of (doc_id, score), best first. Then the API and the eval script
can use any of them without changes.
"""
import json
from pathlib import Path

CORPUS = Path("data/processed/corpus.jsonl")


def load_corpus(path=CORPUS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def doc_text(doc):
    """The text we search in: title + plot."""
    return f"{doc['title']}. {doc['plot']}"


class Retriever:
    name = "base"

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError
