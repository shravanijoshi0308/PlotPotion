"""
Baseline: BM25 word search (like Ctrl+F, but smarter about rare words).

With stemming: "village", "villagers" and "villages" all become "villag",
so they count as the same word.

Uses bm25s (https://github.com/xhluca/bm25s) + PyStemmer.
Run:  python -m src.retrievers.bm25
"""
import bm25s
import numpy as np
import Stemmer

from src.retrievers.base import Retriever, doc_text, load_corpus

STEMMER = Stemmer.Stemmer("english")


def tokenize(texts, return_ids=True):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=return_ids, show_progress=False)


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, corpus=None):
        self.corpus = corpus or load_corpus()
        self.ids = [d["doc_id"] for d in self.corpus]
        self.model = bm25s.BM25()
        self.model.index(tokenize([doc_text(d) for d in self.corpus]), show_progress=False)

    def scores_all(self, query):
        """BM25 score of the query for every movie (used by hybrid search)."""
        toks = tokenize([query], return_ids=False)[0]
        if not toks:                       # e.g. "the" or "x": no real words left -> all zeros
            return np.zeros(len(self.ids), dtype=np.float32)
        return self.model.get_scores(toks)

    def search(self, query, k=10):
        scores = self.scores_all(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    r = BM25Retriever()
    for doc_id, score in r.search("a prince raised in a village learns he is royal and fights a cruel king", k=10):
        print(f"{score:6.2f}  {doc_id}")
