"""
Baseline: BM25 word search (like Ctrl+F, but smarter about rare words).

Uses the bm25s library: https://github.com/xhluca/bm25s
"""
import bm25s

from src.retrievers.base import Retriever, doc_text, load_corpus


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, corpus=None):
        self.corpus = corpus or load_corpus()
        self.ids = [d["doc_id"] for d in self.corpus]
        tokens = bm25s.tokenize([doc_text(d) for d in self.corpus], stopwords="en", show_progress=False)
        self.model = bm25s.BM25()
        self.model.index(tokens, show_progress=False)

    def scores_all(self, query):
        """BM25 score of the query for every movie (used by hybrid search)."""
        toks = bm25s.tokenize(query, stopwords="en", return_ids=False, show_progress=False)[0]
        return self.model.get_scores(toks)

    def search(self, query, k=10):
        k = min(k, len(self.ids))
        q = bm25s.tokenize(query, stopwords="en", show_progress=False)
        idx, scores = self.model.retrieve(q, k=k, show_progress=False)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0])]


if __name__ == "__main__":
    r = BM25Retriever()
    for doc_id, score in r.search("animal king killed by his brother", k=10):
        print(f"{score:6.2f}  {doc_id}")
