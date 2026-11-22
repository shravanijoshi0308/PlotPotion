"""
Hybrid search: BM25 (words) + dense (meaning), mixed with Reciprocal Rank Fusion (RRF).

RRF in one sentence: every method gives each movie a rank, and a movie gets
1 / (60 + rank) points from each method. Movies that rank well in BOTH win.
It only uses ranks, so we never have to compare BM25 scores (0..20) with
cosine scores (-1..1) directly.

Run:  python -m src.retrievers.hybrid
"""
import numpy as np

from src.retrievers.base import Retriever, load_corpus

RRF_K = 60      # standard value from the RRF paper (Cormack et al., 2009)
DEPTH = 100     # how many results from each method take part in the vote


def top_ranks(scores: np.ndarray, depth: int) -> dict[int, int]:
    """{doc index: rank (1 = best)} for the top `depth` docs."""
    depth = min(depth, len(scores))
    top = np.argpartition(-scores, depth - 1)[:depth]
    top = top[np.argsort(-scores[top])]
    return {int(i): r + 1 for r, i in enumerate(top)}


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, bm25, dense, rrf_k=RRF_K, depth=DEPTH):
        self.bm25, self.dense = bm25, dense
        self.ids = bm25.ids
        assert self.ids == dense.ids, "BM25 and dense must use the same corpus"
        self.rrf_k, self.depth = rrf_k, depth

    def search_detailed(self, query, k=10):
        """Top k as dicts with every component's score (used by the API for the bars)."""
        s_bm25 = self.bm25.scores_all(query)
        s_dense = self.dense.scores_all(query)
        fused = {}
        for ranks in (top_ranks(s_bm25, self.depth), top_ranks(s_dense, self.depth)):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{"doc_id": self.ids[i], "hybrid": fused[i],
                 "bm25": float(s_bm25[i]), "dense": float(s_dense[i])} for i in best]

    def search(self, query, k=10):
        return [(r["doc_id"], r["hybrid"]) for r in self.search_detailed(query, k)]


if __name__ == "__main__":
    from src.retrievers.bm25 import BM25Retriever
    from src.retrievers.dense import DenseRetriever

    corpus = load_corpus()
    h = HybridRetriever(BM25Retriever(corpus), DenseRetriever(corpus))
    for q in ["animal king killed by his brother", "a weatherman lives the same day again and again"]:
        print(f"\n{q}")
        for r in h.search_detailed(q, k=5):
            print(f"  rrf={r['hybrid']:.4f}  bm25={r['bm25']:5.2f}  dense={r['dense']:.3f}  {r['doc_id']}")
