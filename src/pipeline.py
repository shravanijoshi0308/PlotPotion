"""
The full search chain:

  question
     ├─> BM25 (words, with stemming) ─┐
     │                                ├─> RRF fusion ─> top 30 ─> reranker (judge) ─> top k
     └─> meaning search (chunks) ─────┘

Every result carries all component scores + timing, so the page can show the bars and
the eval script can compare methods.

Run:  python -m src.pipeline "your question"
"""
import sys
import time

import numpy as np

from src.retrievers.base import Retriever, load_corpus
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever

RRF_K = 60       # standard value (Cormack et al., 2009)
DEPTH = 100      # results from each method that take part in the fusion vote
RERANK_N = 30    # how many candidates the judge re-reads

# Language / industry filter: the page's picker sends one of these keys.
INDIAN = ["Bollywood", "Tamil", "Telugu", "Malayalam", "Kannada", "Bengali", "Marathi", "Punjabi", "Assamese"]
LANGUAGES = {                    # key -> (label on the page, origins in the data)
    "all":        ("All movies", None),
    "indian":     ("Indian (all)", INDIAN),
    "bollywood":  ("Bollywood (Hindi)", ["Bollywood"]),
    "telugu":     ("Telugu", ["Telugu"]),
    "tamil":      ("Tamil", ["Tamil"]),
    "malayalam":  ("Malayalam", ["Malayalam"]),
    "kannada":    ("Kannada", ["Kannada"]),
    "bengali":    ("Bengali", ["Bengali", "Bangladeshi"]),
    "marathi":    ("Marathi", ["Marathi"]),
    "hollywood":  ("Hollywood (American)", ["American"]),
    "british":    ("British", ["British"]),
    "east_asian": ("Japanese, Korean & Chinese", ["Japanese", "South_Korean", "Chinese", "Hong Kong"]),
}


def top_ranks(scores: np.ndarray, depth: int) -> dict[int, int]:
    """{doc index: rank} for the best `depth` docs (docs filtered out have score -inf)."""
    depth = min(depth, len(scores))
    top = np.argpartition(-scores, depth - 1)[:depth]
    top = top[np.argsort(-scores[top])]
    top = top[np.isfinite(scores[top])]
    return {int(i): r + 1 for r, i in enumerate(top)}


class Pipeline:
    def __init__(self, corpus=None, use_reranker=True):
        self.corpus = corpus or load_corpus()
        self.ids = [d["doc_id"] for d in self.corpus]
        self.bm25 = BM25Retriever(self.corpus)
        self.dense = DenseRetriever(self.corpus)
        origins = np.array([d["origin"] for d in self.corpus])
        self.lang_mask = {key: (None if o is None else np.isin(origins, o)) for key, (_, o) in LANGUAGES.items()}
        self.lang_count = {key: int(len(origins) if m is None else m.sum()) for key, m in self.lang_mask.items()}
        self.reranker = None
        if use_reranker:
            from src.retrievers.rerank import Reranker
            self.reranker = Reranker()

    def run(self, query: str, k: int = 5, mode: str = "rerank", lang: str = "all"):
        """mode: bm25 | dense | hybrid | rerank (= hybrid + judge).  lang: a key of LANGUAGES."""
        t, timing = time.perf_counter(), {}
        mask = self.lang_mask.get(lang)                 # None = all movies

        s_bm25 = self.bm25.scores_all(query).astype(np.float32)
        if mask is not None:
            s_bm25 = np.where(mask, s_bm25, -np.inf)
        timing["bm25_ms"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

        if mode == "bm25":                       # words only: skip the slower meaning search
            s_dense = np.zeros(len(self.ids), dtype=np.float32)
        else:
            s_dense, best_chunk = self.dense.scores_all(query, return_best_chunk=True)
            if mask is not None:
                s_dense = np.where(mask, s_dense, -np.inf)
            timing["dense_ms"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

        if mode == "bm25":
            order = list(top_ranks(s_bm25, k))
            fused = {}
        elif mode == "dense":
            order = list(top_ranks(s_dense, k))
            fused = {}
        else:
            fused = {}
            for ranks in (top_ranks(s_bm25, DEPTH), top_ranks(s_dense, DEPTH)):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
            order = sorted(fused, key=fused.get, reverse=True)
            timing["fusion_ms"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

        judge = {}
        if mode == "rerank" and self.reranker is not None:
            cand = order[:RERANK_N]
            passages = [self.dense.chunks[best_chunk[i]] for i in cand]   # the most relevant piece of each plot
            scores = self.reranker.score(query, passages)
            judge = {i: float(s) for i, s in zip(cand, scores)}
            order = sorted(cand, key=judge.get, reverse=True)
            timing["rerank_ms"] = (time.perf_counter() - t) * 1000

        timing["total_ms"] = sum(timing.values())
        results = []
        for i in order[:k]:
            r = {"doc_id": self.ids[i], "bm25": float(s_bm25[i])}
            if mode != "bm25":
                r["dense"] = float(s_dense[i])
            if i in fused:
                r["hybrid"] = fused[i]
            if i in judge:
                r["rerank"] = judge[i]
            results.append(r)
        return results, {key: round(v, 1) for key, v in timing.items()}


class PipelineRetriever(Retriever):
    """Wraps one pipeline mode as a plain Retriever (for the eval script)."""

    def __init__(self, pipeline: Pipeline, mode: str):
        self.p, self.mode, self.name = pipeline, mode, mode

    def search(self, query, k=10):
        res, _ = self.p.run(query, k=k, mode=self.mode)
        return [(r["doc_id"], r.get(self.mode, r.get("hybrid", 0.0))) for r in res]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "a prince raised in a village learns he is royal and fights a cruel king"
    p = Pipeline()
    for mode in ("bm25", "dense", "hybrid", "rerank"):
        res, timing = p.run(q, k=5, mode=mode)
        print(f"\n[{mode}]  {timing}")
        for n, r in enumerate(res, 1):
            extra = f"  judge={r['rerank']:.3f}" if "rerank" in r else ""
            print(f"  {n}. {r['doc_id']}  (bm25={r['bm25']:.2f}" + (f", meaning={r['dense']:.3f}" if "dense" in r else "") + f"{extra})")
