"""
Meaning search ("dense retrieval"), now over plot CHUNKS.

Each plot is cut into overlapping ~200-word pieces. A small model turns every piece
into 384 numbers (an "embedding"). A movie's score = its BEST matching piece, so a
detail from the end of a plot can still be found.

Model: sentence-transformers/all-MiniLM-L6-v2 (small, fast, free).

First run embeds all pieces and saves them (about 10 minutes on a laptop, once).
Run:  python -m src.retrievers.dense
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.base import Retriever, chunk_plot, load_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = Path("data/processed/chunk_embeddings.npy")
MAP_PATH = Path("data/processed/chunk_doc.npy")


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, corpus=None, model_name=MODEL_NAME, emb_path=EMB_PATH, map_path=MAP_PATH):
        self.corpus = corpus or load_corpus()
        self.ids = [d["doc_id"] for d in self.corpus]
        self.model = SentenceTransformer(model_name)

        # chunk texts are rebuilt every start (fast); only the embeddings are cached
        self.chunks, owner = [], []
        for i, d in enumerate(self.corpus):
            for c in chunk_plot(d):
                self.chunks.append(c)
                owner.append(i)
        self.chunk_doc = np.array(owner, dtype=np.int32)

        emb_path, map_path = Path(emb_path), Path(map_path)
        if emb_path.exists() and map_path.exists() and np.array_equal(np.load(map_path), self.chunk_doc):
            self.emb = np.load(emb_path)
        else:
            print(f"Embedding {len(self.chunks)} plot pieces from {len(self.corpus)} movies "
                  f"with {MODEL_NAME} (one-time, ~10 minutes)...")
            self.emb = self.model.encode(self.chunks, batch_size=64, show_progress_bar=True,
                                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, self.emb)
            np.save(map_path, self.chunk_doc)
            print(f"Saved to {emb_path}")

    def chunk_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.emb @ q

    def scores_all(self, query: str, return_best_chunk=False):
        """Per movie: score of its best piece (cosine, -1..1)."""
        cs = self.chunk_scores(query)
        best = np.full(len(self.ids), -1.0, dtype=np.float32)
        np.maximum.at(best, self.chunk_doc, cs)
        if not return_best_chunk:
            return best
        # index of the best piece for every movie (used to show the reranker the right passage)
        order = np.lexsort((-cs, self.chunk_doc))           # by movie, best piece first
        first = np.r_[0, np.flatnonzero(np.diff(self.chunk_doc[order])) + 1]
        best_chunk = np.empty(len(self.ids), dtype=np.int64)
        best_chunk[self.chunk_doc[order[first]]] = order[first]
        return best, best_chunk

    def search(self, query, k=10):
        scores = self.scores_all(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    r = DenseRetriever()
    for q in ["animal king killed by his brother",
              "a weatherman lives the same day again and again",
              "a prince raised in a village learns he is royal and fights a cruel king"]:
        print(f"\n{q}")
        for doc_id, score in r.search(q, k=5):
            print(f"  {score:5.3f}  {doc_id}")
