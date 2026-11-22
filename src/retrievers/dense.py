"""
Meaning search ("dense retrieval").

A small model turns every movie plot into a list of 384 numbers (an "embedding")
that captures what the plot is about. A question gets turned into numbers the same
way, and we pick the movies whose numbers point in the most similar direction.

So "a man lives the same day again and again" can find Groundhog Day,
even though the plot never uses those exact words.

Model: sentence-transformers/all-MiniLM-L6-v2 (small, fast, free).

First run builds the embeddings for all movies and saves them to
data/processed/embeddings.npy (takes a few minutes). After that it loads in seconds.

Run:  python -m src.retrievers.dense
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.base import Retriever, doc_text, load_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = Path("data/processed/embeddings.npy")


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, corpus=None, model_name=MODEL_NAME, emb_path=EMB_PATH):
        self.corpus = corpus or load_corpus()
        self.ids = [d["doc_id"] for d in self.corpus]
        self.model = SentenceTransformer(model_name)
        self.emb_path = Path(emb_path)

        if self.emb_path.exists():
            self.emb = np.load(self.emb_path)
            if len(self.emb) != len(self.ids):  # corpus changed -> rebuild
                print("Corpus changed, rebuilding embeddings...")
                self.emb = self._build()
        else:
            self.emb = self._build()

    def _build(self):
        print(f"Embedding {len(self.corpus)} movies with {MODEL_NAME} (one-time, a few minutes)...")
        # The model reads at most ~256 tokens, so it sees the title + start of the plot.
        texts = [doc_text(d) for d in self.corpus]
        emb = self.model.encode(texts, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        self.emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.emb_path, emb)
        print(f"Saved to {self.emb_path}")
        return emb

    def scores_all(self, query: str) -> np.ndarray:
        """Similarity of the query to every movie (cosine, -1..1)."""
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.emb @ q

    def search(self, query, k=10):
        scores = self.scores_all(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    r = DenseRetriever()
    for q in ["animal king killed by his brother", "a weatherman lives the same day again and again"]:
        print(f"\n{q}")
        for doc_id, score in r.search(q, k=5):
            print(f"  {score:5.3f}  {doc_id}")
