"""
The judge: a cross-encoder reranker.

BM25 and meaning search are fast but rough: they look at the question and the plot
separately. The cross-encoder reads the question AND a plot passage together, word by
word, so it is much more careful, but also slower. So we only let it re-sort the
top 30 candidates from hybrid search.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, free).
"""
import numpy as np
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    name = "rerank"

    def __init__(self, model_name=MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance 0..1 for each passage (sigmoid of the model's raw score)."""
        raw = np.asarray(self.model.predict([(query, p) for p in passages], show_progress_bar=False), dtype=np.float32)
        return 1 / (1 + np.exp(-raw))
