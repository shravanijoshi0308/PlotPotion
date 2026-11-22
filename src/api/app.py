"""
The web API.  GET /search?q=...&k=5

Run locally:  uvicorn src.api.app:app --reload
Then open:    http://127.0.0.1:8000/          (the PlotPotion page)
              http://127.0.0.1:8000/search?q=animal%20king%20killed%20by%20his%20brother
Docs page:    http://127.0.0.1:8000/docs
"""
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.retrievers.base import load_corpus
from src.retrievers.bm25 import BM25Retriever

app = FastAPI(title="PlotPotion", description="Brew a plot, reveal a movie")

corpus = load_corpus()
by_id = {d["doc_id"]: d for d in corpus}
bm25 = BM25Retriever(corpus)


INDEX = Path(__file__).resolve().parents[2] / "static" / "index.html"


@app.get("/")
def home():
    """The PlotPotion web page."""
    return FileResponse(INDEX)


@app.get("/search")
def search(q: str, k: int = 5):
    start = time.perf_counter()
    results = bm25.search(q, k=k)
    ms = (time.perf_counter() - start) * 1000

    return {
        "query": q,
        "latency_ms": round(ms, 1),
        "results": [
            {
                "doc_id": doc_id,
                "title": by_id[doc_id]["title"],
                "year": by_id[doc_id]["year"],
                "plot_snippet": by_id[doc_id]["plot"][:200] + "...",
                # Later: add "dense" and "rerank" scores here too
                "scores": {"bm25": round(score, 3)},
            }
            for doc_id, score in results
        ],
    }