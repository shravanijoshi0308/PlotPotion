"""
The web API.  GET /search?q=...&k=5&mode=hybrid

mode = hybrid (default) | bm25 | dense

Run locally:  uvicorn src.api.app:app --reload --port 8001
Then open:    http://127.0.0.1:8001/          (the PlotPotion page)
Docs page:    http://127.0.0.1:8001/docs
"""
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from src.retrievers.base import load_corpus
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever

app = FastAPI(title="PlotPotion", description="Brew a plot, reveal a movie")

corpus = load_corpus()
by_id = {d["doc_id"]: d for d in corpus}
bm25 = BM25Retriever(corpus)
dense = DenseRetriever(corpus)
hybrid = HybridRetriever(bm25, dense)

INDEX = Path(__file__).resolve().parents[2] / "static" / "index.html"


@app.get("/")
def home():
    """The PlotPotion web page."""
    return FileResponse(INDEX)


@app.get("/search")
def search(q: str, k: int = 5, mode: str = Query("hybrid", pattern="^(hybrid|bm25|dense)$")):
    start = time.perf_counter()
    if mode == "hybrid":
        rows = hybrid.search_detailed(q, k=k)
    else:
        retriever = bm25 if mode == "bm25" else dense
        rows = [{"doc_id": d, mode: s} for d, s in retriever.search(q, k=k)]
    ms = (time.perf_counter() - start) * 1000

    return {
        "query": q,
        "mode": mode,
        "latency_ms": round(ms, 1),
        "results": [
            {
                "doc_id": r["doc_id"],
                "title": by_id[r["doc_id"]]["title"],
                "year": by_id[r["doc_id"]]["year"],
                "plot_snippet": by_id[r["doc_id"]]["plot"][:200] + "...",
                # per-component scores -> the words / meaning / judge bars on the page
                "scores": {key: round(r[key], 4) for key in ("bm25", "dense", "hybrid") if key in r},
            }
            for r in rows
        ],
    }
