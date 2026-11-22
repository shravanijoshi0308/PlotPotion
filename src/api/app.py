"""
The web API.  GET /search?q=...&k=5&mode=rerank

mode = rerank (default: words + meaning + judge) | hybrid | bm25 | dense
lang = all (default) | indian | telugu | tamil | bollywood | hollywood | ...   (see GET /languages)

Run locally:  uvicorn src.api.app:app --reload --port 8001
Then open:    http://127.0.0.1:8001/          (the PlotPotion page)
Docs page:    http://127.0.0.1:8001/docs
"""
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from src.pipeline import LANGUAGES, Pipeline

app = FastAPI(title="PlotPotion", description="Brew a plot, reveal a movie")

pipe = Pipeline()
by_id = {d["doc_id"]: d for d in pipe.corpus}

INDEX = Path(__file__).resolve().parents[2] / "static" / "index.html"


@app.get("/")
def home():
    """The PlotPotion web page."""
    return FileResponse(INDEX)


@app.get("/languages")
def languages():
    """The choices for the language picker on the page."""
    return [{"key": key, "label": label, "count": pipe.lang_count[key]} for key, (label, _) in LANGUAGES.items()]


@app.get("/search")
def search(q: str, k: int = 5, mode: str = Query("rerank", pattern="^(rerank|hybrid|bm25|dense)$"),
           lang: str = "all"):
    if lang not in LANGUAGES:
        lang = "all"
    rows, timing = pipe.run(q, k=k, mode=mode, lang=lang)
    return {
        "query": q,
        "mode": mode,
        "lang": lang,
        "latency_ms": timing["total_ms"],
        "timing_ms": timing,
        "results": [
            {
                "doc_id": r["doc_id"],
                "title": by_id[r["doc_id"]]["title"],
                "year": by_id[r["doc_id"]]["year"],
                "origin": by_id[r["doc_id"]]["origin"],
                "plot_snippet": by_id[r["doc_id"]]["plot"][:200] + "...",
                # per-component scores -> the words / meaning / judge bars on the page
                "scores": {key: round(r[key], 4) for key in ("bm25", "dense", "hybrid", "rerank") if key in r},
            }
            for r in rows
        ],
    }
