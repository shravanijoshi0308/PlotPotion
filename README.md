# PlotPotion 🎬🧪

*Brew a plot, reveal a movie*

**What was that movie?** Describe a movie you remember, get its title.
Hybrid search: BM25 (words) + dense embeddings (meaning) + cross-encoder reranker (judge).

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

1. Download **Wikipedia Movie Plots** from Kaggle:
   https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
2. Put `wiki_movie_plots_deduped.csv` into `data/raw/`.
3. Build the corpus:

```bash
python src/prepare_data.py --min-year 1980
```

## Run

```bash
# Try BM25 on one query
python -m src.retrievers.bm25

# Score BM25 on our questions
python -m eval.evaluate --queries eval/must_show.jsonl --method bm25

# Start the API, then open http://127.0.0.1:8000/docs
uvicorn src.api.app:app --reload
```

## Folders

| Folder | What's inside | Role |
|---|---|---|
| `src/prepare_data.py` | CSV → `corpus.jsonl` | B (Data) |
| `src/retrievers/` | `bm25.py` now; later `dense.py`, `fusion.py`, `rerank.py` | A + B |
| `src/api/app.py` | FastAPI `GET /search?q=` | A (Framework) |
| `eval/` | `must_show.jsonl` questions + `evaluate.py` | C (Evaluation) |

## Writing questions (`eval/must_show.jsonl`)

One line per question:

```json
{"qid": "m04", "query": "your question", "answer": "Title (Year)", "group": "bm25"}
```

- `answer` must match the movie's `doc_id` in `corpus.jsonl` exactly: `Title (Year)`.
  The eval script warns you if it can't find it.
- `group` = which part should win: `bm25`, `dense` or `rerank`.
- Write from memory, by hand. No model labelling.

## Plan

- [x] Repo structure, common retriever interface
- [x] BM25 baseline + eval script + API
- [ ] ToT queries aligned with corpus (`queries.jsonl`, `qrels.jsonl`)
- [ ] 30 hand-written must-show questions (10 per group)
- [ ] Dense retrieval (`all-MiniLM-L6-v2`)
- [ ] Fusion (RRF)
- [ ] Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] Per-component scores + latency in API
- [ ] Deploy to Hugging Face Spaces
- [ ] Error analysis

Data: Wikipedia Movie Plots (Kaggle), CC BY-NC-SA. Used for a university project.
