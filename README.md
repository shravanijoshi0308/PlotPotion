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

## The search chain

```
question ─┬─> BM25 (words, stemmed) ──────┐
          │                               ├─> RRF fusion ─> top 30 ─> cross-encoder judge ─> top 5
          └─> meaning search (plot chunks)┘
```

| Step | File | What it does |
|---|---|---|
| Clean data | `src/prepare_data.py` | 1980+ movies, removes ~740 duplicate entries (keeps their names as aliases) |
| Words | `src/retrievers/bm25.py` | BM25 with English stemming (village = villagers) |
| Meaning | `src/retrievers/dense.py` | all-MiniLM-L6-v2 over ~200-word plot chunks; a movie's score = its best chunk |
| Mix | `src/pipeline.py` | Reciprocal Rank Fusion of the top 100 from each |
| Judge | `src/retrievers/rerank.py` | cross-encoder/ms-marco-MiniLM-L-6-v2 re-reads the best chunk of the top 30 |
| API | `src/api/app.py` | `GET /search?q=...&mode=rerank` with per-component scores and timing |
| Page | `static/index.html` | the PlotPotion potion UI |
| Eval | `eval/evaluate.py` | R@1, R@10, R@50, MRR@10 per method and per group |

## Run (in this order)

```bash
python src/prepare_data.py --min-year 1980     # 1. clean data
python -m src.retrievers.dense                 # 2. embed plot chunks (one-time, ~10 min)
python -m src.pipeline "your question"         # 3. try all 4 methods on one question
python -m eval.evaluate                        # 4. scores for bm25 / dense / hybrid / rerank
uvicorn src.api.app:app --reload --port 8001   # 5. open http://127.0.0.1:8001/
```

## Writing questions (`eval/must_show.jsonl`)

One line per question:

```json
{"qid": "m05", "query": "your question", "answer": "Title (Year)", "group": "dense"}
```

- `answer` must match a `doc_id` (or alias) in `corpus.jsonl`: `Title (Year)`. A list of names is allowed.
- `group` = which part should win: `bm25`, `dense` or `rerank`.
- Write from memory, by hand. No model labelling.

## Plan

- [x] Repo structure, BM25 baseline, eval script, API, UI
- [x] Duplicate removal, stemming
- [x] Dense retrieval over chunks, RRF fusion, cross-encoder reranker
- [x] Per-component scores + latency in the API
- [ ] ToT queries aligned with corpus (`queries.jsonl`, `qrels.jsonl`)
- [ ] 30 hand-written must-show questions (10 per group)
- [ ] Deploy to Hugging Face Spaces
- [ ] Error analysis

Data: Wikipedia Movie Plots (Kaggle), CC BY-SA 4.0. Used for a university project.
