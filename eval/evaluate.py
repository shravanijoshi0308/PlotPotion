"""
Score the search methods on our questions.

Questions file (JSONL), one question per line:
  {"qid": "m01", "query": "...", "answer": "The Lion King (1994)", "group": "dense"}
  "answer" can also be a list when a film has two names in the data:
  {"answer": ["Harry Potter and the Sorcerer's Stone (2001)", "Harry Potter and the Philosopher's Stone (2001)"]}

group = which part SHOULD win: "bm25", "dense" or "rerank".

Prints R@1, R@10, R@50, MRR@10 overall and per group, the average time per question,
and the questions where the right movie was not in the top 10 (read these! course rule 5).

Run:  python -m eval.evaluate                    (all four methods)
      python -m eval.evaluate --method hybrid    (just one)
"""
import argparse
import json
import time
from collections import defaultdict

from src.pipeline import Pipeline, PipelineRetriever

METHODS = ["bm25", "dense", "hybrid", "rerank"]


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever, queries, show_failures=True):
    per_group = defaultdict(lambda: {"n": 0, "r1": 0, "r10": 0, "r50": 0, "mrr": 0.0})
    failures, times = [], []
    for q in queries:
        start = time.perf_counter()
        ranked = [doc_id for doc_id, _ in retriever.search(q["query"], k=50)]
        times.append((time.perf_counter() - start) * 1000)
        rank = next((n for n, d in enumerate(ranked, 1) if d in q["_ok"]), None)
        for g in ("all", q.get("group", "none")):
            s = per_group[g]
            s["n"] += 1
            s["r1"] += int(rank == 1)
            s["r10"] += int(rank is not None and rank <= 10)
            s["r50"] += int(rank is not None)
            s["mrr"] += 1 / rank if rank and rank <= 10 else 0
        if rank is None or rank > 10:
            failures.append((q, rank, ranked[:3]))

    times_sorted = sorted(times[1:] or times)          # skip the first (model warm-up)
    print(f"\n=== {retriever.name} ===   questions: {len(queries)}   "
          f"median time: {times_sorted[len(times_sorted) // 2]:.0f} ms")
    print(f"{'group':<8} {'n':>4} {'R@1':>6} {'R@10':>6} {'R@50':>6} {'MRR@10':>7}")
    for g, s in sorted(per_group.items(), key=lambda x: x[0] != "all"):
        n = s["n"]
        print(f"{g:<8} {n:>4} {s['r1']/n:>6.2f} {s['r10']/n:>6.2f} {s['r50']/n:>6.2f} {s['mrr']/n:>7.2f}")
    if show_failures and failures:
        print("Not in top 10 (look at these and guess why):")
        for q, rank, top3 in failures:
            where = f"rank {rank}" if rank else "not in top 50"
            print(f"- [{q['qid']}] {q['query']}\n    wanted: {q['answer']} ({where})\n    got:    {top3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/must_show.jsonl")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    args = parser.parse_args()

    pipe = Pipeline(use_reranker=args.method in ("rerank", "all"))
    # every name a film goes by (incl. removed duplicates) -> its doc_id
    alias = {}
    for d in pipe.corpus:
        alias[d["doc_id"]] = d["doc_id"]
        for a in d.get("aliases", []):
            alias[a] = d["doc_id"]

    queries = []
    for q in load_queries(args.queries):
        answers = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        q["_ok"] = {alias[a] for a in answers if a in alias}
        if not q["_ok"]:
            print(f"WARNING: answer not in corpus, skipped: {q['qid']} -> {q['answer']}")
            continue
        queries.append(q)

    for m in (METHODS if args.method == "all" else [args.method]):
        evaluate(PipelineRetriever(pipe, m), queries)


if __name__ == "__main__":
    main()
