"""
Score a search method on our questions.

Questions file (JSONL), one question per line:
  {"qid": "m01", "query": "...", "answer": "The Lion King (1994)", "group": "dense"}

group = which part SHOULD win: "bm25", "dense" or "rerank".

Prints Recall@10, Recall@50, MRR@10 overall and per group,
plus the questions where the right movie was not in the top 10 (read these! course rule 5).

Run:  python -m eval.evaluate --method bm25      (or dense, hybrid)
"""
import argparse
import json
import time
from collections import defaultdict

from src.retrievers.base import load_corpus


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_retriever(name):
    if name == "bm25":
        from src.retrievers.bm25 import BM25Retriever
        return BM25Retriever()
    if name == "dense":
        from src.retrievers.dense import DenseRetriever
        return DenseRetriever()
    if name == "hybrid":
        from src.retrievers.base import load_corpus
        from src.retrievers.bm25 import BM25Retriever
        from src.retrievers.dense import DenseRetriever
        from src.retrievers.hybrid import HybridRetriever
        corpus = load_corpus()
        return HybridRetriever(BM25Retriever(corpus), DenseRetriever(corpus))
    # Later: "hybrid_rerank"
    raise ValueError(f"Unknown method: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/must_show.jsonl")
    parser.add_argument("--method", default="bm25")
    args = parser.parse_args()

    corpus_ids = {d["doc_id"] for d in load_corpus()}
    queries = load_queries(args.queries)

    # Warn about answers that are not in our movie list (typo in title/year?)
    missing = [q for q in queries if q["answer"] not in corpus_ids]
    for q in missing:
        print(f"WARNING: answer not in corpus, skipped: {q['qid']} -> {q['answer']}")
    queries = [q for q in queries if q["answer"] in corpus_ids]

    retriever = get_retriever(args.method)

    per_group = defaultdict(lambda: {"n": 0, "r10": 0, "r50": 0, "mrr": 0.0})
    failures, times = [], []

    for q in queries:
        start = time.perf_counter()
        results = retriever.search(q["query"], k=50)
        times.append((time.perf_counter() - start) * 1000)

        ranked = [doc_id for doc_id, _ in results]
        rank = ranked.index(q["answer"]) + 1 if q["answer"] in ranked else None

        for g in ("all", q.get("group", "none")):
            s = per_group[g]
            s["n"] += 1
            s["r10"] += int(rank is not None and rank <= 10)
            s["r50"] += int(rank is not None)
            s["mrr"] += 1 / rank if rank and rank <= 10 else 0
        if rank is None or rank > 10:
            failures.append((q, ranked[:3]))

    print(f"\nMethod: {args.method}   Questions: {len(queries)}")
    print(f"Average time per query: {sum(times) / max(len(times), 1):.1f} ms\n")
    print(f"{'group':<8} {'n':>4} {'R@10':>6} {'R@50':>6} {'MRR@10':>7}")
    for g, s in sorted(per_group.items(), key=lambda x: x[0] != "all"):
        n = s["n"]
        print(f"{g:<8} {n:>4} {s['r10']/n:>6.2f} {s['r50']/n:>6.2f} {s['mrr']/n:>7.2f}")

    if failures:
        print("\nNot in top 10 (look at these and guess why):")
        for q, top3 in failures:
            print(f"- [{q['qid']}] {q['query']}\n    wanted: {q['answer']}\n    got:    {top3}")


if __name__ == "__main__":
    main()
