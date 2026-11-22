"""
Step 1: turn the Kaggle CSV into our corpus file.

Input : data/raw/wiki_movie_plots_deduped.csv  (Kaggle "Wikipedia Movie Plots")
Output: data/processed/corpus.jsonl  (one movie per line)

Cleaning:
  - keep movies from --min-year on
  - remove duplicates: the same film listed twice (e.g. a Tamil and a Telugu entry,
    or a "(Dubbed version)") with the same title+year or the same plot.
    The removed names are kept in "aliases", so a question that names either one still counts.

Each movie gets an id like "The Lion King (1994)".

Run:  python src/prepare_data.py --min-year 1980
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/wiki_movie_plots_deduped.csv")
OUT = Path("data/processed/corpus.jsonl")


def norm_title(t: str) -> str:
    t = re.sub(r"\((dubbed|dubbed version|film)\)", "", t, flags=re.I)
    return re.sub(r"\W+", " ", t).strip().lower()


def plot_key(p: str) -> str:
    return re.sub(r"\W+", " ", p.lower()).strip()[:300]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year", type=int, default=0, help="keep only movies from this year on")
    args = parser.parse_args()

    df = pd.read_csv(RAW).rename(columns={
        "Release Year": "year", "Title": "title", "Origin/Ethnicity": "origin",
        "Director": "director", "Cast": "cast", "Genre": "genre", "Plot": "plot",
    })
    df = df[df["year"] >= args.min_year].dropna(subset=["title", "plot"])

    docs, by_title, by_plot, used_ids = [], {}, {}, set()
    removed = 0
    for _, row in df.iterrows():
        year = int(row["year"])
        title = str(row["title"]).strip()
        doc_id = f"{title} ({year})"
        tkey, pkey = (norm_title(title), year), plot_key(str(row["plot"]))

        keep = by_title.get(tkey)
        if keep is None:
            keep = by_plot.get(pkey)
        if keep is not None:                      # duplicate -> remember its name, skip it
            if "dubbed" in keep["doc_id"].lower() and "dubbed" not in doc_id.lower():
                # prefer the original over a "(Dubbed version)": swap names
                used_ids.discard(keep["doc_id"]); keep["aliases"].append(keep["doc_id"])
                keep["doc_id"], keep["title"], keep["origin"] = doc_id, title, row["origin"]
                used_ids.add(doc_id)
            elif doc_id != keep["doc_id"] and doc_id not in keep["aliases"]:
                keep["aliases"].append(doc_id)
            removed += 1
            continue

        if doc_id in used_ids:                    # same name, genuinely different film
            doc_id = f"{doc_id} [{row['origin']}]"
        used_ids.add(doc_id)
        doc = {"doc_id": doc_id, "title": title, "year": year, "origin": row["origin"],
               "cast": "" if pd.isna(row["cast"]) else row["cast"],
               "plot": str(row["plot"]), "aliases": []}
        docs.append(doc)
        by_title[tkey], by_plot[pkey] = doc, doc

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} movies to {OUT}  (removed {removed} duplicates)")


if __name__ == "__main__":
    main()
