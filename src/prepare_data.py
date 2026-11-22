"""
Step 1: turn the Kaggle CSV into our corpus file.

Input : data/raw/wiki_movie_plots_deduped.csv  (Kaggle "Wikipedia Movie Plots")
Output: data/processed/corpus.jsonl  (one movie per line)

Each movie gets an id like "The Lion King (1994)".
This makes hand-labelling easy: in must_show.jsonl you just write the title and year.

Run:  python src/prepare_data.py --min-year 1980
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/wiki_movie_plots_deduped.csv")
OUT = Path("data/processed/corpus.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year", type=int, default=0, help="keep only movies from this year on")
    args = parser.parse_args()

    df = pd.read_csv(RAW)
    df = df.rename(columns={
        "Release Year": "year", "Title": "title", "Origin/Ethnicity": "origin",
        "Director": "director", "Cast": "cast", "Genre": "genre", "Plot": "plot",
    })
    df = df[df["year"] >= args.min_year]
    df = df.dropna(subset=["title", "plot"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    with OUT.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            doc_id = f"{row['title']} ({int(row['year'])})"
            # Same title + year can appear twice (e.g. remakes in different countries)
            if doc_id in seen:
                seen[doc_id] += 1
                doc_id = f"{doc_id} [{row['origin']}]"
            else:
                seen[doc_id] = 1
            f.write(json.dumps({
                "doc_id": doc_id,
                "title": row["title"],
                "year": int(row["year"]),
                "origin": row["origin"],
                "cast": "" if pd.isna(row["cast"]) else row["cast"],
                "plot": row["plot"],
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} movies to {OUT}")


if __name__ == "__main__":
    main()
