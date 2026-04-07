#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword KMeans Clustering with Semantic Cluster Labels

Takes a CSV of keywords and a chosen k value (from kfinder.py),
clusters every keyword, and auto-labels each cluster.

Inputs : CSV with keywords + optional enrichment columns (URL, Intent)
Outputs: Same CSV with Cluster_ID and Cluster_Label columns appended

Usage:
    python kclusterer.py <input_csv> --k <number>
    python kclusterer.py my_gsc_export.csv --k 250
    python kclusterer.py my_gsc_export.csv --k 250 --output clustered.csv
    python kclusterer.py big_export.csv --k 300 --limit 5000

See README.md for full setup and configuration instructions.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

# --- Hard caps on threading to avoid segfaults ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# ── Config ────────────────────────────────────────────────────────────────────
# Column name mapping — adjust if your CSV headers differ.
# The keyword column is REQUIRED. URL and Intent are optional enrichments.
COL_KEYWORD = "Top queries"
COL_URL     = None               # e.g., "Top pages" or "URL" — set to None if not present
COL_INTENT  = None               # e.g., "Intent"              — set to None if not present
COL_VOLUME  = "Clicks"           # used only for --limit (top-N filtering)
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
def slug_from_url(url: str) -> str:
    """Extract a readable slug from a URL path."""
    try:
        path = urlparse(str(url)).path.rstrip("/")
        slug = path.split("/")[-1].lower()
        return re.sub(r"\.(html?|php|aspx?)$", "", slug).replace("-", " ").replace("_", " ")
    except Exception:
        return ""


def build_text_for_embedding(keyword: str, intent: str = "", url: str = "") -> str:
    """
    Combine keyword + optional signals into a single string
    for embedding. Richer text → better cluster separation.
    """
    parts = [keyword.strip()]

    if url and str(url).strip() not in ("", "nan"):
        slug = slug_from_url(str(url))
        if slug:
            parts.append(slug)

    if intent and str(intent).strip() not in ("", "nan"):
        parts.append(str(intent).strip())

    return " | ".join(parts)


def label_cluster(keywords: list[str], top_n_words: int = 4) -> str:
    """
    Derive a human-readable label for a cluster by finding the most frequent
    meaningful words across all keywords in the cluster.
    """
    stopwords = {
        "the","a","an","and","or","of","in","on","to","for","is","are",
        "how","what","best","top","most","vs","with","from","do","can",
        "your","you","my","our","its","be","at","by","as","this","that",
        "get","have","has","was","were","will","it","he","she","they","we",
    }
    word_counts: Counter = Counter()
    for kw in keywords:
        words = re.findall(r"[a-z]+", kw.lower())
        for w in words:
            if w not in stopwords and len(w) > 2:
                word_counts[w] += 1

    top_words = [w for w, _ in word_counts.most_common(top_n_words)]
    return " ".join(top_words).title() if top_words else "Cluster"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── CLI args ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Cluster keywords using KMeans and auto-label each cluster."
    )
    parser.add_argument("input_csv", help="Path to the input CSV (same file used with kfinder.py)")
    parser.add_argument(
        "--k", type=int, required=True,
        help="Number of clusters — use the best k from kfinder.py",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the output CSV (default: <input>_clustered.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help=f"Optional: only cluster the top N rows by {COL_VOLUME} (descending)",
    )
    args = parser.parse_args()

    input_path  = args.input_csv
    n_clusters  = args.k
    limit_top_n = args.limit

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_clustered{ext}"

    if not os.path.isfile(input_path):
        print(f"Error: file not found — {input_path}")
        sys.exit(1)

    # 1. Load
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]

    if COL_KEYWORD not in df.columns:
        raise ValueError(
            f"Required column '{COL_KEYWORD}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update COL_KEYWORD in the Config section of this script to match your CSV."
        )

    print(f"Loaded {len(df):,} rows from {input_path}")
    print(f"Columns: {list(df.columns)}")

    has_url    = COL_URL    and COL_URL    in df.columns
    has_intent = COL_INTENT and COL_INTENT in df.columns
    print(f"URL column:    {'✓ ' + COL_URL    if has_url    else '✗ not found (optional)'}")
    print(f"Intent column: {'✓ ' + COL_INTENT if has_intent else '✗ not found (optional)'}")

    # Optional: limit to top N by Volume
    if limit_top_n:
        if COL_VOLUME not in df.columns:
            print(f"Warning: --limit was set but '{COL_VOLUME}' column not found. Skipping limit.")
        else:
            original_count = len(df)
            df = df.sort_values(COL_VOLUME, ascending=False).head(limit_top_n).reset_index(drop=True)
            print(f"Filtered to top {limit_top_n:,} rows by {COL_VOLUME} (dropped {original_count - len(df):,})")

    # Validate k vs row count
    if n_clusters >= len(df):
        raise ValueError(
            f"k ({n_clusters}) must be less than the number of rows ({len(df):,}).\n"
            f"Choose a smaller k or use a larger dataset."
        )

    # 2. Build embedding text per row
    intent_col = df[COL_INTENT].astype(str) if has_intent else pd.Series([""] * len(df))
    url_col    = df[COL_URL].astype(str)    if has_url    else pd.Series([""] * len(df))

    df["_embed_text"] = [
        build_text_for_embedding(
            str(df.at[i, COL_KEYWORD]),
            intent_col.iloc[i],
            url_col.iloc[i],
        )
        for i in range(len(df))
    ]

    print(f"\nRows to cluster: {len(df):,}")
    print("Sample embed texts:")
    for t in df["_embed_text"].head(5):
        print("  ", t)

    # 3. Embed
    print("\nEncoding with SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["_embed_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # 4. KMeans
    print(f"\nRunning KMeans with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm="lloyd")
    df["Cluster_ID"] = km.fit_predict(embeddings)

    # 5. Generate a semantic label per cluster
    print("Generating cluster labels...")
    cluster_labels: dict[int, str] = {}
    for cid in sorted(df["Cluster_ID"].unique()):
        kws = df.loc[df["Cluster_ID"] == cid, COL_KEYWORD].tolist()
        cluster_labels[cid] = label_cluster(kws)

    df["Cluster_Label"] = df["Cluster_ID"].map(cluster_labels)

    # 6. Clean up temp column and save
    df.drop(columns=["_embed_text"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    # Summary
    print(f"\nTotal clusters: {df['Cluster_ID'].nunique():,}")
    print("\nCluster size summary (top 20 by size):")
    summary = (
        df.groupby(["Cluster_ID", "Cluster_Label"])
          .size()
          .reset_index(name="Count")
          .sort_values("Count", ascending=False)
          .head(20)
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
