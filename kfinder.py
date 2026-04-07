#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMeans k-finder via silhouette score — optimized for speed.

GSC data: Embeds keywords as primary signal.

SEMRush data: Embeds keywords using Keyword + URL slug as primary signal,
with Intent mixed in at reduced weight.

Speed levers:
  - Parallel k evaluation via joblib
  - Approximate silhouette (sample_size) to avoid O(n²) pairwise distances
  - Reduced n_init (fine for a search pass, not a final model)

Usage:
    python kfinder.py <input_csv>
    python kfinder.py my_gsc_export.csv
    python kfinder.py my_gsc_export.csv --output results.csv

See README.md for full setup and configuration instructions.
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for SSH/CI
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
# K values to evaluate — must all be < number of unique keywords.
# If your dataset has fewer than ~600 unique keywords, lower these values.
K_LIST = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

# Column name mapping — adjust if your CSV headers differ.
# The keyword column is REQUIRED. URL and Intent are optional enrichments, 
# (e.g. for SEMRush or Ahrefs exported data).
COL_KEYWORD = "Top queries"
COL_URL     = None               # e.g., "Top pages"  — set to None if not present
COL_INTENT  = None               # e.g., "Intent"     — set to None if not present

INTENT_WEIGHT  = 0.02            # how much intent influences the embedding (0–1)

N_JOBS         = -1              # -1 = use all CPU cores
N_INIT         = 3               # 3 is fine for a k-search pass (vs 10 for a final model)
SILHOUETTE_SAMPLE = None         # None = exact; set to e.g. 5000 for large datasets
# ─────────────────────────────────────────────────────────────────────────────

# Threading caps — prevents segfaults in some environments
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["OMP_NUM_THREADS"]      = "1"


def slug_from_url(url: str) -> str:
    """Extract a readable slug from a URL path."""
    try:
        path = urlparse(str(url)).path.rstrip("/")
        slug = path.split("/")[-1].lower()
        return re.sub(r"\.(html?|php|aspx?)$", "", slug).replace("-", " ").replace("_", " ")
    except Exception:
        return ""


def build_primary_text(row: pd.Series) -> str:
    """
    Build the text fed into the embedding model.
    Keyword (always) + URL slug (if available and non-empty).
    """
    parts = [str(row[COL_KEYWORD]).strip()]

    if COL_URL and COL_URL in row.index:
        slug = slug_from_url(str(row[COL_URL]))
        if slug:
            parts.append(slug)

    return " | ".join(parts)


def fit_and_score(k: int, embeddings: np.ndarray, sample_size) -> dict:
    """Fit KMeans for a single k and return its scores. Runs in parallel."""
    n_samples = embeddings.shape[0]
    if k >= n_samples:
        print(f"  Skipping k={k}: must be < n_samples ({n_samples})")
        return None

    km = KMeans(n_clusters=k, random_state=42, n_init=N_INIT, algorithm="lloyd")
    labels = km.fit_predict(embeddings)

    sil = silhouette_score(
        embeddings, labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=42,
    )
    print(f"  k={k:>5}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")
    return {"k": k, "silhouette_score": sil, "inertia": km.inertia_}


def main():
    # ── CLI args ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Find the optimal k for KMeans keyword clustering via silhouette score."
    )
    parser.add_argument("input_csv", help="Path to the input CSV (e.g., GSC export)")
    parser.add_argument(
        "--output", "-o",
        default="kmeans_silhouette_results.csv",
        help="Path for the results CSV (default: kmeans_silhouette_results.csv)",
    )
    args = parser.parse_args()

    input_path  = args.input_csv
    results_out = args.output

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

    # 2. Build deduplicated embedding texts
    df["_primary"] = df.apply(build_primary_text, axis=1)
    unique_primary = df["_primary"].drop_duplicates().tolist()
    print(f"\nUnique embedding inputs: {len(unique_primary):,}")

    # Auto-filter K_LIST to valid range
    max_k = len(unique_primary) - 1
    valid_ks = [k for k in K_LIST if k < max_k]
    if len(valid_ks) < len(K_LIST):
        removed = [k for k in K_LIST if k >= max_k]
        print(f"Removed k values >= {max_k}: {removed}")
    if not valid_ks:
        raise ValueError(
            f"All k values in K_LIST are >= the number of unique keywords ({max_k + 1}).\n"
            f"Lower your K_LIST values in the Config section of this script."
        )

    # 3. Encode
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nEncoding keyword embeddings...")
    primary_emb = np.array(model.encode(unique_primary, show_progress_bar=True), dtype="float32")
    embeddings = primary_emb

    if has_intent and INTENT_WEIGHT > 0:
        primary_to_intent = (
            df.drop_duplicates(subset="_primary")
              .set_index("_primary")[COL_INTENT]
              .astype(str).str.strip()
        )
        unique_intents = [primary_to_intent.get(p, "") for p in unique_primary]
        print("Encoding intent embeddings...")
        intent_emb = np.array(model.encode(unique_intents, show_progress_bar=True), dtype="float32")
        embeddings = (1.0 - INTENT_WEIGHT) * embeddings + INTENT_WEIGHT * intent_emb
        print(f"Blended: keyword={1-INTENT_WEIGHT:.0%}, intent={INTENT_WEIGHT:.0%}")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    # 4. Run all k values in parallel
    n_cores = os.cpu_count()
    effective_jobs = n_cores if N_JOBS == -1 else N_JOBS
    print(f"\nRunning {len(valid_ks)} k values in parallel across {effective_jobs} cores...")
    print(f"  n_init={N_INIT}, silhouette sample_size={SILHOUETTE_SAMPLE or 'exact'}\n")

    raw_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fit_and_score)(k, embeddings, SILHOUETTE_SAMPLE)
        for k in valid_ks
    )

    # 5. Save + report
    results = [r for r in raw_results if r is not None]
    res_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    res_df.to_csv(results_out, index=False)

    print("\n── Silhouette Results ──")
    print(res_df.to_string(index=False))

    best = res_df.loc[res_df["silhouette_score"].idxmax()]
    print(f"\n✓ Best k by silhouette: k={int(best['k'])}  (score={best['silhouette_score']:.4f})")
    print(f"Saved to {results_out}")

    # 6. Plot elbow curves
    ks   = res_df["k"].tolist()
    sils = res_df["silhouette_score"].tolist()
    iner = res_df["inertia"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KMeans k Selection", fontsize=14, fontweight="bold")

    ax1.plot(ks, sils, marker="o", linewidth=2, color="#2196F3")
    best_k = int(best["k"])
    best_s = best["silhouette_score"]
    ax1.axvline(best_k, color="#F44336", linestyle="--", linewidth=1.2, label=f"Best k={best_k}")
    ax1.annotate(f"k={best_k}\n{best_s:.4f}",
                 xy=(best_k, best_s),
                 xytext=(10, -20), textcoords="offset points",
                 color="#F44336", fontsize=9)
    ax1.set_title("Silhouette Score (higher = better)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, iner, marker="o", linewidth=2, color="#4CAF50")
    ax2.set_title("Inertia / Elbow Curve (look for the bend)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_out.replace(".csv", "_elbow_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")


if __name__ == "__main__":
    main()