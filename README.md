# Keyword Clustering Toolkit

Two-script workflow for grouping keywords or AI Prompts into topical clusters using KMeans. Designed for SEO keyword analysis on GSC exports, Ahrefs/Semrush data, AI prompts or any CSV with a unique text column.

## How it works

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `kfinder.py` | Tests a range of *k* values and scores each one using silhouette analysis. Outputs the best *k* and saves elbow plots so you can visually confirm. |
| 2 | `kclusterer.py` | Takes the *k* you chose and runs the final clustering pass — assigns every keyword to a cluster and auto-labels each cluster. |

**Always run `kfinder.py` first**, then use the best *k* it recommends as input to `kclusterer.py`.

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd keyword-clustering
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will also download the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB). This is cached locally after the first download.

---

## Requirements

Create a `requirements.txt` in the repo root with:

```
numpy
pandas
matplotlib
scikit-learn
sentence-transformers
joblib
```

Python 3.9+ is recommended.

---

## Quick start

```bash
# Step 1: Find the best k
python kfinder.py my_gsc_export.csv

# Step 2: Cluster using the k value from step 1
python kclusterer.py my_gsc_export.csv --k 250
```

---

## Script 1: kfinder.py

### Usage

```bash
python kfinder.py <input_csv> [--output results.csv]
```

**Arguments:**

- `input_csv` (required) — path to your keyword CSV
- `--output` / `-o` (optional) — where to save results (default: `kmeans_silhouette_results.csv`)

### What you need to configure

Open `kfinder.py` and look at the **Config** section near the top. These are the values you may need to change:

| Variable | What it controls | When to change it |
|----------|-----------------|-------------------|
| `K_LIST` | Range of *k* values to test | Lower these if your dataset has fewer than ~600 unique keywords. All values must be less than the number of unique keywords. Adjust based on how many topics you think may be in your dataset as this can vary widely by site or due to filtering.|
| `COL_KEYWORD` | Column name containing the keywords | Change if your CSV uses a different header (e.g., `"Keyword"`, `"Query"`, `"Search Query"`) |
| `COL_URL` | Column name for the ranking URL (optional) | Set to the column name if your CSV includes URLs — adds topical signal from the URL slug. Set to `None` if not present. |
| `COL_INTENT` | Column name for search intent (optional) | Set to the column name if your data includes intent classification. Set to `None` if not present. |
| `INTENT_WEIGHT` | How much intent influences clustering (0–1) | Default `0.02` is deliberately low. Only relevant if `COL_INTENT` is set. |
| `SILHOUETTE_SAMPLE` | Sample size for silhouette scoring | Leave as `None` (exact) for datasets under ~10k rows. Set to e.g. `5000` for larger datasets to speed things up. |

### Output

- `kmeans_silhouette_results.csv` — table of k values with silhouette scores and inertia
- `kmeans_silhouette_results_elbow_plots.png` — visual elbow curves

### Reading the output

Pick the *k* with the **highest silhouette score**. The script prints this recommendation. Also look at the elbow plot — you want a *k* near the "bend" in the inertia curve. If the silhouette peak and the elbow agree, you have a strong signal. If they diverge, lean toward the silhouette recommendation but consider testing both in the clustering step.

---

## Script 2: kclusterer.py

### Usage

```bash
python kclusterer.py <input_csv> --k <number> [--output clustered.csv] [--limit 5000]
```

**Arguments:**

- `input_csv` (required) — path to your keyword CSV (same file you used with kfinder)
- `--k` (required) — number of clusters to create. Use the best k from kfinder.py.
- `--output` / `-o` (optional) — where to save the clustered CSV (default: `<input>_clustered.csv`)
- `--limit` (optional) — only cluster the top N rows by volume/clicks (descending). Useful for focusing on high-value keywords.

### What you need to configure

Same Config section approach as kfinder. The column mappings **must match between both scripts** — if you changed `COL_KEYWORD` in kfinder, change it here too.

| Variable | What it controls | When to change it |
|----------|-----------------|-------------------|
| `COL_KEYWORD` | Column name containing the keywords | Must match your CSV header. **Must be the same value you used in kfinder.py.** |
| `COL_URL` | Column name for the ranking URL (optional) | Same as kfinder — adds topical signal from URL slugs. |
| `COL_INTENT` | Column name for search intent (optional) | Same as kfinder. |
| `COL_VOLUME` | Column used for `--limit` sorting | Only matters if you use the `--limit` flag. |

### Output

The output CSV is your original data with two columns appended:

- `Cluster_ID` — integer cluster assignment (0 to k-1)
- `Cluster_Label` — auto-generated human-readable label derived from the most frequent meaningful words in that cluster

### How labeling works

Each cluster label is built by counting word frequency across all keywords in the cluster, filtering out stopwords, and taking the top 4 most common terms. For example, a cluster containing "best running shoes", "running shoe reviews", and "top running shoes 2026" would get a label like **"Running Shoes Reviews"**.

Note: Two clusters about similar topics may receive the same label. The `Cluster_ID` is always the unique identifier — use the label for human readability, not as a key.

---

## Common CSV formats

The scripts are tested with these export formats. Just match `COL_KEYWORD` to the right header:

| Source | Keyword column header | Notes |
|--------|----------------------|-------|
| Google Search Console | `Top queries` | Default. May also include `Top pages` (set as `COL_URL`). |
| Ahrefs | `Keyword` | Organic Keywords export. Often includes `URL` and `Volume`. |
| Semrush | `Keyword` | Position Tracking or Organic Research export. |
| Custom / merged | Varies | Match `COL_KEYWORD` to whatever your header is. |

---

## Troubleshooting

**"Required column not found"** — Your CSV header doesn't match `COL_KEYWORD`. Open your CSV, check the exact header name, and update the Config section.

**"All k values are >= n_samples"** — Your dataset is too small for the default `K_LIST`. Lower the values — e.g., `K_LIST = [10, 20, 30, 50, 75, 100]`.

**Slow on large datasets** — Set `SILHOUETTE_SAMPLE = 5000` (or similar) to switch from exact to approximate silhouette scoring. Also verify `N_JOBS = -1` to use all CPU cores.

**Segfaults on macOS** — Usually a threading conflict with numpy. The script already sets `OPENBLAS_NUM_THREADS=1` etc., but if issues persist, try running with `OMP_NUM_THREADS=1 python kfinder.py ...`.

---

## End-to-end example

```bash
# 1. Activate your environment
source .venv/bin/activate

# 2. Find the best k
python kfinder.py sacbee_gsc_export.csv
#    → prints: ✓ Best k by silhouette: k=250 (score=0.0832)
#    → saves:  kmeans_silhouette_results.csv + elbow plots

# 3. Cluster with that k
python kclusterer.py sacbee_gsc_export.csv --k 250
#    → saves:  sacbee_gsc_export_clustered.csv

# 4. Open the clustered CSV — sort/filter by Cluster_Label to explore topics
```

---

## Important: keep Config sections in sync

Both scripts have a Config section with column name mappings (`COL_KEYWORD`, `COL_URL`, `COL_INTENT`). **These must match between scripts** since both need to parse the same CSV. If you update one, update the other.
