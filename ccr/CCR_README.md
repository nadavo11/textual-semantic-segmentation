# CCR Utilities — Usage Guide

This guide explains how to run **CCR (item-level cosine similarity between your texts and questionnaire items)** and how to build **scales** by averaging selected questionnaire items. It complements the functions you already have (`encode_column`, `item_level_ccr`, `ccr_wrapper`) and adds helpers to average columns into a new score.

---

## What is CCR here?
Given a dataset of texts (e.g., narratives) and a questionnaire (a list of items/questions), we:
1. Encode both with a SentenceTransformers model (SBERT).
2. Compute a cosine-similarity matrix (texts × items).
3. Append one column per item to the texts dataframe: `sim_item_1`, `sim_item_2`, …
4. (Optional) Combine specific items into **scales** using weighted or unweighted averages.

---

## File Format Requirements

### 1) Data file (your texts)
A CSV with at least one text column (you choose its name, e.g., `clean_text` or `text`).  
Optional, but recommended for downstream aggregation: a country column like `country_code`.

**Example (CSV):**
```csv
id,country_code,clean_text
1,US,"I believe people are generally fair..."
2,DE,"Society values long-term planning..."
```

### 2) Questionnaire file
A CSV with one row per item and a text column containing the item text (you choose its name, e.g., `question` or `item_text`). **The row order defines item indices** (1..N) used for `sim_item_k`.

**Example (CSV):**
```csv
scale,item_text
HorizInd,"I prefer making my own decisions even if they go against the group."
VertInd,"Competing to be the best motivates me."
VertCol,"Group success matters more than personal success."
```
If your column is named `item_text`, then `q_col="item_text"` in the API.

> ⚠️ The item order in the questionnaire file is critical because `sim_item_1` corresponds to the **first** row, etc. Keep a stable ordering that matches your scale definitions.

---

## Quickstart (GPU recommended)

```python
from your_ccr_module import ccr_wrapper  # uses encode_column -> item_level_ccr
ccr_df = ccr_wrapper(
    data_file="data_texts.csv", 
    data_col="clean_text",
    q_file="questionnaire.csv", 
    q_col="item_text",
    model="all-MiniLM-L6-v2"   # or any SentenceTransformers model
)
# ccr_df now contains: original columns + 'embedding' + sim_item_1..sim_item_N
```

**Tips**
- If you hit VRAM issues, reduce batch size in `encode_column` (default is 2048 in your code).
- Consider saving as Parquet to avoid CSV precision/size overhead:
  ```python
  ccr_df.to_parquet("ccr_results.parquet")
  ```

---

## Building Scales (averaging items)

Use the helpers added to `narratives_helpers.py`:

```python
from narratives_helpers import average_columns, make_scale_from_items

# 1) Unweighted average across specific CCR similarity items
ccr_df = make_scale_from_items(ccr_df, item_indices=[1, 3, 7], new_col="Horizontal_Individualism_score")

# 2) Weighted average (e.g., item 2 counts double)
ccr_df = make_scale_from_items(ccr_df, item_indices=[2, 4, 5], new_col="Authority_weighted",
                               weights=[2.0, 1.0, 1.0])

# 3) Average across arbitrary columns (not just sim_item_*)
ccr_df = average_columns(ccr_df, cols=["sim_item_10", "sim_item_11", "sim_item_15"],
                         new_col="Fairness_score")
```

- `average_columns(df, cols, new_col, weights=None, drop=False)`  
  - A general row-wise (weighted) mean across any columns.  
  - Masks NaNs automatically; if a row has all NaNs, result is NaN.  
  - Set `drop=True` to remove the source columns after averaging.

- `make_scale_from_items(df, item_indices, new_col, item_prefix="sim_item_", weights=None, drop=False)`  
  - Convenience wrapper for `sim_item_*` columns based on item indices.

> ✅ These helpers **do not** assume any questionnaire semantics; you decide which items belong to each scale based on the questionnaire manual you’re using (e.g., Triandis Ind/Col, VSM-2013 LTO, etc.).

---

## End-to-End Example

```python
# 1) Run CCR
ccr_df = ccr_wrapper(
    data_file="data_texts.csv",
    data_col="clean_text",
    q_file="triandis_items.csv",
    q_col="item_text",
    model="all-MiniLM-L6-v2"
)

# 2) Define scale mappings (example indices; check your questionnaire manual!)
Horizontal_Individualism_items = [1, 4, 7, 9]
Vertical_Individualism_items   = [2, 5, 8, 10]
Horizontal_Collectivism_items  = [3, 6, 11, 12]
Vertical_Collectivism_items    = [13, 14, 15, 16]

# 3) Build scores
for name, idxs in {
    "Horizontal Individualism_score": Horizontal_Individualism_items,
    "Vertical Individualism_score":   Vertical_Individualism_items,
    "Horizontal Collectivism_score":  Horizontal_Collectivism_items,
    "Vertical Collectivism_score":    Vertical_Collectivism_items,
}.items():
    ccr_df = make_scale_from_items(ccr_df, item_indices=idxs, new_col=name)

# 4) Country aggregation (mean CCR scale per country)
ccr_df["country_code"] = ccr_df["country_code"].str.upper()
country_means = ccr_df.groupby("country_code")[
    ["Horizontal Individualism_score",
     "Vertical Individualism_score",
     "Horizontal Collectivism_score",
     "Vertical Collectivism_score"]
].mean().reset_index()
```

---

## Quality & Sanity Checks

- **Questionnaire item ordering**: verify that your CSV order matches the manual; otherwise `sim_item_k` won’t align with the intended item.  
- **Encoding drift**: re-encoding with a different model or version changes all similarities; record the model string used.  
- **Scale reliability**: consider reporting Cronbach’s alpha / McDonald’s ω for the items in a scale before averaging.  
- **Country coverage**: after building scales, check how many texts per country pass filters before correlating with external datasets.

---

## Performance Notes

- GPU encoding is much faster; use `device="cuda"` where possible.
- Lower `batch_size` in `encode_column` if you see CUDA OOM.
- Persist the embeddings if you’ll reuse the same texts/items frequently.
- Use `float16` where applicable to save memory (depends on model support).

---

## Integration with Downstream Analyses

- The resulting scale columns can be fed directly into:
  - `prevalence_vs_metrics(...)` — if you treat “prevalence” as % of fully-formed narratives.
  - `countrywise_regression(...)` — if you want country-aggregated CCR scale means vs any external metric (Hofstede, WVS, etc.).

---

## Reproducibility

- Pin the exact SentenceTransformers model:
  - Example: `model="sentence-transformers/all-MiniLM-L6-v2"`
- Record versions of `sentence-transformers`, `torch`, and GPU details.
- Save a manifest (items order and scale mappings).
