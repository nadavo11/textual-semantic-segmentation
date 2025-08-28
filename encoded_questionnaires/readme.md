# Questionnaire Encoding Guide

This repository standardizes how we **encode psychological/cultural questionnaires**
(e.g. Triandis Ind/Col, VSM-2013 LTO, Moral Foundations, Growth Mindset, Tight/Loose, etc.)
for CCR (Cosine Similarity–based Cultural Representations).  

---

## 1. File Format

Every questionnaire is a **CSV file** with this structure:
```
question_id,dimension,question  
HI1,Horizontal Individualism,"I'd rather depend on myself than others."  
LTO13,Long-Term Orientation,"How important is it for you to save money for the future?"  
MFQ2,Fairness,"Whether or not some people were treated differently than others"  
GM1,Growth Mindset,"You have a certain amount of intelligence, and you can’t really do much to change it."  
TL4,Tight-Loose,"People in this country have a great deal of freedom in how they want to behave in most situations."  
```
- `question_id`: unique ID (scale abbreviation + item number)  
- `dimension`: subscale/dimension  
- `question`: verbatim text  

👉 Row order defines `sim_item_k` mapping in CCR.

---

## 2. Layout & Manifest

Each questionnaire CSV lives in `/questionnaires/`.  
The manifest (`questionnaire_manifest.json`) lists the file, scales, items, and reversed items.

Example:
````
"triandis_indcol": {  
  "file": "triandis_indcol.csv",  
  "scales": {  
    "Horizontal Individualism": {"item_ids": ["HI1","HI2","HI3"], "reverse": []},  
    "Vertical Individualism": {"item_ids": ["VI1","VI2"], "reverse": []}  
  }  
}
````
---

## 3. Encoding Process

### Command Line
````
python batch_encode_questionnaires.py \  
  --texts /path/to/narratives.csv \  
  --text-col clean_text \  
  --outdir encoded_outputs \  
  --device cuda  
````
- `--texts`: CSV with narratives  
- `--text-col`: text column name  
- `--outdir`: save directory  
- `--device`: cuda or cpu  

Outputs: one Parquet per questionnaire (`item_text`, `embedding`).

### Python Interactive
```
from ccr_helpers import ccr_wrapper  

ccr_df = ccr_wrapper(  
    data_file="/path/to/narratives.csv",  
    data_col="clean_text",  
    q_file="questionnaires/triandis_indcol.csv",  
    q_col="question"  
)
```
---

## 4. Working with Results

- Narratives get `sim_item_k` columns (cosine similarities).  
- Build scales with `narratives_helpers.py`:  
```
from narratives_helpers import make_scale_from_items  
ccr_df = make_scale_from_items(ccr_df, item_indices=[1,3,7], new_col="Fairness_score")  
```
- Reverse scoring = multiply by –1 or mark in manifest.  
- Aggregate to country-level means.

---

## 5. Practical Tips

- Mark reversed items in CSV (`reverse=true`).  
- Z-score items before averaging if needed.  
- Use GPU for speed; lower batch size on OOM.  
- Save Parquet instead of CSV.

---

## 6. Recommended Prompt for LLM Agent

```
in progress. have a good prompt? please contact and contribute your own!
```

---

## 7. Example Workflow

- **Triandis Ind/Col** → 4 scales (HI, VI, HC, VC)  
- **VSM-2013 LTO** → items Q13, Q14, Q19, Q22 → LTO index  
- **Moral Foundations** → subset Fairness & Authority  
- **Growth Mindset** → 3 fixed-mindset items (reverse)  
- **Tight/Loose** → 6 items  

---
