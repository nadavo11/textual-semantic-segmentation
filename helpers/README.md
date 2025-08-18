# Narratives Helpers

This module provides **helper functions for country-wise comparisons** in the Narratives project.  
It focuses on **narrative prevalence** (percentage of texts with all 5 narrative elements) and its relationship to other country-level variables.

---

## Functions

### 1. `plot_narrative_vs_trust`
Specialized helper for comparing **narrative prevalence vs trust-related variables**.  
- Input: `df_narratives`, `trust_df`
- Computes prevalence of narratives per country.
- Regresses trust metrics (e.g. `trust_clean`, `confidence_major_companies`, `trust_family`) on prevalence.
- Returns a regression results table and (optionally) plots scatter + linear fit with annotated countries.

```python
reg_trust = plot_narrative_vs_trust(df, trust_df, min_samples=80)
```

---

### 2. `prevalence_vs_metrics`
Generalized function for **prevalence vs arbitrary metrics** (any variables in another dataframe).  
- Input: `df_narratives`, `right_df`, and a list of metric columns.
- Returns both the regression results table and the prevalence dataframe (so you can reuse prevalence later).
- Plots optional.

```python
reg_table, prev_df = prevalence_vs_metrics(
    df_narratives=df,
    right_df=trust_df,
    metrics=["trust_clean", "confidence_major_companies"],
    min_samples=80
)
```

---

### 3. `countrywise_regression`
General helper for **two numeric columns from two different dataframes**, aggregated by country.  
- Flexible: specify aggregation (`mean`, `median`, etc.).
- Returns regression stats and the merged per-country dataframe.
- Useful for country-level comparisons where you already have numeric values per row.

```python
reg_xy, merged_xy = countrywise_regression(
    left_df=ccr_df_fixed, left_country_col="country_code", left_value_col="Horizontal Individualism_score",
    right_df=hofstede_df, right_country_col="alpha_2_code", right_value_col="LTO",
    min_samples_left=80
)
```

---

## Regression Output Format
All functions produce regression tables with:

- `x_var`, `y_var` : names of variables compared
- `N` : number of countries
- `slope`, `intercept` : regression line parameters
- `r`, `r2` : correlation and R²
- `p` : p-value
- `stderr` : standard error of slope
- `x_mean`, `y_mean`, `x_std`, `y_std`

---

## Plot Features
- One scatter plot per metric.
- Linear fit with dashed line.
- Country codes annotated.
- Small stats box with regression summary.

---

## Installation
Just place `narratives_helpers.py` in your project and import:

```python
from narratives_helpers import (
    plot_narrative_vs_trust,
    prevalence_vs_metrics,
    countrywise_regression
)
```

---
