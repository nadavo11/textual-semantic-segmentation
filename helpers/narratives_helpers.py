
"""
narratives_helpers.py

Helper utilities for country-wise comparisons and simple regressions/plots
for the Narratives project.

Rules:
- Uses matplotlib only (no seaborn).
- One plot per metric; plotting optional.
- No explicit color choices.
"""

from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# 1) EXACT function as requested: plot_narrative_vs_trust
# ---------------------------------------------------------------------------
def plot_narrative_vs_trust(
    df_narratives: pd.DataFrame,
    trust_df: pd.DataFrame,
    *,
    min_samples: int = 100,
    trust_cols: Iterable[str] = ("trust_clean", "confidence_major_companies", "trust_family"),
    annotate_countries: bool = True,
    show_plots: bool = True
) -> pd.DataFrame:
    """
    Compute narrative prevalence per country and regress each trust metric on it.
    Returns a regression table; optionally plots scatter + linear fit for each metric.
    This is kept identical to the version provided in chat (except for this docstring).
    """
    # ---- inputs & columns ----
    elements = ["causal_sequence", "characters", "internal_states", "plot_structure", "normative_point"]
    value_cols = [f"{el}_value" for el in elements]
    needed = ["country_code"] + value_cols
    df = df_narratives.dropna(subset=needed).copy()
    df["country_code"] = df["country_code"].str.upper()

    # ---- sample-size filter ----
    eligible = df["country_code"].value_counts()
    eligible = eligible[eligible >= min_samples].index
    df = df[df["country_code"].isin(eligible)]

    # ---- narrative prevalence (% with all 5) per country ----
    df["has_all"] = (df[value_cols].sum(axis=1) == 5)
    totals = df["country_code"].value_counts()
    with_all = df[df["has_all"]]["country_code"].value_counts()
    prevalence = (with_all / totals * 100).reindex(totals.index).fillna(0.0)
    prevalence_df = prevalence.reset_index()
    prevalence_df.columns = ["country_code", "narrative_prevalence"]
    prevalence_df["country_code"] = prevalence_df["country_code"].str.upper()

    # ---- trust merge (expects ISO alpha-2 in 'country' or fix as needed) ----
    trust_df = trust_df.copy()
    if "country" in trust_df.columns:
        trust_df["country"] = trust_df["country"].str.upper()
        merged = prevalence_df.merge(trust_df, left_on="country_code", right_on="country", how="inner")
    else:
        merged = prevalence_df.merge(trust_df, on="country_code", how="inner")

    # filter to available trust columns
    trust_cols = [c for c in trust_cols if c in merged.columns]
    merged = merged.dropna(subset=trust_cols)

    if merged.empty or not trust_cols:
        # return empty regression schema
        return pd.DataFrame(columns=[
            "x_var","y_var","N","slope","intercept","r","r2","p","stderr",
            "x_mean","y_mean","x_std","y_std"
        ])

    reg_rows = []
    x = merged["narrative_prevalence"].to_numpy()

    for col in trust_cols:
        y = merged[col].to_numpy()

        # regression
        lr = linregress(x, y)

        reg_rows.append({
            "x_var"   : "narrative_prevalence",
            "y_var"   : col,
            "N"       : int(len(merged)),
            "slope"   : lr.slope,
            "intercept": lr.intercept,
            "r"       : lr.rvalue,
            "r2"      : lr.rvalue**2,
            "p"       : lr.pvalue,
            "stderr"  : lr.stderr,
            "x_mean"  : float(np.mean(x)),
            "y_mean"  : float(np.mean(y)),
            "x_std"   : float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "y_std"   : float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        })

        if show_plots:
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x, y)

            # fit line
            xline = np.linspace(x.min(), x.max(), 200)
            yline = lr.slope * xline + lr.intercept
            plt.plot(xline, yline, linestyle='--', label='Linear Fit')

            if annotate_countries and "country_code" in merged.columns:
                for _, r in merged.iterrows():
                    plt.text(r["narrative_prevalence"] + 0.4, r[col], r["country_code"], fontsize=8)

            plt.title(f"Narrative Prevalence vs {col}")
            plt.xlabel("Narrative Prevalence (%)")
            plt.ylabel(f"{col}")
            # small stats box
            stats_text = (
                f"N={len(merged)}\n"
                f"slope={lr.slope:.3f}\n"
                f"intercept={lr.intercept:.3f}\n"
                f"r={lr.rvalue:.3f} (R²={lr.rvalue**2:.3f})\n"
                f"p={lr.pvalue:.3g}"
            )
            plt.gcf().text(0.02, 0.98, stats_text, va='top', ha='left', fontsize=9,
                           bbox=dict(boxstyle="round", alpha=0.15, lw=0.5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(reg_rows)


# ---------------------------------------------------------------------------
# 2) Generalized: prevalence vs arbitrary metrics (country-wise)
# ---------------------------------------------------------------------------
def prevalence_vs_metrics(
    df_narratives: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    metrics: Iterable[str],
    left_country_col: str = "country_code",
    right_country_col: Optional[str] = "country",   # if None, assumes 'country_code' in right_df
    element_value_suffix: str = "_value",
    elements: Iterable[str] = ("causal_sequence", "characters", "internal_states", "plot_structure", "normative_point"),
    require_all: bool = True,          # if False, uses min_elements
    min_elements: int = 5,             # threshold if require_all=False
    min_samples: int = 100,
    annotate_countries: bool = True,
    show_plots: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute narrative prevalence per country on the left (df_narratives),
    then regress each metric in right_df on that prevalence.

    Returns (regression_table, prevalence_df).

    Parameters
    ----------
    - metrics: columns in right_df to regress on prevalence.
    - right_country_col: if None, uses 'country_code' in right_df.
    - require_all: if True, prevalence = % texts that have ALL elements present (==len(elements)).
                   if False, prevalence = % texts with at least min_elements.
    """
    df = df_narratives.copy()
    df[left_country_col] = df[left_country_col].str.upper()

    value_cols = [f"{e}{element_value_suffix}" for e in elements]
    needed = [left_country_col] + value_cols
    df = df.dropna(subset=needed)

    # sample size filter on left
    eligible = df[left_country_col].value_counts()
    eligible = eligible[eligible >= min_samples].index
    df = df[df[left_country_col].isin(eligible)]

    # prevalence definition
    counts_present = df[value_cols].sum(axis=1)
    if require_all:
        has_pattern = (counts_present == len(value_cols))
    else:
        has_pattern = (counts_present >= min_elements)

    df = df.assign(__has_pattern=has_pattern)
    totals = df[left_country_col].value_counts()
    with_pattern = df[df["__has_pattern"]][left_country_col].value_counts()
    prevalence = (with_pattern / totals * 100).reindex(totals.index).fillna(0.0)

    prevalence_df = prevalence.reset_index()
    prevalence_df.columns = [left_country_col, "narrative_prevalence"]
    prevalence_df[left_country_col] = prevalence_df[left_country_col].str.upper()

    # right merge
    rdf = right_df.copy()
    if right_country_col is None:
        right_country_col = "country_code" if "country_code" in rdf.columns else "country"
    if right_country_col not in rdf.columns:
        raise KeyError(f"right_country_col='{right_country_col}' not found in right_df")

    rdf[right_country_col] = rdf[right_country_col].str.upper()
    merged = prevalence_df.merge(rdf, left_on=left_country_col, right_on=right_country_col, how="inner")

    # choose available metrics
    metrics = [m for m in metrics if m in merged.columns]
    merged = merged.dropna(subset=metrics)

    if merged.empty or not metrics:
        return (
            pd.DataFrame(columns=[
                "x_var","y_var","N","slope","intercept","r","r2","p","stderr",
                "x_mean","y_mean","x_std","y_std"
            ]),
            prevalence_df
        )

    reg_rows = []
    x = merged["narrative_prevalence"].to_numpy()

    for col in metrics:
        y = merged[col].to_numpy()
        lr = linregress(x, y)
        reg_rows.append({
            "x_var"   : "narrative_prevalence",
            "y_var"   : col,
            "N"       : int(len(merged)),
            "slope"   : lr.slope,
            "intercept": lr.intercept,
            "r"       : lr.rvalue,
            "r2"      : lr.rvalue**2,
            "p"       : lr.pvalue,
            "stderr"  : lr.stderr,
            "x_mean"  : float(np.mean(x)),
            "y_mean"  : float(np.mean(y)),
            "x_std"   : float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "y_std"   : float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        })

        if show_plots:
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x, y)
            xline = np.linspace(x.min(), x.max(), 200)
            yline = lr.slope * xline + lr.intercept
            plt.plot(xline, yline, linestyle='--', label='Linear Fit')

            if annotate_countries and left_country_col in merged.columns:
                for _, r in merged.iterrows():
                    plt.text(r["narrative_prevalence"] + 0.4, r[col], r[left_country_col], fontsize=8)

            plt.title(f"Narrative Prevalence vs {col}")
            plt.xlabel("Narrative Prevalence (%)")
            plt.ylabel(f"{col}")
            stats_text = (
                f"N={len(merged)}\n"
                f"slope={lr.slope:.3f}\n"
                f"intercept={lr.intercept:.3f}\n"
                f"r={lr.rvalue:.3f} (R²={lr.rvalue**2:.3f})\n"
                f"p={lr.pvalue:.3g}"
            )
            plt.gcf().text(0.02, 0.98, stats_text, va='top', ha='left', fontsize=9,
                           bbox=dict(boxstyle="round", alpha=0.15, lw=0.5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(reg_rows), prevalence_df


# ---------------------------------------------------------------------------
# 3) Generalized: two columns from two dataframes (country-wise comparison)
# ---------------------------------------------------------------------------
def countrywise_regression(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_country_col: str = "country_code",
    left_value_col: str,
    right_country_col: str = "country",
    right_value_col: str,
    left_agg: str = "mean",    # aggregation for left value by country
    right_agg: str = "mean",   # aggregation for right value by country
    min_samples_left: int = 1, # minimum rows per country in left_df
    min_samples_right: int = 1,
    annotate_countries: bool = True,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate one numeric column from each dataframe by country, merge, and regress.
    Returns (regression_table, merged_country_df).
    """
    # normalize codes
    ldf = left_df.copy()
    ldf[left_country_col] = ldf[left_country_col].str.upper()

    rdf = right_df.copy()
    rdf[right_country_col] = rdf[right_country_col].str.upper()

    # dropna on needed
    ldf = ldf.dropna(subset=[left_country_col, left_value_col])
    rdf = rdf.dropna(subset=[right_country_col, right_value_col])

    # sample filters
    l_counts = ldf[left_country_col].value_counts()
    l_keep = l_counts[l_counts >= min_samples_left].index
    ldf = ldf[ldf[left_country_col].isin(l_keep)]

    r_counts = rdf[right_country_col].value_counts()
    r_keep = r_counts[r_counts >= min_samples_right].index
    rdf = rdf[rdf[right_country_col].isin(r_keep)]

    # aggregate
    l_g = getattr(ldf.groupby(left_country_col)[left_value_col], left_agg)()
    r_g = getattr(rdf.groupby(right_country_col)[right_value_col], right_agg)()

    merged = (
        l_g.rename("x_var").reset_index()
           .merge(r_g.rename("y_var").reset_index(),
                  left_on=left_country_col, right_on=right_country_col, how="inner")
    )

    if merged.empty:
        return pd.DataFrame(columns=[
            "x_var","y_var","N","slope","intercept","r","r2","p","stderr",
            "x_mean","y_mean","x_std","y_std"
        ]), merged

    x = merged["x_var"].to_numpy()
    y = merged["y_var"].to_numpy()
    lr = linregress(x, y)

    reg_table = pd.DataFrame([{
        "x_var"   : left_value_col,
        "y_var"   : right_value_col,
        "N"       : int(len(merged)),
        "slope"   : lr.slope,
        "intercept": lr.intercept,
        "r"       : lr.rvalue,
        "r2"      : lr.rvalue**2,
        "p"       : lr.pvalue,
        "stderr"  : lr.stderr,
        "x_mean"  : float(np.mean(x)),
        "y_mean"  : float(np.mean(y)),
        "x_std"   : float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "y_std"   : float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
    }])

    if show_plots:
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y)
        xline = np.linspace(x.min(), x.max(), 200)
        yline = lr.slope * xline + lr.intercept
        plt.plot(xline, yline, linestyle='--', label='Linear Fit')

        # choose the country code column to annotate from merged
        country_col = left_country_col if left_country_col in merged.columns else right_country_col
        if annotate_countries and country_col in merged.columns:
            for _, r in merged.iterrows():
                plt.text(r["x_var"], r["y_var"], r[country_col], fontsize=8)

        plt.title(f"{left_value_col} vs {right_value_col} (country-wise)")
        plt.xlabel(left_value_col)
        plt.ylabel(right_value_col)
        stats_text = (
            f"N={len(merged)}\n"
            f"slope={lr.slope:.3f}\n"
            f"intercept={lr.intercept:.3f}\n"
            f"r={lr.rvalue:.3f} (R²={lr.rvalue**2:.3f})\n"
            f"p={lr.pvalue:.3g}"
        )
        plt.gcf().text(0.02, 0.98, stats_text, va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle="round", alpha=0.15, lw=0.5))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return reg_table, merged


def get_country_score(df, construct_col ):
    # df is your DataFrame
    # Step 1: pick the relevant construct column (e.g., "Long-Term Orientation")
        
    # Step 2: aggregate by country_code
    country_scores = (df.groupby("country_code", as_index=False)[construct_col]
          .mean() 
          .rename(columns={construct_col: construct_col}))

    print(country_scores.head())
    return country_scores

__all__ = [
    "plot_narrative_vs_trust",
    "prevalence_vs_metrics",
    "countrywise_regression",
]


import plotly.express as px
import pycountry

def plot_country_scores(country_scores: pd.DataFrame,
                        country_col: str = "country_code",
                        score_col: str = "score",
                        title: str = "Country-level CCR score"):
    """
    Plot an interactive world choropleth of country-level scores.

    Parameters
    ----------
    country_scores : pd.DataFrame
        Must contain a column with ISO-2 or ISO-3 country codes and a numeric score column.
    country_col : str
        Name of the column containing country codes (default: 'country_code').
    score_col : str
        Name of the column containing numeric scores (default: 'score').
    title : str
        Title for the plot.
    """
    # Ensure ISO-3 codes
    def to_iso3(code):
        if pd.isna(code): return None
        code = str(code).upper()
        if len(code) == 2:  # ISO-2
            try:
                return pycountry.countries.get(alpha_2=code).alpha_3
            except:
                return None
        if len(code) == 3:
            return code
        return None

    df = country_scores.copy()
    df["iso3"] = df[country_col].apply(to_iso3)

    fig = px.choropleth(
        df.dropna(subset=["iso3"]),
        locations="iso3",
        color=score_col,
        hover_name=country_col,
        color_continuous_scale="Viridis",
        projection="natural earth",
        title=title
    )
    fig.update_layout(coloraxis_colorbar=dict(title=score_col))
    fig.show()


import pandas as pd
import numpy as np
from typing import Iterable, Tuple

def compute_country_prevalence(
    df_narratives: pd.DataFrame,
    *,
    left_country_col: str = "country_code",
    element_value_suffix: str = "_value",
    elements: Iterable[str] = ("causal_sequence", "characters", "internal_states", "plot_structure", "normative_point"),
    require_all: bool = True,     # if False, uses min_elements
    min_elements: int = 5,        # threshold if require_all=False
    min_samples: int = 30        # filter: only countries with >= this many rows
) -> pd.DataFrame:
    """
    Compute narrative prevalence per country.

    Prevalence definition:
      - If require_all=True: % of texts that have ALL elements present.
      - Else: % of texts that have at least `min_elements` present.

    Returns
    -------
    prevalence_df : pd.DataFrame with columns:
        [left_country_col, 'n', 'with_pattern', 'narrative_prevalence']
      where 'narrative_prevalence' is in percent (0..100).
    """
    df = df_narratives.copy()
    df[left_country_col] = df[left_country_col].str.upper()

    value_cols = [f"{e}{element_value_suffix}" for e in elements]
    needed = [left_country_col] + value_cols
    df = df.dropna(subset=needed)

    # sample size filter
    eligible = df[left_country_col].value_counts()
    keep_countries = eligible[eligible >= min_samples].index
    df = df[df[left_country_col].isin(keep_countries)]

    if df.empty:
        return pd.DataFrame(columns=[left_country_col, "n", "with_pattern", "narrative_prevalence"])

    # prevalence condition
    counts_present = df[value_cols].sum(axis=1)
    if require_all:
        has_pattern = (counts_present == len(value_cols))
    else:
        has_pattern = (counts_present >= min_elements)

    df = df.assign(__has_pattern=has_pattern)

    # totals and with-pattern counts
    totals = df[left_country_col].value_counts().sort_index()
    with_pattern = df[df["__has_pattern"]][left_country_col].value_counts().reindex(totals.index).fillna(0.0)

    prevalence = (with_pattern / totals * 100.0)

    prevalence_df = pd.DataFrame({
        left_country_col: totals.index,
        "n": totals.values.astype(int),
        "with_pattern": with_pattern.values.astype(int),
        "narrative_prevalence": prevalence.values.astype(float)
    })

    return prevalence_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

def regress_country_scores(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    score_col: str,
    right_metric: str,
    left_country_col: str = "country_code",
    right_country_col: str = "country",
    annotate: bool = True,
    figsize=(6.5, 4.5)
) -> pd.DataFrame:
    """
    Merge left_df (with CCR country scores) and right_df (with external metrics),
    run regression, plot scatter with regression line, and return regression summary.

    Parameters
    ----------
    left_df : DataFrame
        Contains country-level scores, including [left_country_col, score_col].
    right_df : DataFrame
        Contains external metrics, including [right_country_col, right_metric].
    score_col : str
        Column name in left_df with CCR country scores (e.g. 'Long-Term Orientation').
    right_metric : str
        Column name in right_df with the metric to regress on (e.g. 'GDP').
    left_country_col : str
        Country identifier in left_df (default 'country_code').
    right_country_col : str
        Country identifier in right_df (default 'country').
    annotate : bool
        Whether to annotate points with country codes on the scatter plot.
    figsize : tuple
        Size of the plot.

    Returns
    -------
    pd.DataFrame
        One-row regression summary with slope, intercept, r, r², p, stderr, means/stds.
    """
    # Merge
    merged = left_df.merge(right_df, left_on=left_country_col, right_on=right_country_col, how="inner").dropna(
        subset=[score_col, right_metric]
    )
    if merged.empty:
        raise ValueError("No overlap between country sets or missing values.")

    x = merged[score_col].to_numpy()
    y = merged[right_metric].to_numpy()

    lr = linregress(x, y)

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y)
    line_x = pd.Series(np.linspace(x.min(), x.max(), 200))
    plt.plot(line_x, lr.slope * line_x + lr.intercept, linestyle="--", color="red", label="Linear Fit")

    if annotate:
        for _, r in merged.iterrows():
            plt.text(r[score_col] + 0.2, r[right_metric], r[left_country_col], fontsize=8)

    plt.title(f"{score_col} vs {right_metric}")
    plt.xlabel(score_col)
    plt.ylabel(right_metric)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Regression table ---
    reg_row = {
        "x_var": score_col,
        "y_var": right_metric,
        "N": len(merged),
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r": lr.rvalue,
        "r2": lr.rvalue**2,
        "p": lr.pvalue,
        "stderr": lr.stderr,
        "x_mean": float(x.mean()),
        "y_mean": float(y.mean()),
        "x_std": float(x.std(ddof=1)),
        "y_std": float(y.std(ddof=1)),
    }
    return pd.DataFrame([reg_row])

