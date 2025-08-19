
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


def plot_narrative_prevalence(df, min_samples=100):
    # Define narrative element columns
    elements = ["causal_sequence", "characters", "internal_states", "plot_structure", "normative_point"]
    value_cols = [f"{el}_value" for el in elements]

    # Drop rows with missing country or any narrative element
    df = df.dropna(subset=["country_code"] + value_cols)
    df["country_code"] = df["country_code"].astype(str)

    # Add a flag: does this row have all 5 narrative elements?
    df["has_all_5"] = df[value_cols].sum(axis=1) == 5

    # Count total and 'narrative' per country
    country_counts = df["country_code"].value_counts()
    eligible_countries = country_counts[country_counts >= min_samples].index

    df_filtered = df[df["country_code"].isin(eligible_countries)]

    total_counts = df_filtered["country_code"].value_counts()
    narrative_counts = df_filtered[df_filtered["has_all_5"]]["country_code"].value_counts()

    # Compute percentage
    prevalence_pct = (narrative_counts / total_counts.loc[narrative_counts.index]) * 100
    prevalence_pct = prevalence_pct.sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 5))
    prevalence_pct.plot(kind="bar", color="purple")
    plt.ylabel("Percentage of Texts with All 5 Narrative Elements")
    plt.xlabel("Country Code")
    plt.title(f"Narrative Prevalence by Country (min {min_samples} samples)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return prevalence_pct


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

    return reg_table


__all__ = [
  "plot_narrative_prevalence",
    "plot_narrative_vs_trust",
    "prevalence_vs_metrics",
    "countrywise_regression",

]
