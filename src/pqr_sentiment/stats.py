"""Statistical helpers for category versus sentiment questions.

The original study measured Pearson correlations between one hot encoded
categories and sentiment probabilities, which is hard to interpret and
statistically fragile. This module replaces that with the standard tools
for categorical association: chi square tests with bias corrected Cramér's
V effect sizes, and Kruskal Wallis comparisons for numeric outcomes across
groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = ["Association", "GroupComparison", "association", "kruskal_by_group", "share_table"]


@dataclass
class Association:
    """Chi square association between two categorical columns."""

    chi2: float
    p_value: float
    dof: int
    cramers_v: float
    table: pd.DataFrame


def association(df: pd.DataFrame, row_col: str, col_col: str) -> Association:
    """Chi square test plus bias corrected Cramér's V for effect size."""
    table = pd.crosstab(df[row_col], df[col_col])
    chi2, p_value, dof, _ = scipy_stats.chi2_contingency(table)

    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    # Bergsma bias correction keeps V honest on small samples.
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(r_corrected - 1, k_corrected - 1)
    cramers_v = float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0

    return Association(chi2=float(chi2), p_value=float(p_value), dof=int(dof), cramers_v=cramers_v, table=table)


@dataclass
class GroupComparison:
    """Kruskal Wallis comparison of a numeric column across groups."""

    statistic: float
    p_value: float
    medians: pd.Series
    n: pd.Series


def kruskal_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> GroupComparison:
    """Nonparametric comparison of ``value_col`` distributions per group."""
    valid = df[[value_col, group_col]].dropna()
    samples = [group[value_col].to_numpy() for _, group in valid.groupby(group_col)]
    statistic, p_value = scipy_stats.kruskal(*samples)
    grouped = valid.groupby(group_col)[value_col]
    return GroupComparison(
        statistic=float(statistic),
        p_value=float(p_value),
        medians=grouped.median(),
        n=grouped.size(),
    )


def share_table(df: pd.DataFrame, cat_col: str, label_col: str, order: list[str] | None = None) -> pd.DataFrame:
    """Row normalized proportion table, ready for stacked share charts."""
    table = pd.crosstab(df[cat_col], df[label_col], normalize="index")
    if order is not None:
        table = table.reindex(columns=order, fill_value=0.0)
    return table
