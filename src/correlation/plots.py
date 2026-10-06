"""Correlation scatter plot and QQ plot of the regression residuals."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from .config import X_COLUMN, Y_COLUMN
from .stats import RegressionResult


def get_style_column(merge_columns: list[str]) -> str | None:
    """Pick the column used for the marker style of the scatter plot.

    Args:
        merge_columns: Columns defining one item.

    Returns:
        The first of ``feature``, ``test_type`` or ``gender`` present in
        ``merge_columns``, or None.
    """
    for column in ("feature", "test_type", "gender"):
        if column in merge_columns:
            return column
    return None


def plot_correlation(
    merged_df: pd.DataFrame, result: RegressionResult, merge_columns: list[str]
) -> None:
    """Scatter plot of human vs ASR intelligibility with the regression line.

    Args:
        merged_df: Output of :func:`correlation.data.get_merged_df`.
        result: Output of :func:`correlation.stats.fit_regression`.
        merge_columns: Columns defining one item, used to pick the marker style.
    """
    plt.figure(figsize=(10, 8))

    # Identity line: perfect agreement between ASR and humans
    x_range = [merged_df[X_COLUMN].min(), merged_df[X_COLUMN].max()]
    plt.plot(x_range, x_range, color="gray", linestyle="dashed", linewidth=1)

    sns.scatterplot(
        data=merged_df,
        x=X_COLUMN,
        y=Y_COLUMN,
        hue="condition",
        style=get_style_column(merge_columns),
        palette="Set2",
        alpha=0.7,
    )
    sns.regplot(
        data=merged_df,
        x=X_COLUMN,
        y=Y_COLUMN,
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2, "linestyle": "dashed"},
    )

    plt.text(
        0.05,
        0.95,
        f"R: {result.correlation:.2f}\nR²: {result.r_squared:.2f}\nN: {len(merged_df)}"
        f"\ny = {result.slope:.2f}x + {result.intercept:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
        ),
    )
    plt.title("ASR Intelligibility (Mixture of models) vs Human Intelligibility")
    plt.xlabel("ASR Intelligibility")
    plt.ylabel("Human Intelligibility")
    plt.grid()
    plt.show()


def plot_residuals_qq(residuals: pd.Series) -> None:
    """QQ plot of the regression residuals against a normal distribution.

    Args:
        residuals: Residuals of the regression.
    """
    sm.qqplot(residuals, line="s")
    plt.title("QQ Plot of Residuals")
    plt.grid()
    plt.show()
