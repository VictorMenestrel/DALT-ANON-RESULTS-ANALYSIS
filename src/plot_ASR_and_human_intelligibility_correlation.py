"""Correlation between ASR-predicted and human intelligibility on the DALT.

The ratings of several ASR models are pooled and compared
to the human ratings item by item. A linear regression is fitted between both
intelligibility scores, and its residuals are checked for heteroscedasticity
(Breusch-Pagan) and normality (Shapiro-Wilk, QQ plot).

Example (from the repository root):
    uv run src/plot_ASR_and_human_intelligibility_correlation.py \\
        --carrier-phrase both --merge condition feature --metric HARD_VOTING
"""

import argparse

import pandas as pd

from correlation.cli import add_common_arguments
from correlation.config import CARRIER_PHRASE_CHOICES, MERGE, METRIC, METRICS
from correlation.data import get_merged_df
from correlation.plots import plot_correlation, plot_residuals_qq
from correlation.stats import fit_regression, report_regression


def plot_ASR_probability_confidence_vs_ambiguity(
    merged_df: pd.DataFrame, merge_columns: list[str] = MERGE
) -> None:
    """Fit, report and plot the correlation between ASR and human intelligibility.

    Args:
        merged_df: Output of :func:`correlation.data.get_merged_df`.
        merge_columns: Columns defining one item.
    """
    result = fit_regression(merged_df)
    report_regression(merged_df, result)
    plot_correlation(merged_df, result, merge_columns)
    plot_residuals_qq(result.residuals)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        Parsed arguments, defaulting to the values of :mod:`correlation.config`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, merge_default=MERGE)
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default=METRIC,
        help="ASR scoring method (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the correlation analysis with the command line arguments."""
    args = parse_args()
    merged_df = get_merged_df(
        carrier_phrase_conditions=CARRIER_PHRASE_CHOICES[args.carrier_phrase],
        models=args.models,
        merge_columns=args.merge,
        metric=args.metric,
    )
    plot_ASR_probability_confidence_vs_ambiguity(merged_df, merge_columns=args.merge)


if __name__ == "__main__":
    main()
