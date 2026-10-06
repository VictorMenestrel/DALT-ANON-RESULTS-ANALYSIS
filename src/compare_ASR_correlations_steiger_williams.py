"""Steiger-Williams test between two ASR predictors of human intelligibility on the DALT.

The ratings of several ASR models are pooled and scored twice, either with two
different metrics or on two different carrier phrase configurations. The test
checks whether both predictors correlate equally well with human
intelligibility, given that they are computed on the same items and are
therefore correlated with each other.

Examples (from the repository root):
    # Compare two metrics on the same carrier phrase configuration
    uv run src/compare_ASR_correlations_steiger_williams.py metrics \\
        --metrics HARD_VOTING SOFT_VOTING --carrier-phrase without

    # Compare two carrier phrase configurations with the same metric
    uv run src/compare_ASR_correlations_steiger_williams.py carrier-phrase \\
        --metric LOG_LIKELIHOOD_RATIO --carrier-phrases without both
"""

import argparse

from correlation.cli import add_common_arguments
from correlation.config import CARRIER_PHRASE_CHOICES, METRICS, STEIGER_MERGE
from correlation.data import get_predictors_df
from correlation.stats import compare_correlations, report_steiger


def check_distinct(parser: argparse.ArgumentParser, option: str, values: list) -> None:
    """Exit with an error if the two values of a comparison are identical.

    Args:
        parser: Parser used to report the error.
        option: Name of the option, for the error message.
        values: The two compared values.
    """
    if values[0] == values[1]:
        parser.error(f"{option} must be two different values.")


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        Parsed arguments, defaulting to the values of :mod:`correlation.config`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="comparison", required=True)

    metrics_parser = subparsers.add_parser(
        "metrics", help="Compare two metrics on the same carrier phrase configuration."
    )
    add_common_arguments(metrics_parser, merge_default=STEIGER_MERGE)
    metrics_parser.add_argument(
        "--metrics",
        nargs=2,
        choices=METRICS,
        default=["HARD_VOTING", "SOFT_VOTING"],
        metavar="METRIC",
        help=f"The two ASR scoring methods to compare, among {', '.join(METRICS)} "
        "(default: %(default)s).",
    )

    carrier_parser = subparsers.add_parser(
        "carrier-phrase",
        help="Compare two carrier phrase configurations with the same metric.",
    )
    add_common_arguments(
        carrier_parser, merge_default=STEIGER_MERGE, carrier_phrase=False
    )
    carrier_parser.add_argument(
        "--carrier-phrases",
        nargs=2,
        choices=CARRIER_PHRASE_CHOICES.keys(),
        default=["without", "both"],
        metavar="CARRIER_PHRASE",
        help="The two carrier phrase configurations to compare, among "
        f"{', '.join(CARRIER_PHRASE_CHOICES)} (default: %(default)s).",
    )
    carrier_parser.add_argument(
        "--metric",
        choices=METRICS,
        default="LOG_LIKELIHOOD_RATIO",
        help="ASR scoring method (default: %(default)s).",
    )

    args = parser.parse_args()
    if args.comparison == "metrics":
        check_distinct(metrics_parser, "--metrics", args.metrics)
    else:
        check_distinct(carrier_parser, "--carrier-phrases", args.carrier_phrases)
    return args


def get_predictors(args: argparse.Namespace) -> dict[str, tuple[str, list[bool]]]:
    """Build the two predictors to compare from the command line arguments.

    Args:
        args: Output of :func:`parse_args`.

    Returns:
        Maps each predictor name to its metric and carrier phrase conditions.
    """
    if args.comparison == "metrics":
        carrier_phrase_conditions = CARRIER_PHRASE_CHOICES[args.carrier_phrase]
        return {metric: (metric, carrier_phrase_conditions) for metric in args.metrics}

    return {
        f"{args.metric} ({carrier_phrase})": (
            args.metric,
            CARRIER_PHRASE_CHOICES[carrier_phrase],
        )
        for carrier_phrase in args.carrier_phrases
    }


def main() -> None:
    """Run the Steiger-Williams test with the command line arguments."""
    args = parse_args()
    predictors = get_predictors(args)
    name_1, name_2 = predictors

    predictors_df = get_predictors_df(
        predictors, models=args.models, merge_columns=args.merge
    )
    result = compare_correlations(predictors_df, name_1, name_2)
    report_steiger(result, name_1, name_2)


if __name__ == "__main__":
    main()
