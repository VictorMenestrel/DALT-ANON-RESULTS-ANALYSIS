"""Split-half reliability of the human intelligibility ratings on the DALT.

For each item, the individual ratings are randomly split into two halves of
equal size, and the item-level means of both halves are correlated (Pearson r).
The correlation is corrected for the halved test length with the
Spearman-Brown formula. The random split is repeated to estimate the mean
reliability and its 95 % interval.

Example (from the repository root):
    uv run src/compute_split_half_human.py --n-iterations 100
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import get_individual_human_ratings

# Columns defining one item
ITEM_COLUMNS = [
    "test_type",
    "condition",
    "feature",
    "present_word",
    "absent_word",
    "talker",
]
EXPECTED_N_ITEMS = 2006

N_ITERATIONS = 100
# Items have 9 ratings: one is discarded at random so that both halves have 4
RATINGS_PER_ITEM = 8


@dataclass
class SplitHalfResult:
    """Split-half reliability over repeated random splits."""

    pearson_r: np.ndarray
    spearman_brown: np.ndarray

    @property
    def n_iterations(self) -> int:
        """Number of random splits."""
        return len(self.pearson_r)

    @property
    def ci_95(self) -> tuple[float, float]:
        """Percentile 95 % interval of the Spearman-Brown reliability."""
        low, high = np.percentile(self.spearman_brown, [2.5, 97.5])
        return low, high


def spearman_brown(r: np.ndarray) -> np.ndarray:
    """Spearman-Brown correction of a split-half correlation to full test length.

    Args:
        r: Correlation between the two halves.

    Returns:
        The estimated reliability of the full test.
    """
    return 2 * r / (1 + r)


def split_half_correlation(df: pd.DataFrame, ratings_per_item: int, seed: int) -> float:
    """Correlate the item means of two random halves of the ratings.

    Args:
        df: Individual ratings, as returned by :func:`get_individual_human_ratings`.
        ratings_per_item: Number of ratings kept per item, split equally between
            both halves. Extra ratings are discarded at random.
        seed: Random seed of the shuffle.

    Returns:
        Pearson correlation between the item means of both halves.
    """
    # Shuffling first makes the discarded ratings and the split random
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["rank"] = shuffled.groupby(ITEM_COLUMNS).cumcount()

    kept = shuffled[shuffled["rank"] < ratings_per_item].copy()
    # Alternating ranks give halves of equal size within each item
    kept["half"] = kept["rank"] % 2

    halves = kept.groupby(ITEM_COLUMNS + ["half"])["score"].mean().unstack("half")
    halves = halves.dropna()
    assert len(halves) == EXPECTED_N_ITEMS, (
        f"Expected {EXPECTED_N_ITEMS} unique items, but got {len(halves)}"
    )

    return halves[0].corr(halves[1])


def compute_split_half_reliability(
    df: pd.DataFrame,
    n_iterations: int = N_ITERATIONS,
    ratings_per_item: int = RATINGS_PER_ITEM,
) -> SplitHalfResult:
    """Repeat the random split-half correlation and its Spearman-Brown correction.

    Args:
        df: Individual ratings, as returned by :func:`get_individual_human_ratings`.
        n_iterations: Number of random splits, each using its index as seed.
        ratings_per_item: Number of ratings kept per item, see
            :func:`split_half_correlation`.

    Returns:
        The correlation and reliability of every split.
    """
    pearson_r = np.array(
        [
            split_half_correlation(df, ratings_per_item, seed=i)
            for i in range(n_iterations)
        ]
    )
    return SplitHalfResult(
        pearson_r=pearson_r, spearman_brown=spearman_brown(pearson_r)
    )


def report_split_half(result: SplitHalfResult) -> None:
    """Print the split-half reliability summary.

    Args:
        result: Output of :func:`compute_split_half_reliability`.
    """
    low, high = result.ci_95
    print(f"Iterations: {result.n_iterations}")
    print(
        f"Mean Pearson r: {result.pearson_r.mean():.4f} (± {result.pearson_r.std():.4f})"
    )
    print(
        f"Mean Spearman-Brown: {result.spearman_brown.mean():.4f} "
        f"(± {result.spearman_brown.std():.4f})"
    )
    print(f"95 % interval: [{low:.4f}, {high:.4f}]")


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        Parsed arguments, defaulting to the module-level constants.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=N_ITERATIONS,
        help="Number of random splits (default: %(default)s).",
    )
    parser.add_argument(
        "--ratings-per-item",
        type=int,
        default=RATINGS_PER_ITEM,
        help="Number of ratings kept per item, must be even (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.ratings_per_item < 2 or args.ratings_per_item % 2:
        parser.error("--ratings-per-item must be an even number of at least 2.")
    return args


def main() -> None:
    """Compute the split-half reliability with the command line arguments."""
    args = parse_args()
    result = compute_split_half_reliability(
        get_individual_human_ratings(),
        n_iterations=args.n_iterations,
        ratings_per_item=args.ratings_per_item,
    )
    report_split_half(result)


if __name__ == "__main__":
    main()
