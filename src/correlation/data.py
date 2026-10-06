"""Loading, aggregation and merging of the ASR and human ratings."""

from pathlib import Path

import pandas as pd

from utils import ASR_DATABASE_PATH, get_ASR_ratings, get_human_ratings

from .config import (
    LLR_ITEM_COLUMNS,
    MERGE,
    METRIC,
    METRICS,
    MODELS,
    X_COLUMN,
    Y_COLUMN,
)


def add_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``gender`` column inferred from the first letter of ``talker``.

    Args:
        df: Ratings DataFrame. Left untouched if it has no ``talker`` column.

    Returns:
        The same DataFrame with a ``gender`` column ("female", "male" or "NA").
    """
    if "talker" in df.columns:
        df["gender"] = df["talker"].apply(
            lambda x: (
                "female" if x.startswith("f") else "male" if x.startswith("m") else "NA"
            )
        )
    return df


def load_ASR_ratings(
    models: list[str],
    carrier_phrase_conditions: list[bool],
    database_path: Path = ASR_DATABASE_PATH,
) -> pd.DataFrame:
    """Concatenate the raw ratings of all ASR models and carrier phrase conditions.

    Args:
        models: Names of the ASR models to load.
        carrier_phrase_conditions: Carrier phrase conditions to load for each model.
        database_path: SQLite database containing the ASR results.

    Returns:
        Raw ASR ratings with a ``gender`` column.
    """
    raw_df_ASR = pd.concat(
        [
            get_ASR_ratings(
                model,
                with_carrier_phrase=carrier_phrase,
                database_path=database_path,
            )
            for model in models
            for carrier_phrase in carrier_phrase_conditions
        ],
        ignore_index=True,
    )
    return add_gender_column(raw_df_ASR)


def aggregate_ASR_ratings(
    raw_df_ASR: pd.DataFrame, merge_columns: list[str], metric: str
) -> pd.DataFrame:
    """Aggregate the ASR ratings per item and compute the predicted intelligibility.

    Args:
        raw_df_ASR: Raw ASR ratings, as returned by :func:`load_ASR_ratings`.
        merge_columns: Columns defining one item.
        metric: How ``predicted_intelligibility`` is computed:

            - ``SOFT_VOTING``: sum of soft scores over sum of normalisers (NUM / DEN).
            - ``HARD_VOTING``: proportion of correct hard decisions (NUM_hard / total_count).
            - ``LOG_LIKELIHOOD_RATIO``: log-probability of the present word minus
              that of the absent word, averaged over the items.

    Returns:
        One row per item with the aggregated ASR ratings.

    Raises:
        ValueError: If ``metric`` is unknown.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}, expected one of {METRICS}.")

    group_columns = (
        LLR_ITEM_COLUMNS if metric == "LOG_LIKELIHOOD_RATIO" else merge_columns
    )
    df_ASR = raw_df_ASR.groupby(group_columns, as_index=False).agg(
        {
            "intelligibility": "mean",
            "predicted_intelligibility": "mean",
            "total_count": "sum",
            "NUM": "sum",
            "NUM_hard": "sum",
            "DEN": "sum",
            "present_word_log_proba": "sum",
            "absent_word_log_proba": "sum",
        }
    )

    if metric == "SOFT_VOTING":
        df_ASR[X_COLUMN] = df_ASR["NUM"] / df_ASR["DEN"]

    elif metric == "HARD_VOTING":
        df_ASR[X_COLUMN] = df_ASR["NUM_hard"] / df_ASR["total_count"]

    elif metric == "LOG_LIKELIHOOD_RATIO":
        df_ASR[X_COLUMN] = (
            df_ASR["present_word_log_proba"] - df_ASR["absent_word_log_proba"]
        )
        df_ASR = df_ASR.groupby(merge_columns, as_index=False).agg(
            {
                "intelligibility": "mean",
                "predicted_intelligibility": "mean",
                "total_count": "sum",
                "NUM": "sum",
                "DEN": "sum",
                "present_word_log_proba": "mean",
                "absent_word_log_proba": "mean",
            }
        )

    return df_ASR


def aggregate_human_ratings(merge_columns: list[str]) -> pd.DataFrame:
    """Load the human ratings and aggregate them per item.

    Args:
        merge_columns: Columns defining one item.

    Returns:
        One row per item with the mean human intelligibility and the total count.
    """
    raw_df_humans = add_gender_column(get_human_ratings())
    return raw_df_humans.groupby(merge_columns, as_index=False).agg(
        {
            "intelligibility": "mean",
            "total_count": "sum",
        }
    )


def get_merged_df(
    carrier_phrase_conditions: list[bool],
    models: list[str] = MODELS,
    merge_columns: list[str] = MERGE,
    metric: str = METRIC,
) -> pd.DataFrame:
    """Build the item-level table pairing ASR and human intelligibility.

    Args:
        carrier_phrase_conditions: Carrier phrase conditions of the ASR ratings to pool.
        models: Names of the ASR models to pool.
        merge_columns: Columns defining one item.
        metric: ASR scoring method, see :func:`aggregate_ASR_ratings`.

    Returns:
        Merged DataFrame where columns shared by both sources are suffixed with
        ``_ASR`` and ``_human``.
    """
    raw_df_ASR = load_ASR_ratings(models, carrier_phrase_conditions)
    df_ASR = aggregate_ASR_ratings(raw_df_ASR, merge_columns, metric)
    df_humans = aggregate_human_ratings(merge_columns)

    return pd.merge(
        df_ASR,
        df_humans,
        on=merge_columns,
        suffixes=("_ASR", "_human"),
    )


def get_predictors_df(
    predictors: dict[str, tuple[str, list[bool]]],
    models: list[str],
    merge_columns: list[str],
) -> pd.DataFrame:
    """Build the item-level table pairing human intelligibility with several ASR predictors.

    A predictor is an ASR metric computed on a given carrier phrase
    configuration. The predictors are inner-joined on ``merge_columns`` so that
    they are all compared on the same items.

    Args:
        predictors: Maps each predictor name to its metric (see
            :func:`aggregate_ASR_ratings`) and its carrier phrase conditions.
        models: Names of the ASR models to pool.
        merge_columns: Columns defining one item.

    Returns:
        One row per item with the ``merge_columns``, ``intelligibility_human``
        and one column per predictor, named after the predictor.
    """
    # Load each carrier phrase configuration only once
    raw_dfs_ASR: dict[tuple[bool, ...], pd.DataFrame] = {}

    predictors_df = aggregate_human_ratings(merge_columns).rename(
        columns={"intelligibility": Y_COLUMN}
    )
    for name, (metric, carrier_phrase_conditions) in predictors.items():
        key = tuple(carrier_phrase_conditions)
        if key not in raw_dfs_ASR:
            raw_dfs_ASR[key] = load_ASR_ratings(models, carrier_phrase_conditions)

        df_ASR = aggregate_ASR_ratings(raw_dfs_ASR[key], merge_columns, metric)
        predictors_df = predictors_df.merge(
            df_ASR[merge_columns + [X_COLUMN]].rename(columns={X_COLUMN: name}),
            on=merge_columns,
        )
    return predictors_df
