"""Command line arguments shared by the correlation scripts."""

import argparse

from .config import (
    CARRIER_PHRASE,
    CARRIER_PHRASE_CHOICES,
    MERGE_CHOICES,
    MODELS,
)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    merge_default: list[str],
    carrier_phrase: bool = True,
) -> None:
    """Add the ``--models``, ``--carrier-phrase`` and ``--merge`` arguments.

    Args:
        parser: Parser to extend.
        merge_default: Default columns defining one item.
        carrier_phrase: Whether to add ``--carrier-phrase``.
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        metavar="MODEL",
        help="ASR models to pool (default: all models in MODELS).",
    )
    if carrier_phrase:
        parser.add_argument(
            "--carrier-phrase",
            choices=CARRIER_PHRASE_CHOICES.keys(),
            default=next(
                name
                for name, conditions in CARRIER_PHRASE_CHOICES.items()
                if conditions == CARRIER_PHRASE
            ),
            help="Carrier phrase condition of the ASR ratings (default: %(default)s).",
        )
    parser.add_argument(
        "--merge",
        nargs="+",
        choices=MERGE_CHOICES,
        default=merge_default,
        metavar="COLUMN",
        help=f"Columns defining one item, among {', '.join(MERGE_CHOICES)} "
        "(default: %(default)s).",
    )
