"""Estimate the human intelligibility of an anonymization system with the DALT.

0. Anonymize the clean audio samples of
   data/audio/samples/clean/English/test-data (without carrier phrase), keeping
   the file names. This step is done outside of this script.
1. Run the 12 ASR models on the anonymized audio samples.
2. Score the answers with HARD_VOTING and aggregate them a- over all samples,
   b- per test type and c- per feature.
3. Map each aggregated ASR score to an estimated human intelligibility with the
   linear regression fitted on the 6 conditions rated by humans.

The ASR results are stored in a SQLite database per system, so that an
interrupted run resumes with the models not evaluated yet.

Example (from the repository root):
    uv run src/take_the_test.py --audio-folder path/to/anonymized --name my_system
"""

import argparse
from pathlib import Path

import pandas as pd

from correlation.config import X_COLUMN
from correlation.data import aggregate_ASR_ratings, load_ASR_ratings
from run_eval import MODEL_BACKENDS, evaluate_audio_files, load_backend, table_exists
from utils import get_table_name

TEST_DESIGN_PATH = Path("data/audio/samples/clean/English/test_design.csv")
OUTPUT_FOLDER = Path("data/take_the_test")

# Feature names of the test design mapped to the ones of the ratings databases
FEATURE_NAMES = {
    "voicing": "Voicing",
    "nasality": "Nasality",
    "sustention": "Sustention",
    "sibilation": "Sibilation",
    "graveness": "Graveness",
    "compactness": "Compactness",
}

# Regression lines (slope, intercept) of the human on the ASR HARD_VOTING
# intelligibility, fitted on the 6 conditions rated by humans with
# plot_ASR_and_human_intelligibility_correlation.py --metric HARD_VOTING
# --carrier-phrase without --merge condition [test_type | feature]
HARD_VOTING_MAPPINGS = {
    "all": (0.733863, 0.254060),  # N = 6 conditions
    "test_type": (0.756790, 0.236537),  # N = 12 (condition, test_type)
    "feature": (0.788272, 0.212755),  # N = 36 (condition, feature)
}
# Columns defining one item of each aggregation level, besides the condition
LEVEL_COLUMNS = {
    "all": [],
    "test_type": ["test_type"],
    "feature": ["feature"],
}


def load_design(design_path: Path, audio_folder: Path, name: str) -> pd.DataFrame:
    """Map the anonymized audio files to their DALT trial.

    Some recordings are used in two word pairs, so a file can appear in
    several trials.

    Args:
        design_path: Test design CSV of the clean samples.
        audio_folder: Folder of the anonymized audio files, named like the
            clean ones.
        name: Name of the anonymization system, used as condition.

    Returns:
        One row per trial with the columns expected by
        :func:`run_eval.evaluate_audio_files`.

    Raises:
        FileNotFoundError: If none of the audio files is in ``audio_folder``.
        ValueError: If the design contains an unknown feature.
    """
    design_df = pd.read_csv(design_path)
    # Some cells of the design have trailing spaces
    design_df = design_df.map(lambda v: v.strip() if isinstance(v, str) else v)

    unknown_features = set(design_df["feature"]) - FEATURE_NAMES.keys()
    if unknown_features:
        raise ValueError(f"Unknown features in {design_path}: {unknown_features}")

    samples_df = pd.DataFrame(
        {
            "audio_path": design_df["filename"].map(lambda f: audio_folder / f),
            "carrier_phrase": None,
            "present_word": design_df["target"],
            "absent_word": design_df["alternative"],
            "feature": design_df["feature"].map(FEATURE_NAMES),
            "test_type": design_df["location"],
            "talker": design_df["speaker_id"],
            "condition": name,
            # File names are <word>_<hash>.wav
            "hash": design_df["filename"].map(lambda f: Path(f).stem.split("_")[-1]),
        }
    )

    is_present = samples_df["audio_path"].map(Path.exists)
    if not is_present.any():
        raise FileNotFoundError(f"No audio file of {design_path} in {audio_folder}")
    if not is_present.all():
        print(
            f"Warning: {(~is_present).sum()}/{len(samples_df)} trials have no audio "
            f"file in {audio_folder} and are ignored."
        )
    return samples_df[is_present].reset_index(drop=True)


def run_asr(
    samples_df: pd.DataFrame,
    models: list[str],
    database_path: Path,
    overwrite: bool = False,
) -> None:
    """Run the ASR models on the anonymized audio (step 1).

    Args:
        samples_df: Output of :func:`load_design`.
        models: Names of the ASR models to run.
        database_path: SQLite database receiving one results table per model.
        overwrite: Whether to rerun the models already evaluated. Otherwise
            they are skipped, which resumes an interrupted run.
    """
    for model_name in models:
        table_name = get_table_name(model_name, with_carrier_phrase=False)
        if table_exists(database_path, table_name) and not overwrite:
            print(f"Skipping {model_name}: already evaluated in {database_path}")
            continue

        print(f"Evaluating {model_name}")
        evaluate_audio_files(
            load_backend(model_name),
            samples_df,
            database_path,
            table_name,
            model_name=model_name,
            overwrite=overwrite,
        )


def estimate_intelligibility(
    database_path: Path, models: list[str], name: str
) -> pd.DataFrame:
    """Aggregate the ASR results and estimate the human intelligibility (steps 2-3).

    Args:
        database_path: SQLite database filled by :func:`run_asr`.
        models: Names of the ASR models to pool.
        name: Name of the anonymization system, used as condition.

    Returns:
        One row per aggregated item of each level, with the number of trials,
        the ASR HARD_VOTING intelligibility and the estimated human
        intelligibility, both in [-1, 1] ((right - wrong) / total).

    Raises:
        FileNotFoundError: If some models have not been evaluated.
    """
    missing_models = [
        model_name
        for model_name in models
        if not table_exists(database_path, get_table_name(model_name, False))
    ]
    if missing_models:
        raise FileNotFoundError(
            f"Models not evaluated in {database_path}: {', '.join(missing_models)}"
        )

    raw_df_ASR = load_ASR_ratings(models, [False], database_path=database_path)

    estimates = []
    for level, level_columns in LEVEL_COLUMNS.items():
        df_ASR = aggregate_ASR_ratings(
            raw_df_ASR, ["condition"] + level_columns, "HARD_VOTING"
        )
        slope, intercept = HARD_VOTING_MAPPINGS[level]
        estimates.append(
            pd.DataFrame(
                {
                    "system": name,
                    "level": level,
                    "test_type": df_ASR.get("test_type", "all"),
                    "feature": df_ASR.get("feature", "all"),
                    # Each trial is evaluated once by each model
                    "n_trials": df_ASR["total_count"] // len(models),
                    "ASR_hard_voting": df_ASR[X_COLUMN],
                    "estimated_human_intelligibility": slope * df_ASR[X_COLUMN]
                    + intercept,
                }
            )
        )
    return pd.concat(estimates, ignore_index=True)


def report_estimates(estimates_df: pd.DataFrame) -> None:
    """Print the estimates of each aggregation level.

    Args:
        estimates_df: Output of :func:`estimate_intelligibility`.
    """
    for level, level_df in estimates_df.groupby("level", sort=False):
        print(f"\n== {level}")
        print(
            level_df.drop(columns=["system", "level"]).to_string(
                index=False, float_format="{:.4f}".format
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        Parsed arguments, defaulting to the module-level constants.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--audio-folder",
        type=Path,
        required=True,
        help="Folder of the anonymized audio files, named like the clean ones.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the anonymization system, used in the output file names.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_BACKENDS.keys(),
        default=list(MODEL_BACKENDS),
        metavar="MODEL",
        help="ASR models to run and pool (default: the 12 models of the regression).",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=TEST_DESIGN_PATH,
        help="Test design CSV of the clean samples (default: %(default)s).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Folder of the results database <name>.db and of the estimates "
        "<name>_estimates.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun the models already evaluated.",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Only compute the estimates from the models already evaluated.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the test on an anonymization system with the command line arguments."""
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    database_path = args.output_folder / f"{args.name}.db"

    if not args.skip_asr:
        samples_df = load_design(args.design, args.audio_folder, args.name)
        run_asr(samples_df, args.models, database_path, overwrite=args.overwrite)

    estimates_df = estimate_intelligibility(database_path, args.models, args.name)
    report_estimates(estimates_df)

    estimates_path = args.output_folder / f"{args.name}_estimates.csv"
    estimates_df.to_csv(estimates_path, index=False)
    print(f"\nEstimates saved to {estimates_path}")


if __name__ == "__main__":
    main()
