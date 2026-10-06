import argparse
import importlib
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pandas as pd
from tqdm import tqdm

from utils import get_table_name

AUDIO_ROOT = Path("data/audio")

# TTS audio has no carrier phrase file: the carrier phrase is fixed
TTS_CONDITION = "TTS_Qwen"
TTS_CARRIER_PHRASE = "please select the word "

# Number of results kept in memory before being written to the database
BATCH_SIZE = 500

RESULT_COLUMNS = [
    "file_path",
    "present_word",
    "absent_word",
    "predicted_word",
    "probability_generated_text",
    "is_correct",
    "feature",
    "test_type",
    "talker",
    "condition",
    "hash",
]


def load_samples(sample_folder_path: Path) -> pd.DataFrame:
    """Concatenate all the sample CSV files of a folder.

    Args:
        sample_folder_path: Folder containing the sample CSV files.

    Returns:
        One row per evaluation to run.

    Raises:
        FileNotFoundError: If the folder contains no CSV file.
    """
    sample_files = sorted(sample_folder_path.glob("*.csv"))
    if not sample_files:
        raise FileNotFoundError(f"No sample CSV file found in {sample_folder_path}")
    return pd.concat([pd.read_csv(file) for file in sample_files], ignore_index=True)


def load_carrier_mapping(carrier_metadata_path: Path) -> dict[str, str]:
    """Map each original audio filename to its version with carrier phrase.

    Args:
        carrier_metadata_path: CSV with the ``target_original_filename`` and
            ``output_filename`` columns.

    Returns:
        Mapping from original filename to filename with carrier phrase. The
        first entry is kept for duplicated original filenames.
    """
    metadata_df = pd.read_csv(carrier_metadata_path).drop_duplicates(
        "target_original_filename"
    )
    return dict(
        zip(metadata_df["target_original_filename"], metadata_df["output_filename"])
    )


def get_audio_folder_path(condition: str, audio_folder_name: str) -> Path:
    """Get the folder containing the audio files of a condition.

    Args:
        condition: Condition of the audio.
        audio_folder_name: Name of the audio folder in ``data/audio``.

    Returns:
        Path of the audio folder.
    """
    if condition == "clean":
        return AUDIO_ROOT / audio_folder_name / condition / "English" / "test-data"
    if condition == TTS_CONDITION and audio_folder_name != "samples":
        return AUDIO_ROOT / condition
    return AUDIO_ROOT / audio_folder_name / condition


def get_carrier_phrase(filename_with_carrier: str) -> str:
    """Extract the carrier phrase from an audio filename.

    The filename is the words of the carrier phrase,
    separated by underscores, e.g. ``word1_word2_word3_word4.wav``.

    Args:
        filename_with_carrier: Audio filename with carrier phrase.

    Returns:
        The lower-case carrier phrase followed by a space.
    """
    words = filename_with_carrier.replace(".wav", "").split("_")[:-1]
    return " ".join(words).lower() + " "


def resolve_audio(
    initial_audio_path: str,
    condition: str,
    audio_folder_name: str,
    carrier_mapping: dict[str, str] | None,
) -> tuple[Path, str | None]:
    """Get the audio file to evaluate and its carrier phrase.

    Args:
        initial_audio_path: Audio path from the sample file.
        condition: Condition of the audio.
        audio_folder_name: Name of the audio folder in ``data/audio``.
        carrier_mapping: Output of :func:`load_carrier_mapping`, or None to
            evaluate without carrier phrase.

    Returns:
        The audio path and the carrier phrase (None without carrier phrase).

    Raises:
        KeyError: If the audio has no version with carrier phrase.
    """
    audio_folder_path = get_audio_folder_path(condition, audio_folder_name)
    audio_filename = Path(initial_audio_path).name

    if carrier_mapping is None:
        return audio_folder_path / audio_filename, None

    if condition == TTS_CONDITION:
        return audio_folder_path / audio_filename, TTS_CARRIER_PHRASE

    if audio_filename not in carrier_mapping:
        raise KeyError(f"No audio with carrier phrase for {audio_filename}")
    filename_with_carrier = carrier_mapping[audio_filename]
    return audio_folder_path / filename_with_carrier, get_carrier_phrase(
        filename_with_carrier
    )


def run_evaluation(
    backend: ModuleType,
    sample_folder_path: Path,
    sqlite_database_path: Path,
    table_name: str,
    carrier_metadata_path: Path,
    audio_folder_name: str,
    with_carrier_phrase: bool = False,
    model_name: str | None = None,
    overwrite: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """
    Run ASR evaluation on audio files using the DALT framework.

    The audio path and carrier phrase of each sample are resolved from the
    sample files, then evaluated with :func:`evaluate_audio_files`.

    Args:
        backend: Module of ``src/models`` implementing ``load_model`` and
            ``generate_tokens_from_audio``, see :func:`load_backend`
        sample_folder_path: Path to folder containing sample files
        sqlite_database_path: Path to SQLite database
        table_name: Name of the table to store results
        carrier_metadata_path: Path to the carrier metadata CSV file, only read
            with the carrier phrase
        audio_folder_name: Name of the audio folder
        with_carrier_phrase: Whether to include the carrier phrase in the evaluation
        model_name: Name of the ASR model to use (default is None, which uses the default model)
        overwrite: Whether to replace the table if it already exists
        batch_size: Number of results written to the database at once
    """
    samples_df = load_samples(sample_folder_path)
    carrier_mapping = (
        load_carrier_mapping(carrier_metadata_path) if with_carrier_phrase else None
    )

    resolved_samples = []
    for sample in samples_df.to_dict("records"):
        try:
            audio_path, carrier_phrase = resolve_audio(
                str(sample["audio_path"]),
                str(sample["condition"]),
                audio_folder_name,
                carrier_mapping,
            )
        except KeyError as e:
            print(f"Skipping {sample['audio_path']}: {e}")
            continue
        resolved_samples.append(
            {**sample, "audio_path": audio_path, "carrier_phrase": carrier_phrase}
        )

    evaluate_audio_files(
        backend,
        pd.DataFrame(resolved_samples),
        sqlite_database_path,
        table_name,
        model_name=model_name,
        overwrite=overwrite,
        batch_size=batch_size,
    )


def evaluate_audio_files(
    backend: ModuleType,
    samples_df: pd.DataFrame,
    sqlite_database_path: Path,
    table_name: str,
    model_name: str | None = None,
    overwrite: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """
    Run an ASR model on audio files and write the results to SQLite.

    The results are written to the database every ``batch_size`` evaluations,
    so that a crash only loses the last batch. Failed evaluations are skipped
    and summarized at the end.

    Args:
        backend: Module of ``src/models`` implementing ``load_model`` and
            ``generate_tokens_from_audio``, see :func:`load_backend`
        samples_df: One row per evaluation, with the resolved ``audio_path``,
            the ``carrier_phrase`` (None without carrier phrase) and the
            ``present_word``, ``absent_word``, ``feature``, ``test_type``,
            ``talker``, ``condition`` and ``hash`` columns
        sqlite_database_path: Path to SQLite database
        table_name: Name of the table to store results
        model_name: Name of the ASR model to use (default is None, which uses the default model)
        overwrite: Whether to replace the table if it already exists
        batch_size: Number of results written to the database at once
    """
    # Fail before loading the model if the table cannot be written
    create_results_table(sqlite_database_path, table_name, overwrite=overwrite)

    device, processor, tokenizer, model, prompt = backend.load_model(
        model_name=model_name
    )

    results = []
    failures = []
    for sample in tqdm(
        samples_df.itertuples(index=False), total=len(samples_df), desc="Evaluations"
    ):
        audio_path = sample.audio_path
        # pandas may turn a missing carrier phrase into NaN
        carrier_phrase = (
            sample.carrier_phrase if isinstance(sample.carrier_phrase, str) else None
        )
        try:
            result = backend.generate_tokens_from_audio(
                str(audio_path),
                device,
                processor,
                tokenizer,
                model,
                prompt,
                sample.present_word,
                sample.absent_word,
                carrier_phrase=carrier_phrase,
            )
        # One failing audio file should not stop a long evaluation run
        except Exception as e:
            failures.append(sample.audio_path)
            tqdm.write(
                f"Error processing {sample.audio_path} with pair "
                f"({sample.present_word}, {sample.absent_word}): {e!r}"
            )
            continue

        predicted_word = result["generated_text"]
        results.append(
            {
                "file_path": str(audio_path),
                "present_word": sample.present_word,
                "absent_word": sample.absent_word,
                "predicted_word": predicted_word,
                "probability_generated_text": result.get("probability_generated_text"),
                "is_correct": predicted_word == sample.present_word,
                "feature": sample.feature,
                "test_type": sample.test_type,
                "talker": sample.talker,
                "condition": sample.condition,
                "hash": sample.hash,
            }
        )
        if len(results) >= batch_size:
            write_results_to_sqlite(sqlite_database_path, table_name, results)
            results = []

    write_results_to_sqlite(sqlite_database_path, table_name, results)

    print(
        f"{len(samples_df) - len(failures)}/{len(samples_df)} evaluations written "
        f"to {table_name}."
    )
    if failures:
        print(f"{len(failures)} evaluations failed, see the errors above.")


def table_exists(sqlite_database_path: Path, table_name: str) -> bool:
    """Check whether a table exists in a SQLite database.

    Args:
        sqlite_database_path: Path to SQLite database
        table_name: Name of the table

    Returns:
        True if the database file and the table exist.
    """
    if not sqlite_database_path.exists():
        return False
    with closing(sqlite3.connect(sqlite_database_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def create_results_table(
    sqlite_database_path: Path, table_name: str, overwrite: bool = False
):
    """
    Create an empty results table.

    Args:
        sqlite_database_path: Path to SQLite database, created if missing
        table_name: Name of the table to store results
        overwrite: Whether to drop the table if it already exists

    Raises:
        FileExistsError: If the table exists and ``overwrite`` is False, to
            avoid appending duplicated results.
    """
    if table_exists(sqlite_database_path, table_name) and not overwrite:
        raise FileExistsError(
            f"Table {table_name} already exists in {sqlite_database_path}, "
            "use overwrite to replace it."
        )

    with closing(sqlite3.connect(sqlite_database_path)) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"""
            CREATE TABLE {table_name} (
                file_path TEXT,
                present_word TEXT,
                absent_word TEXT,
                predicted_word TEXT,
                probability_generated_text REAL,
                is_correct BOOLEAN,
                feature TEXT,
                test_type TEXT,
                talker TEXT,
                condition TEXT,
                hash TEXT
            )
        """)
        conn.commit()


def write_results_to_sqlite(sqlite_database_path: Path, table_name: str, results: list):
    """
    Append evaluation results to an existing results table.

    Args:
        sqlite_database_path: Path to SQLite database
        table_name: Name of the table created by :func:`create_results_table`
        results: List of evaluation results, as dicts keyed by ``RESULT_COLUMNS``
    """
    if not results:
        return

    columns = ", ".join(RESULT_COLUMNS)
    placeholders = ", ".join(f":{column}" for column in RESULT_COLUMNS)
    with closing(sqlite3.connect(sqlite_database_path)) as conn:
        conn.executemany(
            f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", results
        )
        conn.commit()


# Backend module in src/models/ implementing each supported ASR model
MODEL_BACKENDS = {
    "CohereLabs/cohere-transcribe-03-2026": "cohere",
    "facebook/wav2vec2-lv-60-espeak-cv-ft": "wav2vec2_espeak",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft": "wav2vec2_espeak",
    "openai/whisper-large-v3": "whisper",
    "openai/whisper-large-v3-turbo": "whisper",
    "distil-whisper/distil-large-v3.5": "whisper",
    "nvidia/canary-qwen-2.5b": "canary_qwen",
    "Qwen/Qwen3-ASR-1.7B-hf": "qwen_transformers",
    "Qwen/Qwen3-ASR-0.6B-hf": "qwen_transformers",
    "ibm-granite/granite-4.0-1b-speech": "granite_ibm",
    "ibm-granite/granite-speech-4.1-2b": "granite_ibm",
    "ibm-granite/granite-speech-4.1-2b-plus": "granite_ibm",
}
DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"

SQLITE_DATABASE_PATH = Path("data/ASR_ratings.db")
SAMPLE_FOLDER = Path("data/samples")
CARRIER_AUDIO_FOLDER_NAME = "level_normalized_26dbfs_w_concat_carrier_phrase_2"
NO_CARRIER_AUDIO_FOLDER_NAME = "samples"
CARRIER_METADATA_PATH = (
    AUDIO_ROOT
    / CARRIER_AUDIO_FOLDER_NAME
    / "clean"
    / "English"
    / "test-data"
    / "concat_metadata.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        Parsed arguments, defaulting to the module-level constants.
    """
    parser = argparse.ArgumentParser(
        description="Run the DALT evaluation of one or several ASR models and store "
        "the results in a SQLite database.",
        epilog="Example (from the repository root): "
        "uv run src/run_eval.py --models openai/whisper-large-v3 --carrier-phrase",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_BACKENDS.keys(),
        default=[DEFAULT_MODEL],
        metavar="MODEL",
        help="ASR models to evaluate, one after the other, among "
        f"{', '.join(MODEL_BACKENDS)} (default: %(default)s).",
    )
    parser.add_argument(
        "--carrier-phrase",
        action="store_true",
        help="Evaluate the audio with the carrier phrase.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the results tables that already exist.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=SQLITE_DATABASE_PATH,
        help="SQLite database receiving the results (default: %(default)s).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=SAMPLE_FOLDER,
        help="Folder of the sample CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--carrier-metadata",
        type=Path,
        default=CARRIER_METADATA_PATH,
        help="CSV mapping the original audio to the audio with carrier phrase "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--audio-folder",
        default=None,
        help=f"Audio folder name in data/audio/ (default: {CARRIER_AUDIO_FOLDER_NAME} "
        f"with --carrier-phrase, {NO_CARRIER_AUDIO_FOLDER_NAME} otherwise).",
    )
    args = parser.parse_args()

    # Check every table before starting
    existing_tables = [
        table_name
        for model_name in args.models
        if table_exists(
            args.database,
            table_name := get_table_name(model_name, args.carrier_phrase),
        )
    ]
    if existing_tables and not args.overwrite:
        parser.error(
            f"tables already in {args.database}: {', '.join(existing_tables)}. "
            "Use --overwrite to replace them."
        )
    return args


def load_backend(model_name: str) -> ModuleType:
    """Import the module of ``src/models`` implementing an ASR model.

    Args:
        model_name: Hugging Face name of the ASR model, a key of ``MODEL_BACKENDS``.

    Returns:
        The module, providing ``load_model`` and ``generate_tokens_from_audio``.
    """
    return importlib.import_module(f"models.{MODEL_BACKENDS[model_name]}")


def main() -> None:
    """Evaluate the ASR models given on the command line."""
    args = parse_args()
    audio_folder_name = args.audio_folder or (
        CARRIER_AUDIO_FOLDER_NAME
        if args.carrier_phrase
        else NO_CARRIER_AUDIO_FOLDER_NAME
    )

    for model_name in args.models:
        print(f"Evaluating {model_name}")
        run_evaluation(
            load_backend(model_name),
            args.samples,
            args.database,
            get_table_name(model_name, args.carrier_phrase),
            args.carrier_metadata,
            audio_folder_name,
            args.carrier_phrase,
            model_name=model_name,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
