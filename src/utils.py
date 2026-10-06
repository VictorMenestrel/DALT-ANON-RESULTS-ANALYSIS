import sqlite3
from pathlib import Path

import pandas as pd

ASR_DATABASE_PATH = Path("data/ASR_ratings.db")


def get_human_ratings() -> pd.DataFrame:
    humans_sqlite_database_path = "data/human_ratings.db"
    conn_humans = sqlite3.connect(humans_sqlite_database_path)
    query_humans = """
        -- Query to get the intelligibility from the human ratings.
        SELECT
        r.test_type,
        r.condition,
        r.feature,
        r.present_word,
        r.absent_word,
        r.talker,
        COUNT(*) AS total_count,
        SUM(CASE WHEN r.answer = r.present_word THEN 1.0 ELSE -1.0 END) / COUNT(*) AS intelligibility
        FROM ratingsession r
        INNER JOIN job j 
            ON r.session_id = j.session_id
        WHERE j.golden_question_passed = TRUE 
        AND j.trap_question_passed = TRUE
        GROUP BY 
        r.test_type, 
        r.condition, 
        r.feature,
        r.present_word,
        r.absent_word,
        r.talker
    """

    raw_df_humans = pd.read_sql_query(query_humans, conn_humans)
    conn_humans.close()

    return raw_df_humans


def get_individual_human_ratings() -> pd.DataFrame:
    humans_sqlite_database_path = "data/human_ratings.db"
    conn_humans = sqlite3.connect(humans_sqlite_database_path)
    query_humans = """
        -- Query to get every individual human rating, scored 1 if right and -1 if wrong.
        SELECT
        r.session_id,
        r.test_type,
        r.condition,
        r.feature,
        r.present_word,
        r.absent_word,
        r.talker,
        CASE WHEN r.answer = r.present_word THEN 1.0 ELSE -1.0 END AS score
        FROM ratingsession r
        INNER JOIN job j
            ON r.session_id = j.session_id
        WHERE j.golden_question_passed = TRUE
        AND j.trap_question_passed = TRUE
    """

    raw_df_humans = pd.read_sql_query(query_humans, conn_humans)
    conn_humans.close()

    return raw_df_humans


def get_table_name(model_name: str, with_carrier_phrase: bool) -> str:
    """Build the name of the SQLite table storing the results of a model.

    Args:
        model_name: Hugging Face name of the ASR model.
        with_carrier_phrase: Whether the audio contains the carrier phrase.

    Returns:
        A table name such as ``results_openai_whisper_large_v3_without_carrier_phrase``.
    """
    sanitized_model_name = (
        model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    )
    carrier_suffix = "with" if with_carrier_phrase else "without"
    return f"results_{sanitized_model_name}_{carrier_suffix}_carrier_phrase"


def get_ASR_ratings(
    specific_model_name: str,
    with_carrier_phrase: bool = False,
    table_name: str | None = None,
    database_path: Path = ASR_DATABASE_PATH,
) -> pd.DataFrame:
    conn_ASR = sqlite3.connect(database_path)
    query_ASR = f"""
        -- Query to get the intelligibility from an ASR model.
        SELECT
        r.test_type,
        r.condition,
        r.feature,
        r.present_word,
        r.absent_word,
        r.talker,
        COUNT(*) AS total_count,
        SUM(CASE WHEN r.predicted_word = r.present_word THEN 1.0 ELSE -1.0 END) / COUNT(*) AS intelligibility,
        SUM(CASE WHEN r.predicted_word = r.present_word THEN 1.0 ELSE -1.0 END) AS NUM_hard,
        SUM(CASE WHEN r.predicted_word = r.present_word THEN r.probability_generated_text ELSE -r.probability_generated_text END) / SUM(r.probability_generated_text) AS predicted_intelligibility,
        SUM(CASE WHEN r.predicted_word = r.present_word THEN r.probability_generated_text ELSE -r.probability_generated_text END) as NUM,
        SUM(r.probability_generated_text) AS DEN,
        SUM(CASE WHEN r.predicted_word = r.present_word THEN ln(r.probability_generated_text) ELSE ln(1-r.probability_generated_text) END) as present_word_log_proba,
        SUM(CASE WHEN r.predicted_word != r.present_word THEN ln(r.probability_generated_text) ELSE ln(1-r.probability_generated_text) END) as absent_word_log_proba
        FROM {table_name or get_table_name(specific_model_name, with_carrier_phrase)} r
        GROUP BY
        r.test_type,
        r.condition,
        r.feature,
        r.present_word,
        r.absent_word,
        r.talker
    """

    raw_df_ASR = pd.read_sql_query(query_ASR, conn_ASR)
    conn_ASR.close()

    return raw_df_ASR
