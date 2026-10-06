"""Default settings of the correlation analysis, overridable from the command line."""

MODELS = [
    "CohereLabs/cohere-transcribe-03-2026",
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "distil-whisper/distil-large-v3.5",
    "nvidia/canary-qwen-2.5b",
    "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen/Qwen3-ASR-0.6B-hf",
    "ibm-granite/granite-4.0-1b-speech",
    "ibm-granite/granite-speech-4.1-2b",
    "ibm-granite/granite-speech-4.1-2b-plus",
]
CARRIER_PHRASE = [False]  # can be set to [True, False] to pool both conditions
CARRIER_PHRASE_CHOICES = {
    "without": [False],
    "with": [True],
    "both": [False, True],
}

MERGE_CHOICES = [
    "test_type",
    "condition",
    "feature",
    "present_word",
    "absent_word",
    "talker",
    # "gender",
]

# Columns defining one item on which ASR and human ratings are compared
MERGE = [
    # "test_type",
    "condition",
    "feature",
    # "present_word",
    # "absent_word",
    # "talker",
]

# Item definition used by the Steiger-Williams test script
STEIGER_MERGE = [
    "test_type",
    "condition",
    "feature",
    "present_word",
    "absent_word",
    "talker",
]

METRICS = ("SOFT_VOTING", "HARD_VOTING", "LOG_LIKELIHOOD_RATIO")
METRIC = "HARD_VOTING"

# The log-likelihood ratio is computed per stimulus-level item before being
# averaged over the MERGE columns
LLR_ITEM_COLUMNS = [
    "test_type",
    "condition",
    "feature",
    "present_word",
    "absent_word",
    "talker",
]

X_COLUMN = "predicted_intelligibility"
Y_COLUMN = "intelligibility_human"

# The significance level for the tests
ALPHA = 0.05
