import numpy as np
import torch
from huggingface_hub import login
from torch import nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)
from transformers.audio_utils import load_audio

login()


def load_model(model_name: str | None = None):
    """
    Load the Phoneme speech model and its associated processor and tokenizer.
    Args:
        model_name: Optional; the name of the model to load. If None, defaults to "facebook/wav2vec2-lv-60-espeak-cv-ft".

    Returns:
        device: The device on which the model is loaded (cuda or cpu).
        processor: The processor associated with the model.
        tokenizer: The tokenizer associated with the model.
        model: The loaded Phoneme speech model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name is None:
        model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, device_map=device)

    return device, processor, tokenizer, model, None


def generate_tokens_from_audio(
    audio_path: str,
    device,
    processor,
    tokenizer,
    model,
    prompt,
    word_a: str,
    word_b: str,
    carrier_phrase: str | None = None,
):
    """
    Generate token predictions from an audio file path.

    The generation is constrained with prefix_allowed_tokens_fn so that the
    model can only emit the token associated with word_a or the token associated
    with word_b.

    Args:
        audio_path: Path to the audio file
        device: Device to run inference on (cuda or cpu)
        processor: Audio processor from AutoProcessor
        tokenizer: Tokenizer from processor
        model: Speech model
        prompt: Prompt template from tokenizer
        word_a: First allowed word/token
        word_b: Second allowed word/token

    Returns:
        dict: Contains 'word_a' and 'word_b' as the decoded predictions
    """
    # Load audio file
    audio_input = load_audio(audio_path)

    # Process audio input
    model_inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").to(
        device, dtype=model.dtype
    )

    # Generate tokens with prefix_allowed_tokens_fn constraint
    with torch.no_grad():
        logits = model(model_inputs.input_values).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long)
    ctc_loss = nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)

    if carrier_phrase:
        word_a_input = f"{carrier_phrase}{word_a}"
        word_b_input = f"{carrier_phrase}{word_b}"
    else:
        word_a_input = word_a
        word_b_input = word_b

    # I don't know why, but using the text instead of phonemes representation outputs better results.
    target_A_ids = torch.tensor(tokenizer.encode(word_a_input), dtype=torch.long)
    target_A_lengths = torch.tensor([target_A_ids.shape[0]], dtype=torch.long)
    loss_A = ctc_loss(
        log_probs, target_A_ids.unsqueeze(0), input_lengths, target_A_lengths
    ).item()

    target_B_ids = torch.tensor(tokenizer.encode(word_b_input), dtype=torch.long)
    target_B_lengths = torch.tensor([target_B_ids.shape[0]], dtype=torch.long)
    loss_B = ctc_loss(
        log_probs, target_B_ids.unsqueeze(0), input_lengths, target_B_lengths
    ).item()

    # Get the probability of the generated text being word_a or word_b
    scores = np.array([-loss_A, -loss_B])
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()

    prob_A_relative = probabilities[0]
    prob_B_relative = probabilities[1]

    return {
        "word_a": word_a,
        "word_b": word_b,
        "generated_text": word_a if loss_A < loss_B else word_b,
        "probability_generated_text": prob_A_relative
        if loss_A < loss_B
        else prob_B_relative,
    }
