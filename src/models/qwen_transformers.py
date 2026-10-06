import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.audio_utils import load_audio

from models.utils import normalize_text


def load_model(model_name: str | None = None):
    """
    Load the Qwen speech model and its associated processor and tokenizer.
    Args:
        model_name: Optional; the name of the model to load. If None, defaults to "Qwen/Qwen3-ASR-1.7B-hf".

    Returns:
        device: The device on which the model is loaded (cuda or cpu).
        processor: The processor associated with the model.
        tokenizer: The tokenizer associated with the model.
        model: The loaded Qwen speech model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name is None:
        model_name = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model: AutoModelForMultimodalLM = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        device_map=device if device == "cuda" else None,
        dtype=dtype,
    )

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
    model_inputs = processor.apply_transcription_request(
        audio=audio_input, language="English"
    ).to(device, dtype=model.dtype)

    num_input_tokens = model_inputs.data["input_ids"].shape[-1]

    # Get token IDs for word_a and word_b using variants (currently not used)
    def token_ids_for_word(word: str):
        variants = [word]
        if carrier_phrase:
            variants = [f"{carrier_phrase}{word}" for word in variants]
        token_sequences = []
        seen = set()
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            key = tuple(ids)
            if ids and key not in seen:
                token_sequences.append(ids)
                seen.add(key)
        return token_sequences

    candidates = []
    for word in (word_a, word_b):
        for ids in token_ids_for_word(word):
            candidates.append((word, ids))

    # Get the first index position where the id differs between the two candidates, so we can get the probability of the generated token
    first_differing_index = None
    for i in range(min(len(candidates[0][1]), len(candidates[1][1]))):
        if candidates[0][1][i] != candidates[1][1][i]:
            first_differing_index = i
            break

    eos_token_id = tokenizer.eos_token_id

    # We want to force the model to generate either the token for word_a or the token for word_b
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        generated = input_ids.tolist()[
            num_input_tokens:
        ]  # It seems there is always num_input_tokens prefix tokens for qwen.
        allowed = set()

        for _, token_ids in candidates:
            prefix_len = len(generated)
            if prefix_len < len(token_ids) and token_ids[:prefix_len] == generated:
                allowed.add(token_ids[prefix_len])
            elif (
                prefix_len == len(token_ids)
                and generated == token_ids
                and eos_token_id is not None
            ):
                allowed.add(eos_token_id)

        if not allowed:
            return [eos_token_id] if eos_token_id is not None else [0]
        return sorted(allowed)

    # Determine the maximum candidate token length for generation to ensure we generate
    # enough tokens to cover all candidates and avoid unnecessary generation steps
    max_candidate_len = max(len(ids) for _, ids in candidates)

    # Generate tokens with prefix_allowed_tokens_fn constraint
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_candidate_len + 1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=2,
            do_sample=False,
            num_return_sequences=1,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(
        generated_ids.sequences[:, num_input_tokens:-1],
        return_format="transcription_only",
    )[0]
    generated_text = normalize_text(generated_text)

    probability_generated_text = torch.max(
        torch.softmax(generated_ids.scores[first_differing_index], dim=-1)
    ).item()

    # Ensure the generated text is either word_a or word_b, take in account carrier phrase if present
    if carrier_phrase:
        if generated_text.startswith(carrier_phrase):
            generated_text = generated_text[len(carrier_phrase) :]
        else:
            raise ValueError(
                f"Generated text '{generated_text}' does not start with the expected carrier phrase '{carrier_phrase}'."
            )

    else:
        if generated_text not in [word_a, word_b]:
            raise ValueError(
                f"Generated text '{generated_text}' is not one of the allowed words: '{word_a}' or '{word_b}'."
            )

    return {
        "word_a": word_a,
        "word_b": word_b,
        "generated_text": generated_text,
        "probability_generated_text": probability_generated_text,
    }
