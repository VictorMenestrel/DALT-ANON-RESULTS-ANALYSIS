import torch

# pyrefly: ignore [missing-import]
from nemo.collections.speechlm2.models import SALM

from models.utils import normalize_text


def load_model(model_name: str | None = None):
    """
    Load the canary speech model and its associated tokenizer.
    Args:
        model_name: Optional; the name of the model to load. If None, defaults to "nvidia/canary-qwen-2.5b".

    Returns:
        device: The device on which the model is loaded (cuda or cpu).
        _ None: Placeholder for processor (not used for canary).
        tokenizer: The tokenizer associated with the model.
        model: The loaded canary speech model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name is None:
        model_name = "nvidia/canary-qwen-2.5b"
    model = SALM.from_pretrained(
        model_name,
    ).to(device)
    tokenizer = model.tokenizer

    return device, None, tokenizer, model, None


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
        model: Speech model
        prompt: Prompt for the model (not used for Qwen)
        word_a: First allowed word/token
        word_b: Second allowed word/token
        carrier_phrase: Optional carrier phrase to prepend to audio inputs for ASR evaluation (not used for Qwen)

    Returns:
        dict: Contains 'word_a' and 'word_b' as the decoded predictions
    """

    # Get token IDs for word_a and word_b using variants (currently not used)
    def token_ids_for_word(word: str):
        variants = [word]
        if carrier_phrase:
            variants = [f"{carrier_phrase}{word}" for word in variants]
        token_sequences = []
        seen = set()
        for variant in variants:
            ids = tokenizer.text_to_ids(variant)
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

    eos_token_id = tokenizer.eos_id

    # We want to force the model to generate either the token for word_a or the token for word_b
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        generated = input_ids.tolist()[:]
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

    with torch.no_grad():
        generated_ids = model.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [audio_path],
                    }
                ]
            ],
            max_new_tokens=max_candidate_len + 1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=2,
            do_sample=False,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_text = tokenizer.ids_to_text(generated_ids[0].cpu())
    generated_text = normalize_text(generated_text[0])

    probability_generated_text = torch.max(
        torch.softmax(generated_ids.scores[first_differing_index], dim=-1)
    ).item()

    # Ensure the generated text is either word_a or word_b
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
