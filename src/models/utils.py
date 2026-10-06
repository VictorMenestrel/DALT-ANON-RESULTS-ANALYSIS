import re


def normalize_text(s: str) -> str:
    """
    Normalize a string by converting it to lowercase, removing non-alphanumeric characters,
    and collapsing multiple spaces into a single space.

    Args:
        s (str): The input string to normalize.

    Returns:
        str: The normalized string.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
