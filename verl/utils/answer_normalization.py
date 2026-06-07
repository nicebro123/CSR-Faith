import re
import string


_TAG_RE = re.compile(r"<[^>]+>")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Normalize short answer text for counterfactual answer comparison."""
    if text is None:
        return ""

    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        text = answer_match.group(1)

    text = _TAG_RE.sub(" ", str(text))
    text = text.strip().lower()

    # Drop common multiple-choice prefixes without assuming the dataset uses letters.
    text = re.sub(r"^\s*(?:option|choice|answer)\s*[:\-]?\s*", "", text)
    text = re.sub(r"^\s*[\(\[]?[a-z][\)\].:]\s+", "", text)

    translator = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answers_equal(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)
