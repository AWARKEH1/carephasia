"""Linguistic features used by CarePhasia.

Four transcript-level features are computed:

* ``average_tree_depth``      - mean depth of the dependency tree per sentence (syntax)
* ``average_lexical_density`` - mean percentage of function words per sentence (morphology)
* ``word_stutter_count``      - number of sentences containing an immediate word repetition
* ``syllable_stutter_ratio``  - share of out-of-vocabulary tokens that contain a repeated
                                letter sequence (e.g. "ba-ba-ball" -> "bababall")

Note: in the original project the morphology feature was called "lexical density",
but it actually measures the proportion of *function* words (ADP, AUX, CONJ, DET,
PRON, SCONJ). The name is kept for compatibility with the training data.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Iterable

FEATURE_COLUMNS = [
    "average_tree_depth",
    "average_lexical_density",
    "word_stutter_count",
    "syllable_stutter_ratio",
]

FUNCTION_POS = {"ADP", "AUX", "CONJ", "CCONJ", "DET", "PRON", "SCONJ"}

_WORD_REPETITION = re.compile(r"\b(\w+)\s+(\1\s+)*\1\b", flags=re.IGNORECASE)
_SYLLABLE_REPETITION = re.compile(r"(\w+)\1+")


@lru_cache(maxsize=1)
def get_nlp(model: str = "en_core_web_sm"):
    """Load the spaCy pipeline once."""
    import spacy

    return spacy.load(model)


# --- Syntax -------------------------------------------------------------------

def _node_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_node_depth(child) for child in children)


def tree_depth(sentence: str) -> int:
    """Maximum depth of the dependency tree of ``sentence`` (0 if no root)."""
    doc = get_nlp()(sentence)
    root = next((tok for tok in doc if tok.dep_ == "ROOT"), None)
    return 0 if root is None else _node_depth(root)


# --- Morphology ---------------------------------------------------------------

def lexical_density(sentence: str) -> float:
    """Percentage of function words in ``sentence``."""
    if not isinstance(sentence, str) or not sentence.strip():
        return 0.0
    doc = get_nlp()(sentence)
    if len(doc) == 0:
        return 0.0
    n_function = sum(tok.pos_ in FUNCTION_POS for tok in doc)
    return 100.0 * n_function / len(doc)


# --- Disfluencies -------------------------------------------------------------

def word_stutter_count(sentences: Iterable[str]) -> int:
    """Number of sentences with a consecutive repetition of the same word."""
    return sum(
        1 for s in sentences if isinstance(s, str) and _WORD_REPETITION.search(s)
    )


def default_is_known_word(word: str) -> bool:
    """English vocabulary check based on ``wordfreq`` (replaces the original
    ``en_US-large.txt`` dictionary, which was not versioned)."""
    from wordfreq import zipf_frequency

    return zipf_frequency(word.lower(), "en") > 0


def syllable_stutter_ratio(
    sentences: Iterable[str],
    is_known_word: Callable[[str], bool] = default_is_known_word,
) -> float:
    """Share of unknown tokens that contain a repeated letter sequence."""
    unknown = [
        word
        for s in sentences
        for word in str(s).split()
        if not is_known_word(word)
    ]
    if not unknown:
        return 0.0
    repeated = [w for w in unknown if _SYLLABLE_REPETITION.search(w)]
    return len(repeated) / len(unknown)


# --- Full feature vector ------------------------------------------------------

def clean_sentence(text: str) -> str:
    """Keep letters and spaces only (same rule as the original pipeline)."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def segment(text: str) -> list[str]:
    """Split raw text into non-empty sentences with spaCy."""
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def extract_features(text: str) -> dict[str, float]:
    """Compute the four CarePhasia features for one transcript."""
    sentences = segment(text)
    if not sentences:
        raise ValueError("The transcript is empty.")
    depths = [tree_depth(s) for s in sentences]
    densities = [lexical_density(s) for s in sentences]
    cleaned = [clean_sentence(s) for s in sentences]
    return {
        "average_tree_depth": sum(depths) / len(depths),
        "average_lexical_density": sum(densities) / len(densities),
        "word_stutter_count": float(word_stutter_count(cleaned)),
        "syllable_stutter_ratio": syllable_stutter_ratio(cleaned),
    }
