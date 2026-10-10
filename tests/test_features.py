"""Unit tests for the disfluency features (no spaCy model needed)."""

from carephasia.features import clean_sentence, syllable_stutter_ratio, word_stutter_count

KNOWN = {"the", "girl", "went", "to", "ball", "she", "was", "happy", "and"}


def is_known(word: str) -> bool:
    return word.lower() in KNOWN


def test_word_stutter_counts_sentences_with_repetition():
    sentences = ["the the girl went", "she was happy", "and and and the ball"]
    assert word_stutter_count(sentences) == 2


def test_word_stutter_ignores_non_strings():
    assert word_stutter_count([None, 3.0, "she was happy"]) == 0


def test_syllable_stutter_ratio():
    # unknown words: "bababall" (repetition) and "xyz" (no repetition)
    sentences = ["the girl went to the bababall", "xyz"]
    assert syllable_stutter_ratio(sentences, is_known_word=is_known) == 0.5


def test_syllable_stutter_ratio_without_unknown_words():
    assert syllable_stutter_ratio(["the girl was happy"], is_known_word=is_known) == 0.0


def test_clean_sentence():
    assert clean_sentence("Once, long... place far 2!") == "Once long place far"
