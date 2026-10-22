import pytest

from wolof_normalizer import normalize, normalize_with_changes, normalize_word
from wolof_normalizer.cli import main


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dieuredieuf", "jërëjëf"),
        ("nanga def", "na nga def"),
        ("Nanga def ?", "Na nga def ?"),
        ("mangui fi rek", "maa ngi fi rekk"),
        ("dafa beug wakh", "dafa bëgg wax"),
    ],
)
def test_lexicon(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("kham", "xam"),      # kh -> x
        ("khale", "xale"),
        ("lou", "lu"),        # ou -> u
        ("gnou", "ñu"),       # gn + ou
        ("tchi", "ci"),       # tch -> c
        ("bou", "bu"),
        ("dokh", "dox"),      # via le lexique
    ],
)
def test_word_normalization(word, expected):
    assert normalize_word(word)[0] == expected


def test_unknown_words_are_kept():
    assert normalize("Ousmane dafa dem Paris") == "Ousmane dafa dem Paris"


def test_official_words_are_kept():
    assert normalize("Jërëjëf, maa ngi fi") == "Jërëjëf, maa ngi fi"


def test_case_and_punctuation_are_preserved():
    assert normalize("WAKH bou bakh!") == "WAX bu baax!"


def test_changes_are_reported():
    _, changes = normalize_with_changes("kham nga ko ?")
    assert [(c.original, c.normalized, c.reason) for c in changes] == [("kham", "xam", "rules")]


def test_cli(capsys):
    assert main(["nanga", "def"]) == 0
    assert capsys.readouterr().out == "na nga def\n"
