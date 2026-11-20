import pytest

from src.grammar_hints import hints


def pairs(text):
    return [(h["previous"], h["word"], h["suggestion"]) for h in hints(text)]


@pytest.mark.parametrize("text, expected", [
    ("Pupunta daw sila.", [("Pupunta", "daw", "raw")]),     # vowel -> raw
    ("Bukas raw sila pupunta.", [("Bukas", "raw", "daw")]),  # consonant -> daw
    ("Ikaw din ba?", [("Ikaw", "din", "rin")]),              # w -> rin
    ("Siya daw ang nanalo.", [("Siya", "daw", "raw")]),      # a -> raw
    ("Ganoon rin.", [("Ganoon", "rin", "din")]),
])
def test_suggests_the_formal_form(text, expected):
    assert pairs(text) == expected


@pytest.mark.parametrize("text", [
    "Maganda raw si Aleli.", "Masarap din ang luto.", "Ako rin.", "Bukas daw.",
    "Daw sabi nila.",            # nothing before it
    "Bukas. Raw ang isda.",      # new sentence
    "Lahad daw nila.",           # sources differ after a final d
])
def test_no_suggestion_when_already_formal_or_unclear(text):
    assert hints(text) == []


def test_keeps_capital_letter():
    assert hints("Pupunta. Ako Din.")[0]["suggestion"] == "Rin"
