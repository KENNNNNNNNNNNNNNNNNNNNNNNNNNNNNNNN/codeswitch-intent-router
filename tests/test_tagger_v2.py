import pandas as pd
import pytest

from src import tagger as v1
from src.tagger_v2 import analyse, tag_sentence, tokenize


def tags(text):
    return [(t.text, t.tag) for t in tag_sentence(text)]


def test_users_example_sentence():
    assert tags("Ako'y hindi na makapag wait kumain 'cause I am gutom na") == [
        ("Ako", "TL"), ("'y", "TL"), ("hindi", "TL"), ("na", "TL"), ("makapag", "TL"),
        ("wait", "EN"), ("kumain", "TL"), ("'cause", "EN"), ("I", "EN"), ("am", "EN"),
        ("gutom", "TL"), ("na", "TL")]


@pytest.mark.parametrize("text, expected", [
    ("Ako'y", [("Ako", "ako"), ("'y", "ay")]),
    ("ika'y", [("ika", "ika"), ("'y", "ay")]),
    ("Ako't", [("Ako", "ako"), ("'t", "at")]),
    ("Ano'ng", [("Ano", "ano"), ("'ng", "ang")]),
    ("sa'yo", [("sa", "sa"), ("'yo", "iyo")]),
    ("sa'kin", [("sa", "sa"), ("'kin", "akin")]),
    ("'to", [("'to", "ito")]),
    ("'Di", [("'Di", "hindi")]),
    ("'wag", [("'wag", "huwag")]),
    ("'cause", [("'cause", "because")]),
])
def test_contractions_are_split_and_expanded(text, expected):
    assert [(t.text, t.norm) for t in tag_sentence(text)] == expected


def test_curly_apostrophes_work_like_straight_ones():
    assert tags("Ako’y gutom") == tags("Ako'y gutom")


@pytest.mark.parametrize("word", ["don't", "I'm", "registrar's", "let's", "you're", "Ma'am"])
def test_english_apostrophe_words_stay_one_english_token(word):
    assert tags(word) == [(word, "EN")]


@pytest.mark.parametrize("word, tag", [
    ("nagutom", "TL"), ("kumain", "TL"), ("nakakapagod", "TL"), ("mag-usap", "TL"),
    ("nag-wait", "EN"), ("i-request", "EN"), ("Nagre-reflect", "EN"), ("naka-online", "EN"),
])
def test_root_decides_affixed_words(word, tag):
    assert tags(word)[0][1] == tag


def test_bare_prefix_is_tagalog():
    assert tags("makapag wait")[0] == ("makapag", "TL")


def test_may_is_decided_by_the_next_word():
    assert tags("May I ask")[0][1] == "EN"
    assert tags("kasi may unpaid balance")[1][1] == "TL"


def test_loanwords_common_in_english_are_english():
    assert dict(tags("may class bukas"))["class"] == "EN"
    assert dict(tags("traffic kanina"))["traffic"] == "EN"


def test_unknown_words_are_unk_and_left_out_of_density():
    a = analyse("zxqvbn ako")
    assert [t.tag for t in a["tokens"]] == ["UNK", "TL"]
    assert a["n_unknown"] == 1 and a["density"] == 0.0


def test_capitalised_unknown_word_mid_sentence_is_a_name():
    assert dict(tags("Kasama ko si Xylandra"))["Xylandra"] == "NE"


def test_tokenize_keeps_leading_apostrophe():
    assert tokenize("'Wag na, 'to lang.") == ["'Wag", "na", "'to", "lang"]


def test_v2_matches_v1_on_the_registrar_dataset():
    """Day 1-7 densities stay the same with the new tagger."""
    texts = pd.read_csv("data/taglish_dataset.csv")["text"]
    for text in texts:
        assert [v1.tag_token(t) for t in v1.tokenize(text)] == [t.tag for t in tag_sentence(text)]
