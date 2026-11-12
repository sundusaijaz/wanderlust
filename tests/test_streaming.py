import pytest

from wanderlust.streaming import WordWrapper


def wrap(pieces, width=20) -> str:
    out: list[str] = []
    wrapper = WordWrapper(out.append, width)
    for piece in pieces:
        wrapper.feed(piece)
    wrapper.close()
    return "".join(out)


SENTENCE = "The heritage tram runs every fifteen minutes to the lighthouse."


def test_wraps_between_words():
    lines = wrap([SENTENCE]).split("\n")
    assert all(len(line) <= 20 for line in lines)
    assert " ".join(lines) == SENTENCE


def test_fragments_splitting_words_are_rejoined():
    assert wrap(["Ligh", "thouse is o", "pen [1", "]."], width=40) == "Lighthouse is open [1]."


def test_keeps_newlines_and_list_indentation():
    assert (
        wrap(["Tips:\n- Go early\n  - before 9\n"], width=40) == "Tips:\n- Go early\n  - before 9\n"
    )


def test_long_word_gets_its_own_line():
    assert wrap(["see https://example.org/a/very/long/path now"], width=20) == (
        "see\nhttps://example.org/a/very/long/path\nnow"
    )


def test_repeated_spaces_collapse():
    assert wrap(["a    b"]) == "a b"


def test_rejects_tiny_width():
    with pytest.raises(ValueError):
        WordWrapper(print, 5)
