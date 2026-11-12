import pytest

from wanderlust.llm.types import UserTurn
from wanderlust.qa.prompt import (
    NOT_COVERED,
    SYSTEM_PROMPT,
    build_conversation,
    is_not_covered,
)
from wanderlust.qa.sources import number_sources


def test_system_prompt_states_the_grounding_rules():
    assert "Use only information stated in the passages" in SYSTEM_PROMPT
    assert "[2]" in SYSTEM_PROMPT  # shows the citation format
    assert NOT_COVERED in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT


def test_conversation_contains_passages_then_question(searcher):
    sources = number_sources(searcher.search("fish soup", k=2).results)
    convo = build_conversation("  Where can I eat fish soup?  ", "Examplestad", sources)

    assert convo.system == SYSTEM_PROMPT
    (turn,) = convo.turns
    assert isinstance(turn, UserTurn)
    assert turn.text.startswith("Destination: Examplestad\n\n<passages>\n")
    assert turn.text.endswith("</passages>\n\nQuestion: Where can I eat fish soup?")
    assert turn.text.index("<passages>") < turn.text.index("Question:")
    assert '<passage number="2"' in turn.text


def test_empty_question_is_rejected(searcher):
    with pytest.raises(ValueError):
        build_conversation("   ", "Examplestad", [])


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The guide doesn't cover this. It does list two restaurants.", True),
        ("  the guide doesn't cover this.", True),
        ("Fish soup is served at Harbour Kitchen [1].", False),
        ("Mostly yes, but the guide doesn't cover this part.", False),
    ],
)
def test_is_not_covered(answer, expected):
    assert is_not_covered(answer) is expected
