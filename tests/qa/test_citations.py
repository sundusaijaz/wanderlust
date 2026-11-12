import pytest

from wanderlust.qa.citations import check_citations, cited_numbers
from wanderlust.qa.sources import number_sources


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No citations here.", []),
        ("Trams run often [2].", [2]),
        ("Cheap [1][3] and central [1].", [1, 3]),
        ("See [2, 4] or [4,1].", [2, 4, 1]),
        ("Ignore [a] and [1-2] and [].", []),
        ("Arrays like list[0] look similar [5].", [0, 5]),
    ],
)
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


def test_check_citations_splits_cited_unknown_and_uncited(searcher):
    sources = number_sources(searcher.search("lighthouse tram fish", k=3).results)
    report = check_citations("Climb the lighthouse [2]. The tram [2][9] is slow.", sources)

    assert [s.number for s in report.cited] == [2]
    assert report.unknown == [9]
    assert [s.number for s in report.uncited] == [1, 3]


def test_answer_without_citations(searcher):
    sources = number_sources(searcher.search("tram", k=2).results)
    report = check_citations("The guide doesn't cover this.", sources)
    assert report.cited == [] and report.unknown == []
    assert len(report.uncited) == 2
