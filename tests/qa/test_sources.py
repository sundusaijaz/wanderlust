from wanderlust.qa.sources import format_sources, number_sources
from wanderlust.retrieval.search import SearchResult


def test_sources_are_numbered_in_ranking_order(searcher):
    results = searcher.search("lighthouse sunset", k=3).results
    sources = number_sources(results)
    assert [s.number for s in sources] == [1, 2, 3]
    assert [s.chunk for s in sources] == [r.chunk for r in results]


def test_duplicate_passages_get_one_number(searcher):
    (first, *_) = searcher.search("fish soup", k=1).results
    duplicate = SearchResult(chunk=first.chunk, score=0.1)
    assert len(number_sources([first, duplicate])) == 1


def test_label_and_url(searcher):
    lighthouse = next(r for r in searcher.search("lighthouse", k=5).results if r.chunk.listing_name)
    (source,) = number_sources([lighthouse])
    assert source.label == "Examplestad > See · Old Lighthouse"
    assert source.url == "https://en.wikivoyage.org/wiki/Examplestad#See"


def test_format_sources_wraps_each_passage(searcher):
    sources = number_sources(searcher.search("tram", k=2).results)
    text = format_sources(sources)
    assert text.count("<passage ") == 2
    assert text.count("</passage>") == 2
    assert '<passage number="1" location="Examplestad' in text
    assert sources[0].chunk.text in text


def test_location_attribute_is_escaped(searcher):
    from dataclasses import replace

    (result,) = searcher.search("tram", k=1).results
    tricky = replace(result, chunk=result.chunk.model_copy(update={"section": 'A "quoted" <b>'}))
    text = format_sources(number_sources([tricky]))
    assert "A &quot;quoted&quot; &lt;b>" in text
