import pytest

from wanderlust.agent.guides import GuideLibrary, SourceBook, guide_tools, index_opener
from wanderlust.llm.types import ToolCall
from wanderlust.retrieval.index import IndexStore
from wanderlust.retrieval.search import SearchMode
from wanderlust.tools.http import ToolError
from wanderlust.wikivoyage.client import WikivoyageError


def search(registry, **arguments):
    return registry.run(ToolCall(id="c1", name="search_guide", arguments=arguments))


def test_search_returns_numbered_passages(library, book):
    tools = guide_tools(library, book)
    result = search(tools, destination="Examplestad", query="lighthouse at sunset", limit=3)
    assert not result.is_error
    assert result.content.startswith("Passages from the Examplestad guide:")
    assert result.content.count("<passage ") == 3
    assert '<passage number="1"' in result.content
    assert len(book) == 3
    assert "Old Lighthouse" in {s.chunk.listing_name for s in book.sources}


def test_numbers_are_stable_across_searches(library, book):
    tools = guide_tools(library, book)
    search(tools, destination="Examplestad", query="lighthouse", limit=2)
    first = {s.chunk.id: s.number for s in book.sources}
    search(tools, destination="Examplestad", query="fish soup harbour", limit=4)
    for source in book.sources:
        if source.chunk.id in first:
            assert source.number == first[source.chunk.id]
    assert [s.number for s in book.sources] == list(range(1, len(book) + 1))


def test_guides_are_opened_once_and_aliases_share_them(library, book, guide_loader):
    tools = guide_tools(library, book)
    search(tools, destination="Examplestad", query="tram")
    search(tools, destination="example town", query="harbour")  # redirects to Examplestad
    search(tools, destination="examplestad", query="soup")
    search(tools, destination="Example Town", query="soup")
    assert library.destinations == ["Examplestad"]
    assert library.searcher("example town") is library.searcher("Examplestad")
    pages = [r.url.params["page"] for r in guide_loader.requests if "page" in r.url.params]
    assert pages == ["Examplestad", "example town"]  # one lookup to resolve the redirect


def test_saved_indexes_are_reused(guide_loader, fake_embedder, tmp_path):
    store = IndexStore(tmp_path / "data")
    built = []
    opener = index_opener(store, guide_loader, fake_embedder, on_build=built.append)
    opener("Examplestad")
    opener("Examplestad")
    assert built == ["Examplestad"]
    assert store.load("en", "Examplestad") is not None


def test_unknown_guide_is_a_tool_error_with_suggestion(library, book):
    result = search(guide_tools(library, book), destination="Exampleshire", query="food")
    assert result.is_error
    assert "no Wikivoyage guide called 'Exampleshire'" in result.content
    assert "Did you mean 'Examplestad'?" in result.content


def test_network_problems_are_tool_errors(book):
    def broken(destination):
        raise WikivoyageError("connection refused")

    result = search(guide_tools(GuideLibrary(broken), book), destination="Lisbon", query="food")
    assert result.is_error and "Could not open the Lisbon guide" in result.content


def test_guide_limit(library, book):
    library.max_guides = 1
    tools = guide_tools(library, book)
    search(tools, destination="Examplestad", query="tram")
    result = search(tools, destination="Bigville", query="museums")
    assert result.is_error
    assert "already uses 1 guides (Examplestad)" in result.content
    assert not search(tools, destination="Examplestad", query="soup").is_error


def test_nothing_found_is_not_an_error(library, book):
    result = search(
        guide_tools(library, book, mode=SearchMode.KEYWORD),
        destination="Examplestad",
        query="snorkelling",
    )
    assert not result.is_error
    assert "No passages in the Examplestad guide match 'snorkelling'" in result.content
    assert len(book) == 0


def test_vector_search_without_embeddings_is_a_tool_error(guide_loader, tmp_path, book):
    library = GuideLibrary(index_opener(IndexStore(tmp_path / "plain"), guide_loader, None))
    tools = guide_tools(library, book, mode="vector")
    result = search(tools, destination="Examplestad", query="tram")
    assert result.is_error and "Vector search is unavailable" in result.content


def test_bad_arguments(library, book):
    result = search(guide_tools(library, book), destination="Examplestad", query="x", limit=99)
    assert result.is_error
    assert "query" in result.content and "limit" in result.content


def test_source_book_select_skips_unknown_numbers(library, book):
    search(guide_tools(library, book), destination="Examplestad", query="tram", limit=2)
    assert [s.number for s in book.select([2, 9, 1])] == [2, 1]
    assert 1 in book and 9 not in book and book.get(9) is None


def test_library_rejects_a_zero_limit():
    with pytest.raises(ValueError):
        GuideLibrary(lambda d: None, max_guides=0)


def test_library_errors_are_tool_errors(library):
    library.max_guides = 1
    library.searcher("Examplestad")
    with pytest.raises(ToolError):
        library.searcher("Bigville")


def test_empty_book():
    book = SourceBook()
    assert len(book) == 0 and book.sources == []
