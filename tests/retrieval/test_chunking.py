import pytest

from wanderlust.retrieval.chunking import OVERVIEW, ChunkKind, chunk_article, split_text
from wanderlust.wikivoyage.models import ListingType
from wanderlust.wikivoyage.parser import parse_article


@pytest.fixture
def article(examplestad_wikitext):
    return parse_article(
        examplestad_wikitext, title="Examplestad", language="en", page_id=42, revision_id=7
    )


def test_split_text_packs_paragraphs():
    text = "First para.\nSecond para.\nThird para."
    assert split_text(text, 100) == ["First para.\nSecond para.\nThird para."]
    assert split_text(text, 25) == ["First para.\nSecond para.", "Third para."]


def test_split_text_breaks_long_paragraphs_at_sentences():
    text = "One sentence here. Another sentence there. A third one follows."
    pieces = split_text(text, 45)
    assert pieces == ["One sentence here.\nAnother sentence there.", "A third one follows."]
    assert all(len(piece) <= 45 for piece in pieces)


def test_split_text_breaks_very_long_sentences_at_words():
    text = " ".join(["word"] * 100)
    pieces = split_text(text, 120)
    assert all(len(piece) <= 120 for piece in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_text_ignores_blank_input():
    assert split_text("  \n\n ", 100) == []


def test_article_chunks_cover_summary_text_and_listings(article):
    chunks = chunk_article(article)
    sections = {chunk.section for chunk in chunks}
    assert {OVERVIEW, "Understand", "By tram", "See", "Eat", "Go next"} <= sections

    listings = [c for c in chunks if c.kind is ChunkKind.LISTING]
    assert [c.listing_name for c in listings] == [
        "Old Lighthouse",
        "Clock Square",
        "Harbour Kitchen",
        "The Salty Mug",
        "Otherville",
    ]


def test_listing_chunk_contains_practical_details(article):
    lighthouse = next(c for c in chunk_article(article) if c.listing_name == "Old Lighthouse")
    assert lighthouse.listing_type is ListingType.SEE
    assert lighthouse.text.startswith("Old Lighthouse (Vieux Phare): A striped 19th-century")
    assert "Price: Adult 5 coins, children free" in lighthouse.text
    assert "Hours: 09:00-18:00" in lighthouse.text
    assert (lighthouse.lat, lighthouse.lon) == (12.3456, -7.6543)


def test_chunks_keep_attribution_and_breadcrumb(article):
    tram = next(c for c in chunk_article(article) if c.section == "By tram")
    assert tram.breadcrumb == "Examplestad > Get around > By tram"
    assert tram.search_text.startswith("Examplestad > Get around > By tram\n")
    assert tram.attribution.url.endswith("#By_tram")

    overview = next(c for c in chunk_article(article) if c.section == OVERVIEW)
    assert overview.attribution.url == "https://en.wikivoyage.org/wiki/Examplestad"


def test_chunk_ids_are_unique_and_stable(article):
    first = [c.id for c in chunk_article(article)]
    second = [c.id for c in chunk_article(article)]
    assert first == second
    assert len(set(first)) == len(first)


def test_chunk_size_is_respected(article):
    chunks = chunk_article(article, max_chars=120)
    assert all(len(c.text) <= 120 for c in chunks if c.kind is ChunkKind.TEXT)


def test_rejects_tiny_chunk_size(article):
    with pytest.raises(ValueError):
        chunk_article(article, max_chars=10)
