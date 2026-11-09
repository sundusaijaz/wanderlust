from pathlib import Path

import pytest

from wanderlust.wikivoyage.models import ListingType
from wanderlust.wikivoyage.parser import parse_article

FIXTURE = Path(__file__).parent.parent / "fixtures" / "examplestad.wikitext"


@pytest.fixture(scope="module")
def article():
    return parse_article(
        FIXTURE.read_text(encoding="utf-8"),
        title="Examplestad",
        language="en",
        page_id=42,
        revision_id=7,
    )


def test_lead_becomes_summary_without_markup(article):
    assert article.summary.startswith("Examplestad is a small, entirely fictional harbour town")
    assert "lighthouse walks" in article.summary
    for leftover in ("[[", "{{", "'''", "<ref", "Invented for", "File:", "thumb"):
        assert leftover not in article.summary


def test_sections_keep_order_level_and_parent(article):
    assert [(s.title, s.level, s.parent) for s in article.sections] == [
        ("Understand", 2, None),
        ("Get around", 2, None),
        ("By tram", 3, "Get around"),
        ("By bicycle", 3, "Get around"),
        ("See", 2, None),
        ("Eat", 2, None),
        ("Go next", 2, None),
    ]


def test_comments_and_categories_are_dropped(article):
    understand = article.section("Understand")
    assert understand.text.startswith("The town grew around a fishing harbour.")
    assert "editor comment" not in understand.text
    assert "Category" not in article.section("Go next").text


def test_see_listings_are_structured(article):
    see = article.section("See")
    assert see.text == ""
    lighthouse, square = see.listings
    assert lighthouse.type is ListingType.SEE
    assert lighthouse.name == "Old Lighthouse"
    assert lighthouse.alt == "Vieux Phare"
    assert lighthouse.price == "Adult 5 coins, children free"
    assert lighthouse.hours == "09:00-18:00"
    assert (lighthouse.lat, lighthouse.lon) == (12.3456, -7.6543)
    assert "viewpoint at the top" in lighthouse.description
    assert square.name == "Clock Square"
    assert not square.has_coordinates


def test_bad_coordinates_are_ignored(article):
    kitchen = article.section("Eat").listings[0]
    assert kitchen.lat == 12.3401
    assert kitchen.lon is None


def test_generic_listing_uses_declared_type(article):
    eat = article.section("Eat")
    assert eat.text == "Fish soup is the local speciality."
    assert [(item.name, item.type) for item in eat.listings] == [
        ("Harbour Kitchen", ListingType.EAT),
        ("The Salty Mug", ListingType.DRINK),
    ]


def test_markers_keep_their_name_in_the_text(article):
    go_next = article.section("Go next")
    assert "Otherville" in go_next.text
    assert "day trip by ferry" in go_next.text
    (marker,) = go_next.listings
    assert marker.type is ListingType.MARKER
    assert marker.name == "Otherville"


def test_sections_link_back_to_their_anchor(article):
    tram = article.section("By tram")
    assert tram.attribution.url == "https://en.wikivoyage.org/wiki/Examplestad#By_tram"
    assert tram.attribution.license == "CC BY-SA 4.0"


def test_listing_without_name_is_skipped():
    article = parse_article(
        "==See==\n* {{see | content=No name here }}\nSome text.",
        title="T",
        language="en",
        page_id=1,
        revision_id=1,
    )
    see = article.section("See")
    assert see.listings == []
    assert see.text == "Some text."


def test_article_without_headings_is_all_summary():
    article = parse_article("Just a stub.", title="T", language="en", page_id=1, revision_id=1)
    assert article.summary == "Just a stub."
    assert article.sections == []
