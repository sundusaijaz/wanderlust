from wanderlust.wikivoyage.models import (
    Article,
    Attribution,
    Listing,
    ListingType,
    Section,
    article_url,
)


def attribution(anchor: str | None = None) -> Attribution:
    return Attribution(title="Examplestad", url=article_url("en", "Examplestad", anchor))


def make_article() -> Article:
    return Article(
        title="Examplestad",
        language="en",
        page_id=1,
        revision_id=10,
        sections=[
            Section(title="Get around", level=2, attribution=attribution("Get around")),
            Section(
                title="By tram",
                level=3,
                parent="Get around",
                text="Trams run often.",
                attribution=attribution("By tram"),
            ),
            Section(
                title="See",
                level=2,
                listings=[
                    Listing(type=ListingType.SEE, name="Old Lighthouse", lat=1.0, lon=2.0),
                    Listing(type=ListingType.SEE, name="Clock Square"),
                ],
                attribution=attribution("See"),
            ),
        ],
    )


def test_article_url_encodes_spaces_and_anchor():
    assert article_url("en", "New York City") == "https://en.wikivoyage.org/wiki/New_York_City"
    assert (
        article_url("de", "São Paulo", "Get around")
        == "https://de.wikivoyage.org/wiki/S%C3%A3o_Paulo#Get_around"
    )


def test_section_lookup_is_case_insensitive():
    article = make_article()
    assert article.section("  see ").title == "See"
    assert article.section("Nightlife") is None


def test_subsections_and_listings():
    article = make_article()
    assert [s.title for s in article.subsections("get around")] == ["By tram"]
    assert [listing.name for listing in article.listings] == ["Old Lighthouse", "Clock Square"]


def test_listing_coordinates_flag():
    assert Listing(type=ListingType.SEE, name="A", lat=1.0, lon=2.0).has_coordinates
    assert not Listing(type=ListingType.SEE, name="B", lat=1.0).has_coordinates


def test_every_section_carries_license_attribution():
    article = make_article()
    for section in article.sections:
        assert section.attribution.license == "CC BY-SA 4.0"
        assert section.attribution.url.startswith("https://en.wikivoyage.org/wiki/Examplestad#")


def test_article_round_trips_through_json():
    article = make_article()
    assert Article.model_validate_json(article.model_dump_json()) == article
