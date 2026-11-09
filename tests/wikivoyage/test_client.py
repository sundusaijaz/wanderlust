import httpx
import pytest

from wanderlust.wikivoyage.client import (
    USER_AGENT,
    ArticleNotFoundError,
    WikivoyageClient,
    WikivoyageError,
)


def test_fetch_article_parses_response(make_client):
    client, requests = make_client()
    article = client.fetch_article("Examplestad")

    assert article.title == "Examplestad"
    assert (article.page_id, article.revision_id) == (42, 7)
    assert article.section("See").listings[0].name == "Old Lighthouse"

    (request,) = requests
    assert request.url.host == "en.wikivoyage.org"
    assert request.url.params["action"] == "parse"
    assert request.url.params["redirects"] == "1"
    assert request.url.params["formatversion"] == "2"


def test_redirected_title_uses_canonical_name(make_client):
    client, _ = make_client()
    assert client.fetch_article("  example town ").title == "Examplestad"


def test_missing_article_suggests_alternative(make_client):
    client, requests = make_client()
    with pytest.raises(ArticleNotFoundError) as info:
        client.fetch_article("Exampel")
    # no hit for "exampel" in the fake search index
    assert info.value.suggestion is None
    assert [r.url.params["action"] for r in requests] == ["parse", "query"]


def test_missing_article_with_search_hit(make_client):
    client, _ = make_client()
    with pytest.raises(ArticleNotFoundError, match="Did you mean 'Examplestad'"):
        client.fetch_article("Example harbour")


def test_spelling_suggestion_is_used_when_no_hits(make_client):
    def handler(request):
        if request.url.params["action"] == "parse":
            return httpx.Response(200, json={"error": {"code": "missingtitle"}})
        return httpx.Response(
            200, json={"query": {"searchinfo": {"suggestion": "lisbon"}, "search": []}}
        )

    client, _ = make_client(handler)
    with pytest.raises(ArticleNotFoundError) as info:
        client.fetch_article("Lisbn")
    assert info.value.suggestion == "Lisbon"


def test_other_api_errors_are_reported(make_client):
    client, _ = make_client(
        lambda r: httpx.Response(200, json={"error": {"code": "ratelimited", "info": "slow down"}})
    )
    with pytest.raises(WikivoyageError, match="slow down"):
        client.fetch_article("Examplestad")


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx.Response(503), "HTTP 503"),
        (lambda r: httpx.Response(200, text="<html>not json</html>"), "invalid JSON"),
        (lambda r: httpx.Response(200, json={"parse": {"title": "X"}}), "Unexpected response"),
    ],
)
def test_bad_responses_raise_wikivoyage_error(make_client, handler, message):
    client, _ = make_client(handler)
    with pytest.raises(WikivoyageError, match=message):
        client.fetch_article("Examplestad")


def test_network_failure_is_wrapped(make_client):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    client, _ = make_client(handler)
    with pytest.raises(WikivoyageError, match="Could not reach Wikivoyage"):
        client.fetch_article("Examplestad")


def test_empty_title_is_rejected(make_client):
    client, requests = make_client()
    with pytest.raises(ValueError):
        client.fetch_article("   ")
    assert requests == []


def test_default_client_identifies_itself():
    with WikivoyageClient("de") as client:
        assert client.endpoint == "https://de.wikivoyage.org/w/api.php"
        assert client._http.headers["User-Agent"] == USER_AGENT
        assert "github.com/sundusaijaz/wanderlust" in USER_AGENT
