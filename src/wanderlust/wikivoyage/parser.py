"""Turn raw Wikivoyage wikitext into plain-text sections and structured listings."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass, field

import mwparserfromhell
from mwparserfromhell.nodes import Heading, Node, Tag, Template, Wikilink
from mwparserfromhell.wikicode import Wikicode

from wanderlust.wikivoyage.models import (
    Article,
    Attribution,
    Listing,
    ListingType,
    Section,
    article_url,
)

LISTING_TEMPLATES: dict[str, ListingType] = {
    "see": ListingType.SEE,
    "do": ListingType.DO,
    "buy": ListingType.BUY,
    "eat": ListingType.EAT,
    "drink": ListingType.DRINK,
    "sleep": ListingType.SLEEP,
    "go": ListingType.GO,
    "listing": ListingType.OTHER,
}
MARKER_TEMPLATE = "marker"

# Links to these namespaces are media or metadata, not readable text.
_HIDDEN_LINK_PREFIXES = ("file:", "image:", "category:", "media:")
# Tags whose contents are footnotes, galleries or maps rather than article prose.
_HIDDEN_TAGS = frozenset(
    {"ref", "references", "gallery", "imagemap", "mapframe", "maplink", "score", "timeline"}
)
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class _RawSection:
    title: str | None
    level: int
    parent: str | None
    nodes: list[Node] = field(default_factory=list)


def parse_article(
    wikitext: str,
    *,
    title: str,
    language: str,
    page_id: int,
    revision_id: int,
) -> Article:
    """Parse a full article's wikitext into an :class:`Article`."""
    code = mwparserfromhell.parse(wikitext)
    districts = _district_titles(code, title)
    lead, raw_sections = _split_sections(code)
    summary, _ = _clean(Wikicode(lead.nodes))

    sections: list[Section] = []
    for raw in raw_sections:
        text, listings = _clean(Wikicode(raw.nodes))
        assert raw.title is not None
        sections.append(
            Section(
                title=raw.title,
                level=raw.level,
                parent=raw.parent,
                text=text,
                listings=listings,
                attribution=Attribution(
                    title=title, url=article_url(language, title, anchor=raw.title)
                ),
            )
        )

    return Article(
        title=title,
        language=language,
        page_id=page_id,
        revision_id=revision_id,
        summary=summary,
        sections=sections,
        districts=districts,
    )


def _district_titles(code: Wikicode, title: str) -> list[str]:
    """Links to sub-articles such as 'Lisbon/Baixa', which Wikivoyage uses for districts."""
    prefix = f"{title}/".casefold()
    found: dict[str, str] = {}
    for link in code.filter_wikilinks(recursive=True):
        target = " ".join(str(link.title).split("#")[0].replace("_", " ").split())
        if target.casefold().startswith(prefix) and len(target) > len(prefix):
            canonical = title + target[len(title) :]
            found.setdefault(canonical.casefold(), canonical)
    return list(found.values())


def _split_sections(code: Wikicode) -> tuple[_RawSection, list[_RawSection]]:
    """Group top-level nodes under their level-2 and level-3 headings."""
    lead = _RawSection(title=None, level=0, parent=None)
    sections: list[_RawSection] = []
    current = lead
    current_parent: str | None = None

    for node in code.nodes:
        if isinstance(node, Heading) and node.level in (2, 3):
            heading = node.title.strip_code().strip()
            if node.level == 2:
                current_parent = heading
                current = _RawSection(title=heading, level=2, parent=None)
            else:
                current = _RawSection(title=heading, level=3, parent=current_parent)
            sections.append(current)
        else:
            current.nodes.append(node)
    return lead, sections


def _clean(code: Wikicode) -> tuple[str, list[Listing]]:
    """Pull listings out of a block of wikitext and return its readable text."""
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() in _HIDDEN_TAGS:
            _remove(code, tag)
    for link in code.filter_wikilinks(recursive=True):
        if str(link.title).strip().lower().startswith(_HIDDEN_LINK_PREFIXES):
            _remove(code, link)

    listings: list[Listing] = []
    for template in code.filter_templates(recursive=False):
        name = _template_name(template)
        if name in LISTING_TEMPLATES:
            listing = _listing_from_template(template, LISTING_TEMPLATES[name])
            if listing:
                listings.append(listing)
            _remove(code, template)
        elif name == MARKER_TEMPLATE:
            listing = _listing_from_template(template, ListingType.MARKER)
            if listing:
                listings.append(listing)
                # Markers sit inside sentences, so keep the place name in the text.
                code.replace(template, listing.name)
            else:
                _remove(code, template)

    text = code.strip_code(normalize=True, collapse=True)
    return _tidy(text), listings


def _listing_from_template(template: Template, default_type: ListingType) -> Listing | None:
    name = _param(template, "name")
    if not name:
        return None

    listing_type = default_type
    if default_type is ListingType.OTHER:
        declared = (_param(template, "type") or "").lower()
        listing_type = LISTING_TEMPLATES.get(declared, ListingType.OTHER)

    return Listing(
        type=listing_type,
        name=name,
        alt=_param(template, "alt"),
        description=_param(template, "content") or "",
        address=_param(template, "address"),
        directions=_param(template, "directions"),
        phone=_param(template, "phone"),
        hours=_param(template, "hours"),
        price=_param(template, "price"),
        url=_param(template, "url"),
        lat=_coordinate(_param(template, "lat"), limit=90),
        lon=_coordinate(_param(template, "long"), limit=180),
    )


def _template_name(template: Template) -> str:
    return str(template.name).strip().lower()


def _param(template: Template, name: str) -> str | None:
    if not template.has(name):
        return None
    value = _tidy(template.get(name).value.strip_code(normalize=True, collapse=True))
    return value or None


def _coordinate(value: str | None, *, limit: float) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if -limit <= number <= limit else None


def _remove(code: Wikicode, node: Tag | Template | Wikilink) -> None:
    # A ValueError means it was already removed together with an enclosing node.
    with suppress(ValueError):
        code.remove(node)


def _tidy(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    # Drop list bullets left behind once their listing template was removed.
    lines = [line for line in lines if line not in {"*", "#", "-", "\u2013"}]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
