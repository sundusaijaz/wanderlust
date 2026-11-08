# Wanderlust 🧭

**An agentic trip planner grounded in real travel guides.**

Tell Wanderlust where you want to go and what you like:

> *"5 days in Portugal in March. I love food and hiking, medium budget."*

and it builds a day-by-day itinerary. Every suggestion is backed by a cited
passage from [Wikivoyage](https://www.wikivoyage.org/), checked against live
weather, and shown on a map.

> **Status:** early development (v0.0.x). See the [roadmap](ROADMAP.md) for
> what is built and what is next.

## Why

General-purpose chatbots happily invent restaurants that closed years ago and
hikes that do not exist. Wanderlust takes a different approach:

- **Grounded (RAG):** answers come from retrieved Wikivoyage passages, and
  every recommendation links back to its source.
- **Agentic:** an LLM agent plans the trip step by step, calling tools to
  search guides, check the weather, geocode places and balance the schedule.
- **Transparent:** you can see which tools the agent called and why.

## Who it is for

Travellers who want a solid first draft of a trip in minutes, and developers
who want a readable, well-tested example of retrieval-augmented generation
combined with tool-using agents.

## Built with

- Python 3.11+
- [Typer](https://typer.tiangolo.com/) for the command line
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration
- OpenAI or Anthropic APIs for the language model (your choice)
- Wikivoyage content via the MediaWiki API
- [Open-Meteo](https://open-meteo.com/) for weather and geocoding (free, no key)

Planned: a vector store for retrieval, a web UI and an interactive map. See the
[roadmap](ROADMAP.md).

## Getting started

```bash
git clone https://github.com/sundusaijaz/wanderlust.git
cd wanderlust
uv sync            # or: pip install -e ".[dev]"
cp .env.example .env
# edit .env and add the API key for the provider you want to use
uv run wanderlust --help
```

## Development

```bash
uv run pytest        # run the tests
uv run ruff check .  # lint
uv run ruff format . # format
```

## Data and attribution

Travel content comes from [Wikivoyage](https://www.wikivoyage.org/) and is
available under the [Creative Commons Attribution-ShareAlike
license](https://creativecommons.org/licenses/by-sa/4.0/). Wanderlust fetches it
at runtime and does not redistribute it in this repository. Generated
itineraries include links to the Wikivoyage articles they draw on.

## License

The Wanderlust source code is released under the [MIT License](LICENSE).
