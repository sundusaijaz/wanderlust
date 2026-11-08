# Roadmap

Wanderlust is built in small milestones. Each one is tracked as a GitHub issue
and lands through its own pull request.

## v0.1: a working planner in the terminal

- [ ] **M1: Project foundation.** Package layout, configuration, CLI skeleton,
      tests and CI.
- [ ] **M2: Wikivoyage client.** Fetch and parse destination articles through
      the MediaWiki API, split them by section (See, Do, Eat, Sleep, Get around),
      and cache them locally.
- [ ] **M3: Retrieval.** Chunk articles, embed them and store them in a local
      vector index, with hybrid (keyword + vector) search.
- [ ] **M4: LLM provider layer.** One interface for OpenAI and Anthropic,
      with streaming and a fake provider for tests.
- [ ] **M5: Grounded Q&A.** Answer travel questions from retrieved passages
      with inline citations back to Wikivoyage.
- [ ] **M6: Tools.** Weather forecast and geocoding through Open-Meteo, plus
      travel-time estimates between places.
- [ ] **M7: Planning agent.** A tool-calling agent loop that turns a trip
      request into a structured, day-by-day itinerary.

## v0.2: something you would actually share

- [ ] **M8: Web UI.** A chat-style interface that streams the plan as the agent
      works and shows which tools it used.
- [ ] **M9: Map view.** Plot each day's stops on an interactive map.
- [ ] **M10: Export.** Save itineraries as Markdown, PDF or calendar (.ics) files.
- [ ] **M11: Docker and deployment.** One-command local run and a hosted demo.

## Later ideas

- Refine a plan conversationally ("swap day 3 for something rainy-day friendly").
- Budget estimates per day.
- Multi-language Wikivoyage support.
- A small regression test set that checks itinerary quality as the code changes.
