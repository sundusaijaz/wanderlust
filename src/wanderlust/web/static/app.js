// Wanderlust planner page: sends the trip request, shows the agent's tool calls as they
// stream in, then renders the itinerary. All model text goes through textContent, never
// innerHTML, so nothing the model writes can run as code on this page.
"use strict";

const EXAMPLES = [
  "4 days in Lisbon in March, food and history",
  "A weekend in Porto with port wine and viewpoints",
  "5 days in Portugal in May, hiking and seafood",
];
const DEMO_EXAMPLES = ["3 days in Examplestad from 13 November, lighthouses and seafood"];
const ICONS = {
  search_guide: "🔎",
  geocode: "📍",
  get_weather: "🌦️",
  get_distance: "📏",
  submit_itinerary: "📝",
};
const TIMES = { morning: "Morning", afternoon: "Afternoon", evening: "Evening" };
const WEATHER_ICONS = [
  [/thunder/i, "⛈️"],
  [/snow|sleet/i, "❄️"],
  [/rain|shower|drizzle|wet/i, "🌧️"],
  [/fog|mist/i, "🌫️"],
  [/overcast|cloud/i, "☁️"],
  [/clear|sun/i, "☀️"],
];

function weatherIcon(text) {
  const match = WEATHER_ICONS.find(([pattern]) => pattern.test(text));
  return match ? match[1] : "🌦️";
}

const $ = (id) => document.getElementById(id);
const form = $("plan-form");
const input = $("request");
const goButton = $("go");
const stopButton = $("stop");
const steps = $("steps");
const result = $("result");

let controller = null;
let tripMap = null; // the Leaflet map of the current itinerary, if any

function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  for (const [name, value] of Object.entries(options.attrs || {})) node.setAttribute(name, value);
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function safeUrl(url) {
  return typeof url === "string" && url.startsWith("https://") ? url : null;
}

// -- status and examples ------------------------------------------------------

async function loadStatus() {
  let status;
  try {
    status = await (await fetch("/api/status")).json();
  } catch {
    showNotice("Could not reach the Wanderlust server.");
    return;
  }
  $("version").textContent = status.version ? `v${status.version}` : "";
  const pill = $("model");
  pill.hidden = false;
  if (status.ready) {
    pill.textContent = `${status.provider} · ${status.model}`;
  } else {
    pill.textContent = "No model configured";
    pill.classList.add("warn");
    showNotice(`${status.problem} Add it to your .env file and restart 'wanderlust serve'.`);
    goButton.disabled = true;
  }
  if (status.demo) {
    showNotice("Demo mode: a scripted agent plans a trip to a made-up town, no API key needed.");
  }
  renderExamples(status.demo ? DEMO_EXAMPLES : EXAMPLES);

  const query = new URLSearchParams(location.search).get("q");
  if (query) input.value = query.slice(0, 500);
  // Only the offline demo may start a plan from a link (real plans cost API calls).
  if (status.demo && query && status.ready) form.requestSubmit();
}

function showNotice(message) {
  const notice = $("notice");
  notice.hidden = false;
  notice.textContent = message;
}

function renderExamples(examples) {
  const box = $("examples");
  box.replaceChildren(
    ...examples.map((text) => {
      const chip = el("button", { className: "chip", text, attrs: { type: "button" } });
      chip.addEventListener("click", () => {
        input.value = text;
        input.focus();
      });
      return chip;
    }),
  );
}

// -- streaming --------------------------------------------------------------

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

async function plan(request) {
  controller = new AbortController();
  setBusy(true);
  steps.replaceChildren();
  $("run-summary").textContent = "";
  $("workspace").hidden = false;
  clearMap();
  result.replaceChildren(placeholder());

  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = typeof body.detail === "string" ? body.detail : "Check your request.";
      showError({ message: `The request was not accepted (${response.status}). ${detail}` });
      return;
    }
    for await (const { event, data } of readEvents(response)) handle(event, data);
  } catch (error) {
    if (error.name === "AbortError") {
      addStep("✋", "Stopped", "You stopped the agent.");
      result.replaceChildren();
    } else {
      showError({ message: "The connection to the server was lost." });
    }
  } finally {
    setBusy(false);
    controller = null;
  }
}

const running = new Map();

function handle(event, data) {
  switch (event) {
    case "start":
      addStep("🤖", `Planning with ${data.provider} · ${data.model}`);
      break;
    case "status":
      addStep("🗂️", data.message);
      break;
    case "thinking":
      addStep("💭", data.text, null, "thinking");
      break;
    case "tool_started":
      running.set(data.id, addStep(ICONS[data.name] || "🔧", data.label, "", "running"));
      break;
    case "tool_finished": {
      const step = running.get(data.id);
      if (step) {
        step.classList.remove("running");
        step.classList.toggle("error", data.is_error);
        step.querySelector(".step-outcome").textContent =
          (data.is_error ? "✗ " : "→ ") + data.summary;
        running.delete(data.id);
      }
      break;
    }
    case "plan":
      $("run-summary").textContent = `Done: ${data.summary}.`;
      showItinerary(data);
      break;
    case "error":
      if (data.summary) $("run-summary").textContent = `Stopped after ${data.summary}.`;
      showError(data);
      break;
    default:
      break;
  }
}

function addStep(icon, label, outcome = null, kind = "") {
  const step = el(
    "li",
    { className: `step ${kind}`.trim() },
    el("span", { className: "step-icon", text: icon, attrs: { "aria-hidden": "true" } }),
    el(
      "span",
      { className: "step-label", text: label },
      outcome === null ? null : el("span", { className: "step-outcome", text: outcome }),
    ),
  );
  steps.append(step);
  return step;
}

function setBusy(busy) {
  goButton.disabled = busy;
  goButton.textContent = busy ? "Planning…" : "Plan my trip";
  stopButton.hidden = !busy;
  input.readOnly = busy;
}

function placeholder() {
  return el(
    "div",
    { className: "card placeholder" },
    el("div", { className: "spinner", attrs: { "aria-hidden": "true" } }),
    el("p", { text: "The agent is researching your trip…" }),
  );
}

// -- rendering the itinerary -----------------------------------------------------

function citation(number, known) {
  if (!known.has(number)) return el("sup", { className: "cite", text: `[${number}]` });
  return el("a", {
    className: "cite",
    text: `[${number}]`,
    attrs: { href: `#source-${number}`, title: known.get(number).label },
  });
}

// Turn "Take the tram [3]." into text plus a link for [3].
function withCitations(text, known) {
  const fragment = document.createDocumentFragment();
  let last = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    fragment.append(text.slice(last, match.index), citation(Number(match[1]), known));
    last = match.index + match[0].length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

function formatDate(iso) {
  const date = new Date(`${iso}T12:00:00`);
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function renderDay(day, known, mapDay) {
  const card = $("day-template").content.firstElementChild.cloneNode(true);
  const meta = [`Day ${day.day}`];
  if (day.date) meta.push(formatDate(day.date));
  meta.push(day.base);
  const metaLine = card.querySelector(".day-meta");
  metaLine.textContent = meta.join(" · ");
  if (mapDay?.stops.length && window.WanderlustMap) {
    const swatch = el("span", { className: "day-swatch", attrs: { "aria-hidden": "true" } });
    swatch.style.setProperty("--day-colour", window.WanderlustMap.colour(day.day));
    metaLine.prepend(swatch);
  }
  card.querySelector(".day-title").textContent = day.title;
  if (day.weather) {
    const weather = card.querySelector(".weather");
    weather.hidden = false;
    weather.textContent = `${weatherIcon(day.weather)} ${day.weather}`;
  }
  const placed = new Set((mapDay?.stops || []).map((stop) => stop.number));
  const list = card.querySelector(".stops");
  day.stops.forEach((stop, index) => {
    const name = el("p", { className: "stop-name", text: stop.name });
    for (const number of stop.sources) name.append(citation(number, known));
    if (stop.indoor) name.append(el("span", { className: "badge", text: "☔ indoor" }));
    if (placed.has(index + 1)) name.append(locateButton(day.day, index + 1, stop.name));
    list.append(
      el(
        "li",
        { className: "stop" },
        el("span", { className: `when ${stop.time_of_day}`, text: TIMES[stop.time_of_day] }),
        el("div", {}, name, el("p", { className: "stop-text", text: stop.description })),
      ),
    );
  });
  if (mapDay?.route_km) {
    card.append(
      el("p", {
        className: "getting-around muted",
        text: `🗺️ Mapped stops are about ${mapDay.route_km.toFixed(1)} km apart in a straight line.`,
      }),
    );
  }
  if (day.getting_around) {
    const around = card.querySelector(".getting-around");
    around.hidden = false;
    around.textContent = `🚋 ${day.getting_around}`;
  }
  return card;
}

function locateButton(dayNumber, stopNumber, name) {
  const button = el("button", {
    className: "locate",
    text: "📍 map",
    attrs: { type: "button", "aria-label": `Show ${name} on the map` },
  });
  button.addEventListener("click", () => {
    if (!tripMap) return;
    $("trip-map").scrollIntoView({ behavior: "smooth", block: "center" });
    tripMap.focus(dayNumber, stopNumber);
  });
  return button;
}

function mapCard(trip) {
  const note =
    trip.placed === trip.total
      ? `All ${trip.total} stops are placed using coordinates from the guides.`
      : `${trip.placed} of ${trip.total} stops have coordinates in the guides; ` +
        "the others are in the day plans below.";
  return el(
    "div",
    { className: "card map-card", attrs: { id: "trip-map" } },
    el("h2", { text: "Map" }),
    el("div", { className: "map-container" }),
    el("p", { className: "map-note muted small", text: note }),
  );
}

function clearMap() {
  tripMap?.destroy();
  tripMap = null;
}

function showItinerary(data) {
  const itinerary = data.itinerary;
  const known = new Map(data.sources.map((source) => [source.number, source]));
  const trip = data.map && data.map.placed > 0 && window.L && window.WanderlustMap ? data.map : null;
  const mapDays = new Map((trip?.days || []).map((day) => [day.day, day]));
  const parts = [
    el(
      "div",
      { className: "card overview" },
      el("h2", { text: itinerary.title }),
      el("p", { text: itinerary.summary }),
    ),
  ];
  if (trip) parts.push(mapCard(trip));
  parts.push(...itinerary.days.map((day) => renderDay(day, known, mapDays.get(day.day))));

  const extras = el("div", { className: "card extras" });
  if (itinerary.tips.length) {
    extras.append(
      el("h3", { text: "Tips" }),
      el("ul", {}, ...itinerary.tips.map((tip) => el("li", {}, withCitations(tip, known)))),
    );
  }
  if (itinerary.assumptions.length) {
    extras.append(
      el("h3", { text: "Assumptions" }),
      el(
        "ul",
        { className: "muted" },
        ...itinerary.assumptions.map((text) => el("li", { text })),
      ),
    );
  }
  extras.append(
    el("h3", { text: "Sources" }),
    el(
      "ol",
      { className: "sources" },
      ...data.sources.map((source) => {
        const url = safeUrl(source.url);
        const item = el("li", { attrs: { id: `source-${source.number}`, value: source.number } });
        item.append(
          url
            ? el("a", { text: source.label, attrs: { href: url, target: "_blank", rel: "noopener" } })
            : source.label,
          el("span", { className: "muted small", text: ` · Wikivoyage, ${source.license}` }),
        );
        return item;
      }),
    ),
  );
  if (data.used_weather) {
    extras.append(el("p", { className: "muted small", text: "Weather data by Open-Meteo.com." }));
  }
  parts.push(extras);
  clearMap();
  result.replaceChildren(...parts);
  if (trip) {
    tripMap = window.WanderlustMap.render(result.querySelector(".map-container"), trip, known);
  }
}

function showError(data) {
  const card = el(
    "div",
    { className: "card error-card", attrs: { role: "alert" } },
    el("h2", { text: "No plan this time" }),
    el("p", { text: data.message }),
  );
  if (data.model_text) {
    card.append(el("p", { className: "muted", text: `The model said: ${data.model_text}` }));
  }
  result.replaceChildren(card);
}

// -- wiring ----------------------------------------------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const request = input.value.trim();
  if (request.length < 3 || controller) return;
  plan(request);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

stopButton.addEventListener("click", () => controller?.abort());

loadStatus();
