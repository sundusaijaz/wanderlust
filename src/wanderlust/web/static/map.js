// Trip map: each day's stops as numbered markers in the day's colour, joined in
// visiting order, with a day filter. Popups are built from DOM nodes (textContent),
// never HTML strings. Needs Leaflet (vendor/leaflet/leaflet.js) loaded first.
"use strict";

window.WanderlustMap = (() => {
  const DAY_COLOURS = ["#d8678a", "#4f3488", "#c97a1e", "#2a9d8f", "#3a6ea5", "#b5446e", "#6a994e"];
  const TIMES = { morning: "Morning", afternoon: "Afternoon", evening: "Evening" };
  const TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
  const ATTRIBUTION =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  const colour = (day) => DAY_COLOURS[(day - 1) % DAY_COLOURS.length];

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function popup(day, stop, sources) {
    const box = node("div", "map-popup");
    box.append(node("strong", "", stop.name));
    const when = `Day ${day.day} · ${TIMES[stop.time_of_day] || stop.time_of_day}`;
    box.append(node("div", "map-popup-when", stop.indoor ? `${when} · ☔ indoor` : when));
    const source = sources.get(stop.source);
    if (source && typeof source.url === "string" && source.url.startsWith("https://")) {
      const link = node("a", "", `[${stop.source}] ${source.label}`);
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener";
      box.append(link);
    }
    return box;
  }

  function marker(day, stop) {
    const icon = L.divIcon({
      className: "map-marker",
      html: "",
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      popupAnchor: [0, -14],
    });
    const result = L.marker([stop.latitude, stop.longitude], {
      icon,
      title: `Day ${day.day}, stop ${stop.number}: ${stop.name}`,
      alt: stop.name,
    });
    // Fill the icon after Leaflet creates it, so the label is plain text.
    result.on("add", () => {
      const element = result.getElement();
      if (!element) return;
      element.style.setProperty("--day-colour", colour(day.day));
      element.textContent = String(stop.number);
    });
    return result;
  }

  /**
   * Draw the trip in `container`. `trip` is the plan's "map" payload and `sources`
   * maps passage numbers to {label, url}. With `tiles: false` (the offline demo, whose
   * town is made up) no map tiles are requested. Returns {focus(day, number), destroy()}.
   */
  function render(container, trip, sources = new Map(), { tiles = true } = {}) {
    const filters = node("div", "map-filters");
    filters.setAttribute("role", "group");
    filters.setAttribute("aria-label", "Show days on the map");
    const canvas = node("div", "map-canvas");
    canvas.setAttribute("role", "region");
    canvas.setAttribute("aria-label", "Map of the itinerary's stops");
    container.replaceChildren(filters, canvas);

    const map = L.map(canvas, { scrollWheelZoom: false });
    if (tiles) {
      L.tileLayer(TILES, { maxZoom: 19, attribution: ATTRIBUTION }).addTo(map);
    } else {
      canvas.classList.add("no-tiles");
      map.attributionControl.addAttribution("Made-up town: no map tiles");
    }

    const layers = new Map(); // day number -> layer group
    const markers = new Map(); // "day:number" -> marker
    for (const day of trip.days) {
      if (!day.stops.length) continue;
      const group = L.featureGroup();
      if (day.stops.length > 1) {
        L.polyline(
          day.stops.map((stop) => [stop.latitude, stop.longitude]),
          { color: colour(day.day), weight: 3, opacity: 0.7, dashArray: "6 6" },
        ).addTo(group);
      }
      for (const stop of day.stops) {
        const item = marker(day, stop).bindPopup(popup(day, stop, sources));
        item.addTo(group);
        markers.set(`${day.day}:${stop.number}`, item);
      }
      group.addTo(map);
      layers.set(day.day, group);
    }

    function fit(groups) {
      const bounds = L.latLngBounds([]);
      for (const group of groups) bounds.extend(group.getBounds());
      if (!bounds.isValid()) return;
      if (bounds.getNorthEast().equals(bounds.getSouthWest())) {
        map.setView(bounds.getCenter(), 15);
      } else {
        map.fitBounds(bounds, { padding: [28, 28], maxZoom: 16 });
      }
    }

    let selected = null;

    function show(dayNumber) {
      selected = dayNumber;
      for (const [number, group] of layers) {
        const visible = dayNumber === null || number === dayNumber;
        if (visible && !map.hasLayer(group)) group.addTo(map);
        if (!visible && map.hasLayer(group)) map.removeLayer(group);
      }
      for (const button of filters.children) {
        button.setAttribute("aria-pressed", String(button.dataset.day === String(dayNumber)));
      }
      fit(dayNumber === null ? layers.values() : [layers.get(dayNumber)]);
    }

    const addFilter = (label, dayNumber, swatch) => {
      const button = node("button", "map-filter", label);
      button.type = "button";
      button.dataset.day = String(dayNumber);
      if (swatch) button.style.setProperty("--day-colour", swatch);
      button.addEventListener("click", () => show(dayNumber));
      filters.append(button);
    };
    addFilter("All days", null);
    for (const dayNumber of layers.keys()) addFilter(`Day ${dayNumber}`, dayNumber, colour(dayNumber));

    show(null);

    // Leaflet measures its container once; if the page was hidden or resized since,
    // measure again and re-fit the selected days (until the reader moves the map).
    let readerMoved = false;
    map.on("dragstart", () => {
      readerMoved = true;
    });
    let lastWidth = canvas.clientWidth;
    const observer = new ResizeObserver(() => {
      if (canvas.clientWidth === lastWidth || !canvas.clientWidth) return;
      lastWidth = canvas.clientWidth;
      map.invalidateSize();
      if (!readerMoved) show(selected);
    });
    observer.observe(canvas);

    return {
      focus(dayNumber, stopNumber) {
        const item = markers.get(`${dayNumber}:${stopNumber}`);
        if (!item) return;
        show(dayNumber);
        map.setView(item.getLatLng(), Math.max(map.getZoom(), 16));
        item.openPopup();
      },
      has(dayNumber, stopNumber) {
        return markers.has(`${dayNumber}:${stopNumber}`);
      },
      destroy() {
        observer.disconnect();
        map.remove();
      },
    };
  }

  return { render, colour };
})();
