"use strict";

// Catalog explorer: a scattergl plot of two columns of the joined catalog +
// summary table, with a selection-driven table. Every control lives in the
// URL query string, so a view can be bookmarked or shared.

const $ = (id) => document.getElementById(id);
const CONTROLS = ["x", "y", "color", "status", "filter", "cols"];
const FLAGS = ["logx", "logy", "revx", "revy"];

let COLUMNS = []; // [{name, dtype, unit, numeric}]
let rows = null; // the table's current rows, as {column: values}
let sortKey = null;
let sortDir = 1;
// Each request gets a token; a response whose token is no longer the latest
// (a slower, older request) is dropped instead of overwriting newer results.
let plotToken = 0;
let rowsToken = 0;

const sourceUrl = (id) => `/source/${encodeURIComponent(id)}`;

function label(name) {
  const col = COLUMNS.find((c) => c.name === name);
  return col && col.unit ? `${name} [${col.unit}]` : name;
}

function fillSelect(select, names, allowNone) {
  select.innerHTML = "";
  if (allowNone) select.add(new Option("(none)", ""));
  for (const name of names) select.add(new Option(name, name));
}

function readUrl() {
  const params = new URLSearchParams(location.search);
  for (const id of CONTROLS) if (params.has(id)) $(id).value = params.get(id);
  for (const id of FLAGS) $(id).checked = params.get(id) === "1";
}

function writeUrl() {
  const params = new URLSearchParams();
  for (const id of CONTROLS) if ($(id).value) params.set(id, $(id).value);
  for (const id of FLAGS) if ($(id).checked) params.set(id, "1");
  history.replaceState(null, "", `?${params}`);
}

function tableColumns() {
  const extra = $("cols")
    .value.split(",")
    .map((c) => c.trim())
    .filter(Boolean);
  return [...new Set(["source_id", $("x").value, $("y").value, ...extra])];
}

async function getJson(url, options) {
  const resp = await fetch(url, options);
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.detail || resp.statusText);
  return body;
}

async function plot() {
  writeUrl();
  const x = $("x").value,
    y = $("y").value,
    color = $("color").value;
  const columns = ["source_id", x, y, ...(color ? [color] : [])];
  const query = new URLSearchParams({
    columns: [...new Set(columns)].join(","),
    filter: $("filter").value,
    status: $("status").value,
  });
  const token = ++plotToken;
  let data;
  try {
    data = await getJson(`/api/catalog?${query}`);
  } catch (err) {
    if (token !== plotToken) return;
    $("message").textContent = err.message;
    $("message").className = "hint error";
    return;
  }
  if (token !== plotToken) return;
  $("message").className = "hint";
  $("message").textContent = `${data.source_id.length} sources`;
  const marker = { size: 4, opacity: 0.7 };
  if (color) {
    marker.color = data[color];
    marker.colorscale = "Viridis";
    marker.showscale = true;
    marker.colorbar = { title: label(color) };
  }
  const axis = (name, log, rev) => ({
    title: label(name),
    type: log ? "log" : "linear",
    autorange: rev ? "reversed" : true,
  });
  Plotly.react(
    $("scatter"),
    [
      {
        type: "scattergl",
        mode: "markers",
        x: data[x],
        y: data[y],
        customdata: data.source_id,
        marker,
        hovertemplate: "%{customdata}<extra></extra>",
      },
    ],
    {
      dragmode: "lasso",
      margin: { t: 20 },
      xaxis: axis(x, $("logx").checked, $("revx").checked),
      yaxis: axis(y, $("logy").checked, $("revy").checked),
    },
  );
}

async function showRows(ids) {
  const token = ++rowsToken;
  let fetched;
  try {
    fetched = await getJson("/api/rows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_ids: ids, columns: tableColumns() }),
    });
  } catch (err) {
    if (token !== rowsToken) return;
    $("message").textContent = err.message;
    $("message").className = "hint error";
    return;
  }
  if (token !== rowsToken) return;
  rows = fetched;
  renderTable();
}

function renderTable() {
  const names = Object.keys(rows);
  if (!names.includes(sortKey)) sortKey = null; // its column was removed
  const n = rows.source_id.length;
  const order = [...Array(n).keys()];
  if (sortKey) {
    const v = rows[sortKey];
    order.sort((a, b) => {
      if (v[a] === v[b]) return 0;
      if (v[a] === null) return 1;
      if (v[b] === null) return -1;
      return (v[a] < v[b] ? -1 : 1) * sortDir;
    });
  }
  const head = $("rows").tHead;
  head.innerHTML = "";
  const tr = head.insertRow();
  for (const name of names) {
    const th = document.createElement("th");
    th.textContent = label(name) + (name === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "");
    th.onclick = () => {
      sortDir = sortKey === name ? -sortDir : 1;
      sortKey = name;
      renderTable();
    };
    tr.appendChild(th);
  }
  const body = $("rows").tBodies[0];
  body.innerHTML = "";
  for (const i of order) {
    const row = body.insertRow();
    for (const name of names) {
      const v = rows[name][i];
      const cell = row.insertCell();
      if (name === "source_id") {
        // A real link, so the table works from the keyboard too.
        const link = Object.assign(document.createElement("a"), {
          href: sourceUrl(v),
          textContent: v,
        });
        cell.appendChild(link);
      } else {
        cell.textContent = typeof v === "number" ? v.toPrecision(6) : v === null ? "" : v;
      }
    }
  }
}

async function main() {
  COLUMNS = await getJson("/api/columns");
  // Not source_id: IDs travel as strings, so they cannot be plotted as numbers.
  const numeric = COLUMNS.filter((c) => c.numeric && c.name !== "source_id").map(
    (c) => c.name,
  );
  fillSelect($("x"), numeric, false);
  fillSelect($("y"), numeric, false);
  fillSelect($("color"), numeric, true);
  $("x").value = numeric[0];
  $("y").value = numeric[1] || numeric[0];
  readUrl();

  $("controls").addEventListener("submit", (ev) => {
    ev.preventDefault();
    plot();
  });
  for (const id of ["x", "y", "color", ...FLAGS]) $(id).onchange = plot;
  $("search").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const id = $("search-id").value.trim();
    if (id) location.href = sourceUrl(id);
  });
  // An empty plot first, so the selection handler is attached even when the
  // first request fails (e.g. a bookmarked filter with a typo).
  await Plotly.newPlot($("scatter"), [], { margin: { t: 20 } });
  $("scatter").on("plotly_selected", (ev) => {
    if (ev && ev.points && ev.points.length) {
      showRows(ev.points.map((p) => p.customdata));
    }
  });
  await plot();
}

main();
