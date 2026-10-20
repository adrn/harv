"use strict";

// Catalog explorer: a scattergl plot of two columns of the joined catalog +
// summary table, with a selection-driven table. Every control, the selection
// and the table's sort live in the URL query string, so a view can be
// bookmarked or shared, and going back to it from a source page restores it.

const $ = (id) => document.getElementById(id);
const CONTROLS = ["x", "y", "color", "status", "filter", "cols"];
const FLAGS = ["logx", "logy", "revx", "revy"];

// The table shows at most this many rows of a selection; the server sorts
// the whole selection first, so the column headers still order all of it.
const MAX_TABLE_ROWS = 1000;

let COLUMNS = []; // [{name, dtype, unit, numeric}]
// Every source ID, by table position, for the summary version in idsVersion.
// The plot's points carry positions, so the IDs are fetched once per version
// instead of with every plot.
let ids = [];
let idsVersion = null;
let selected = []; // the selected points' source IDs
// The selection as text (see selection.js), and the plot it was drawn on:
// a box or lasso drawn on other axes, or on other rows, selects other points.
let selection = null;
let selectionKey = null;
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

// What decides which points a selection covers.
const plotKey = () =>
  ["x", "y", "logx", "logy", "filter", "status"]
    .map((id) => ($(id).type === "checkbox" ? $(id).checked : $(id).value))
    .join("|");

function readUrl() {
  const params = new URLSearchParams(location.search);
  for (const id of CONTROLS) if (params.has(id)) $(id).value = params.get(id);
  for (const id of FLAGS) $(id).checked = params.get(id) === "1";
  selection = params.get("sel");
  selectionKey = plotKey();
  sortKey = params.get("sort");
  sortDir = params.get("desc") === "1" ? -1 : 1;
}

function writeUrl() {
  const params = new URLSearchParams();
  for (const id of CONTROLS) if ($(id).value) params.set(id, $(id).value);
  for (const id of FLAGS) if ($(id).checked) params.set(id, "1");
  if (selection) params.set("sel", selection);
  if (sortKey) params.set("sort", sortKey);
  if (sortKey && sortDir < 0) params.set("desc", "1");
  history.replaceState(null, "", `?${params}`);
}

function clearSelection() {
  selection = null;
  selected = [];
  rows = null;
  $("rows").tHead.innerHTML = "";
  $("rows").tBodies[0].innerHTML = "";
  $("table-note").textContent = "";
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

// The plot's columns, decoded from /api/catalog: one float32 array per
// column, then the rows' table positions as int32.
async function getColumns(columns) {
  const query = new URLSearchParams({
    columns: columns.join(","),
    filter: $("filter").value,
    status: $("status").value,
  });
  const resp = await fetch(`/api/catalog?${query}`);
  if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
  const version = resp.headers.get("X-HQ-Version");
  const buffer = await resp.arrayBuffer();
  const n = buffer.byteLength / 4 / (columns.length + 1);
  const data = { positions: new Int32Array(buffer, 4 * n * columns.length, n) };
  columns.forEach((name, i) => (data[name] = new Float32Array(buffer, 4 * n * i, n)));
  if (version !== idsVersion) {
    const idsResp = await fetch("/api/ids");
    ids = await idsResp.json();
    idsVersion = idsResp.headers.get("X-HQ-Version");
    // The summary changed between the two requests: start over.
    if (idsVersion !== version) return getColumns(columns);
  }
  return data;
}

async function plot() {
  if (selection && selectionKey !== plotKey()) clearSelection();
  writeUrl();
  const x = $("x").value,
    y = $("y").value,
    color = $("color").value;
  const columns = [...new Set([x, y, ...(color ? [color] : [])])];
  const token = ++plotToken;
  let data;
  try {
    data = await getColumns(columns);
  } catch (err) {
    if (token !== plotToken) return;
    $("message").textContent = err.message;
    $("message").className = "hint error";
    return;
  }
  if (token !== plotToken) return;
  $("message").className = "hint";
  $("message").textContent = `${data.positions.length} sources`;
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
  const trace = {
    type: "scattergl",
    mode: "markers",
    x: data[x],
    y: data[y],
    customdata: Array.from(data.positions, (i) => ids[i]),
    marker,
    hovertemplate: "%{customdata}<extra></extra>",
  };
  const layout = {
    dragmode: "lasso",
    margin: { t: 20 },
    xaxis: axis(x, $("logx").checked, $("revx").checked),
    yaxis: axis(y, $("logy").checked, $("revy").checked),
  };
  // Redraw a selection kept from before (from the URL, say, after going
  // back from a source page), and select its points again.
  const shape = decodeSelection(selection);
  if (shape) {
    const inside = selectedIndices(shape, trace.x, trace.y, $("logx").checked, $("logy").checked);
    trace.selectedpoints = inside;
    layout.selections = [selectionShape(shape)];
    selected = inside.map((i) => trace.customdata[i]);
  }
  Plotly.react($("scatter"), [trace], layout);
  if (shape) showRows();
}

async function showRows() {
  const token = ++rowsToken;
  const columns = tableColumns();
  if (!columns.includes(sortKey)) sortKey = null; // its column was removed
  let fetched;
  try {
    fetched = await getJson("/api/rows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_ids: selected,
        columns,
        sort: sortKey,
        descending: sortDir < 0,
        limit: MAX_TABLE_ROWS,
      }),
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
  const n = rows.source_id.length;
  $("table-note").textContent =
    n < selected.length
      ? `Showing ${n} of ${selected.length} selected sources; click a column to sort them all.`
      : `${n} selected sources.`;
  const head = $("rows").tHead;
  head.innerHTML = "";
  const tr = head.insertRow();
  for (const name of names) {
    const th = document.createElement("th");
    th.textContent = label(name) + (name === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "");
    th.onclick = () => {
      sortDir = sortKey === name ? -sortDir : 1;
      sortKey = name;
      writeUrl();
      showRows();
    };
    tr.appendChild(th);
  }
  const body = $("rows").tBodies[0];
  body.innerHTML = "";
  for (let i = 0; i < n; i++) {
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
      // Plotly also reports a redrawn selection, with no shape: keep ours.
      const text = encodeSelection(ev);
      if (text) {
        selection = text;
        selectionKey = plotKey();
      }
      selected = ev.points.map((p) => p.customdata);
      writeUrl();
      showRows();
    }
  });
  $("scatter").on("plotly_deselect", () => {
    clearSelection();
    writeUrl();
  });
  await plot();
}

main();
