"use strict";

// Source page: the data with posterior curves, a splom of the samples whose
// selection limits the curves, a rejection/MCMC toggle, the summary row, and
// any error or captured warnings.

const $ = (id) => document.getElementById(id);
const wrap01 = (x) => ((x % 1) + 1) % 1;
// Data and curve colors, per SB2 star; an RV run uses the first.
const COLORS = { primary: [31, 119, 180], secondary: [214, 39, 40] };
const rgba = (c, a) => `rgba(${c.join(",")},${a})`;
const DEFAULT_DIMS = [
  "period",
  "eccentricity",
  "rv_semiamp",
  "semi_major_axis",
  "primary.rv_semiamp",
  "secondary.rv_semiamp",
];

let DATA = null;
let stage = null; // the sample set shown: "rejection" or "mcmc"
let selected = []; // sample indices selected in the splom

const S = () => DATA.stages[stage];
const label = (name, units) => (units[name] ? `${name} [${units[name]}]` : name);

function median(values) {
  const a = values.filter(Number.isFinite).sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

// The selected sample whose period is nearest the selected median period.
function referenceIndex(indices) {
  const P = S().params.period;
  const valid = indices.filter((i) => Number.isFinite(P[i]));
  if (!valid.length) return null;
  const mid = median(valid.map((i) => P[i]));
  return valid.reduce((a, b) => (Math.abs(P[b] - mid) < Math.abs(P[a] - mid) ? b : a));
}

// Many curves as one trace, separated by nulls.
function curvesTrace(xs, ys, indices, type = "scattergl", color = COLORS.primary) {
  const x = [],
    y = [];
  for (const i of indices) {
    const xi = Array.isArray(xs[0]) ? xs[i] : xs;
    x.push(...xi, null);
    y.push(...ys[i], null);
  }
  return { x, y, type, mode: "lines", line: { color: rgba(color, 0.15), width: 1 }, hoverinfo: "skip" };
}

function errorTrace(x, y, err, color = "black", name = "") {
  return {
    x,
    y,
    name,
    type: "scatter",
    mode: "markers",
    marker: { color, size: 6 },
    error_y: { type: "data", array: err, color },
  };
}

// The RV panel's series: the one dataset of an RV run, or both SB2 stars,
// each with its data, its curves, and its color.
function rvSeries() {
  const c = S().curves;
  if (DATA.kind === "rv") return [{ name: "", d: DATA.data, curves: c && c.rv, color: COLORS.primary, dots: "black" }];
  return Object.entries(COLORS).map(([name, color]) => ({
    name,
    d: DATA.data[name],
    curves: c && c[name],
    color,
    dots: rgba(color, 1),
  }));
}

function renderRv(indices) {
  const series = rvSeries(),
    u = series[0].d.units,
    c = S().curves;
  const layout = {
    showlegend: DATA.kind === "sb2",
    margin: { t: 30 },
    yaxis: { title: `RV [${u.rv}]` },
    xaxis: { title: `time [${u.time}]` },
  };
  const ref = c && $("fold").checked ? referenceIndex(indices) : null;
  const dots = (s, x) => errorTrace(x, s.d.rv, s.d.rv_err, s.dots, s.name);
  if (ref === null) {
    const curves = c ? series.map((s) => curvesTrace(c.time, s.curves, indices, "scattergl", s.color)) : [];
    for (const t of curves) t.showlegend = false;
    Plotly.react($("data-plot"), [...curves, ...series.map((s) => dots(s, s.d.time))], layout);
    return;
  }
  const P = S().params.period[ref],
    t0 = S().params.time_peri[ref];
  const phase = (t) => wrap01((t - t0) / P);
  layout.xaxis = { title: `phase (P = ${P.toPrecision(5)} ${S().units.period})`, range: [0, 1] };
  const folded = series.map((s) => {
    const pts = c.time.map((t, k) => [phase(t), s.curves[ref][k]]).sort((a, b) => a[0] - b[0]);
    const line = { color: rgba(s.color, 1) };
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), mode: "lines", line, showlegend: false };
  });
  Plotly.react($("data-plot"), [...folded, ...series.map((s) => dots(s, s.d.time.map(phase)))], layout);
}

function renderGaia(indices) {
  const d = DATA.data,
    u = d.units,
    c = S().curves;
  // Clear the panels first, so a stage without curves never shows another's.
  if (!c) {
    Plotly.purge($("data-plot"));
    Plotly.purge($("second-plot"));
    return;
  }
  const sky = c.ra ? [curvesTrace(c.ra, c.dec, indices, "scatter")] : [];
  Plotly.react($("data-plot"), sky, {
    showlegend: false,
    margin: { t: 30 },
    title: c.ra ? "Sky-plane orbits" : "No sky-plane orbit for this parameterization",
    xaxis: { title: `ΔRA* [${u.al_position}]`, autorange: "reversed" },
    yaxis: { title: `ΔDec [${u.al_position}]`, scaleanchor: "x" },
  });
  const ref = referenceIndex(indices);
  if (ref === null) {
    Plotly.purge($("second-plot"));
    return;
  }
  Plotly.react($("second-plot"), [errorTrace(d.time, c.residual[ref], d.al_position_err)], {
    showlegend: false,
    margin: { t: 30 },
    title: "Along-scan residuals (sample nearest the median period)",
    xaxis: { title: `time [${u.time}]` },
    yaxis: { title: `residual [${u.al_position}]` },
  });
}

function renderData() {
  const indices = selected.length ? selected : S().params.period.map((_, i) => i);
  (DATA.kind === "gaia_astrometry" ? renderGaia : renderRv)(indices);
}

function renderSplom() {
  const s = S();
  const dims = [...$("dims").querySelectorAll("input:checked")].map((b) => ({
    label: label(b.value, s.units),
    values: s.params[b.value],
  }));
  Plotly.react(
    $("splom"),
    [
      {
        type: "splom",
        dimensions: dims,
        marker: { size: 3, color: "rgba(31,119,180,0.5)" },
        diagonal: { visible: false },
        showupperhalf: false,
      },
    ],
    { dragmode: "select", margin: { t: 20 } },
  );
  selected = [];
  renderData();
}

function buildStage() {
  const s = S();
  // Folding needs the time of periastron, which a Fourier fit lacks.
  $("fold-label").hidden = DATA.kind === "gaia_astrometry" || !s.params.time_peri;
  if ($("fold-label").hidden) $("fold").checked = false;
  $("curves-error").textContent = s.curves_error ? `No curves: ${s.curves_error}` : "";
  const dims = $("dims");
  dims.innerHTML = "";
  for (const name of Object.keys(s.params)) {
    const box = Object.assign(document.createElement("input"), { type: "checkbox", value: name });
    box.checked = DEFAULT_DIMS.includes(name);
    box.onchange = renderSplom;
    const wrap = document.createElement("label");
    wrap.append(box, ` ${name}`);
    dims.appendChild(wrap);
  }
  renderSplom();
}

function renderSummary() {
  const table = $("summary");
  for (const [key, value] of Object.entries(DATA.summary || {})) {
    const row = table.insertRow();
    row.insertCell().textContent = key;
    row.insertCell().textContent = typeof value === "number" ? value.toPrecision(6) : value ?? "";
  }
  const problems = $("problems");
  for (const [name, s] of Object.entries(DATA.stages)) {
    for (const [what, text] of [["error", s.error], ["warnings", s.warnings]]) {
      if (!text) continue;
      const h = document.createElement("h2");
      h.textContent = `${name} ${what}`;
      const pre = document.createElement("pre");
      pre.textContent = text;
      problems.append(h, pre);
    }
  }
}

async function main() {
  const resp = await fetch(`/api/source/${encodeURIComponent(SOURCE_ID)}`);
  if (!resp.ok) {
    const text = await resp.text();
    let detail = text;
    try {
      detail = JSON.parse(text).detail || text;
    } catch {}
    $("title").textContent = SOURCE_ID;
    $("statuses").textContent = `Could not load this source: ${resp.status} ${detail}`;
    $("statuses").className = "hint error";
    return;
  }
  DATA = await resp.json();
  $("title").textContent = DATA.source_id;
  const st = DATA.stages;
  $("statuses").textContent =
    `rejection: ${st.rejection.status} · mcmc: ${st.mcmc.mcmc_status || st.mcmc.status}`;
  $("fold-label").hidden = true; // shown by buildStage when it can fold
  $("second-plot").hidden = DATA.kind !== "gaia_astrometry";
  renderSummary();

  const withSamples = Object.keys(st).filter((k) => st[k].params);
  if (!withSamples.length) {
    $("statuses").textContent += " · no samples to show";
    return;
  }
  // MCMC when converged, as in the summary's "final" column.
  stage = st.mcmc.params && st.mcmc.mcmc_status === "converged" ? "mcmc" : withSamples[0];
  for (const name of withSamples) {
    const radio = Object.assign(document.createElement("input"), {
      type: "radio",
      name: "stage",
      value: name,
      checked: name === stage,
    });
    radio.onchange = () => {
      stage = name;
      buildStage();
    };
    const wrap = document.createElement("label");
    wrap.append(radio, ` ${name} `);
    $("stage-toggle").appendChild(wrap);
  }
  $("fold").onchange = renderData;
  buildStage();
  $("splom").on("plotly_selected", (ev) => {
    if (ev && ev.points && ev.points.length) {
      selected = [...new Set(ev.points.map((p) => p.pointIndex))];
      renderData();
    }
  });
  $("splom").on("plotly_deselect", () => {
    selected = [];
    renderData();
  });
}

main();
