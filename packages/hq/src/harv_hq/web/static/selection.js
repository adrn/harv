"use strict";

// A plot selection (a box or a lasso), as text for the URL, and the points
// inside it. Plotly reports both shapes in data values, also on log axes, so
// a selection read back from the URL picks the same points once the plot is
// drawn again. Loaded by catalog.js in the browser, and by node in the tests.

// Plotly's selection event as "box:x0,x1,y0,y1" or "lasso:x,y,x,y,...";
// null for an event without a shape.
function encodeSelection(ev) {
  const fmt = (v) => Number(v.toPrecision(7));
  if (ev && ev.range && ev.range.x && ev.range.y) {
    return `box:${[...ev.range.x, ...ev.range.y].map(fmt).join(",")}`;
  }
  if (ev && ev.lassoPoints && ev.lassoPoints.x) {
    const { x, y } = ev.lassoPoints;
    return `lasso:${x.flatMap((xi, i) => [fmt(xi), fmt(y[i])]).join(",")}`;
  }
  return null;
}

// {type, x, y} from encodeSelection's text (a box as its two corners); null
// for text that is not a selection.
function decodeSelection(text) {
  const match = /^(box|lasso):(.*)$/.exec(text || "");
  if (!match) return null;
  const v = match[2].split(",").map(Number);
  if (v.some(Number.isNaN)) return null;
  if (match[1] === "box") {
    return v.length === 4 ? { type: "box", x: [v[0], v[1]], y: [v[2], v[3]] } : null;
  }
  if (v.length < 6 || v.length % 2) return null;
  return {
    type: "lasso",
    x: v.filter((_, i) => i % 2 === 0),
    y: v.filter((_, i) => i % 2 === 1),
  };
}

// Indices of the points (xs[i], ys[i]) inside the selection. On a log axis
// the lasso's edges are straight in log space, so the test is done there.
function selectedIndices(selection, xs, ys, logx, logy) {
  const tx = logx ? Math.log10 : (v) => v;
  const ty = logy ? Math.log10 : (v) => v;
  const px = selection.x.map(tx);
  const py = selection.y.map(ty);
  const [x0, x1] = [Math.min(...px), Math.max(...px)];
  const [y0, y1] = [Math.min(...py), Math.max(...py)];
  const inside = [];
  for (let i = 0; i < xs.length; i++) {
    const x = tx(xs[i]);
    const y = ty(ys[i]);
    // NaN (a null value, or <= 0 on a log axis) fails every comparison.
    if (!(x >= x0 && x <= x1 && y >= y0 && y <= y1)) continue;
    if (selection.type === "box" || inPolygon(x, y, px, py)) inside.push(i);
  }
  return inside;
}

// Even-odd ray casting, as Plotly's own lasso test does.
function inPolygon(x, y, px, py) {
  let inside = false;
  for (let i = 0, j = px.length - 1; i < px.length; j = i++) {
    if (py[i] > y !== py[j] > y && x < ((px[j] - px[i]) * (y - py[i])) / (py[j] - py[i]) + px[i]) {
      inside = !inside;
    }
  }
  return inside;
}

// Plotly's layout.selections entry that draws the selection's outline.
function selectionShape(selection) {
  if (selection.type === "box") {
    const [x0, x1] = selection.x;
    const [y0, y1] = selection.y;
    return { type: "rect", xref: "x", yref: "y", x0, x1, y0, y1 };
  }
  const points = selection.x.map((x, i) => `${x},${selection.y[i]}`);
  return { type: "path", xref: "x", yref: "y", path: `M${points.join("L")}Z` };
}

if (typeof module !== "undefined") {
  module.exports = { encodeSelection, decodeSelection, selectedIndices, selectionShape };
}
