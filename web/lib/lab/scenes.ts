/**
 * Turns (dataset, black box, explained instance) into geometry + a manim-style storyboard
 * for the C++/WebGL2 engine. Four views:
 *   surface   2 axes + the model's output as height; LIME plane -> ArcTanh S-surface (Transform)
 *   space3d   3 axes; P = 0.5 iso-surface of the model vs the surrogates' decision planes,
 *             optionally swept along a 4th direction ("4th dimension as time")
 *   space2d   2 axes, top view; decision contours
 *   tour      all d dimensions: a grand tour of the data with the walk and attribution arrows
 * All coordinates are standardised units (training standard deviations).
 */
import { ArcTanhLime, DeltaAnchorEnsemble, LimeTabular, Rng, TanhSurrogate } from "@/lib/core/index.ts";
import type { ArcTanhResult, Bounds, EnsembleResult, LimeResult, Surrogate } from "@/lib/core/index.ts";
import type { LabDataset, LabModel } from "./data";
import { pca } from "./data";
import { CAM, CAMERA, EASE, PROP } from "./viz";
import type { RGBA, Viz } from "./viz";

export type ViewMode = "surface" | "space3d" | "space2d" | "tour";
export type AxisMode = "explanation" | "pca" | "features";

export const R = 3; // half-width of the scene box, in standard deviations
const H = 3; // world height of P = 1 in the surface view

export const COLORS = {
  lime: [0.224, 0.529, 0.898, 1] as RGBA,
  atl: [0.922, 0.408, 0.204, 1] as RGBA,
  blend: [0.98, 0.74, 0.55, 1] as RGBA,
  model: [0.8, 0.8, 0.78, 1] as RGBA,
  neg: [0.106, 0.686, 0.478, 1] as RGBA,
  pos: [0.91, 0.482, 0.643, 1] as RGBA,
  x0: [0.97, 0.97, 0.95, 1] as RGBA,
  grid: [0.42, 0.44, 0.5, 0.35] as RGBA,
  box: [0.58, 0.6, 0.66, 0.55] as RGBA,
};

const withAlpha = (c: RGBA, a: number): RGBA => [c[0], c[1], c[2], a];

export interface Explanation {
  x0: number[];
  f0: number;
  lime: LimeResult;
  at: ArcTanhResult;
  delta: EnsembleResult<ArcTanhResult> | null;
}

export function explain(ds: LabDataset, model: LabModel, x0: number[], withDelta: boolean, seed = 1): Explanation {
  const bounds = ds.bounds as Bounds;
  const lime = new LimeTabular(ds.scale, { numSamples: 1000 }).explain(model.predict, x0, new Rng(seed));
  const at = new ArcTanhLime(ds.scale, { numSamples: 1000 }, bounds).explain(model.predict, x0, new Rng(seed));
  const delta = withDelta
    ? new DeltaAnchorEnsemble(new ArcTanhLime(ds.scale, { numSamples: 600 }, bounds), { delta: 1 }, bounds)
      .explain(model.predict, x0, new Rng(seed))
    : null;
  return { x0, f0: model.predict(x0), lime, at, delta };
}

// ---------------------------------------------------------------------------------------------
// Axes in standardised space
// ---------------------------------------------------------------------------------------------

export interface Axes {
  dirs: number[][];
  names: string[];
}

const dotv = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const normv = (a: number[]) => Math.sqrt(dotv(a, a));

function orthonormalize(vectors: number[][], d: number): number[][] {
  const out: number[][] = [];
  for (const v0 of vectors) {
    const v = v0.slice();
    for (const u of out) {
      const s = dotv(v, u);
      for (let j = 0; j < d; j++) v[j] -= s * u[j];
    }
    const n = normv(v);
    if (n > 1e-8) out.push(v.map((x) => x / n));
  }
  for (let j = 0; out.length < vectors.length && j < d; j++) {
    // pad with basis vectors if something was degenerate
    const e = Array.from({ length: d }, (_, k) => (k === j ? 1 : 0));
    for (const u of out) {
      const s = dotv(e, u);
      for (let k = 0; k < d; k++) e[k] -= s * u[k];
    }
    const n = normv(e);
    if (n > 1e-6) out.push(e.map((x) => x / n));
  }
  return out;
}

export function standardRows(ds: LabDataset, center: number[]): number[][] {
  return ds.rows.map((r) => r.map((v, j) => (v - center[j]) / ds.scale[j]));
}

let pcaCache: { key: string; result: ReturnType<typeof pca> } | null = null;
export function principalComponents(ds: LabDataset) {
  const key = `${ds.key}:${ds.title}:${ds.rows.length}`;
  if (!pcaCache || pcaCache.key !== key) pcaCache = { key, result: pca(standardRows(ds, ds.mean)) };
  return pcaCache.result;
}

/** The surrogate's ridge direction beta (standardised units), falling back to LIME's coefficients. */
export function explanationDirection(ex: Explanation): number[] {
  const s = ex.at.surrogate;
  const v = s instanceof TanhSurrogate ? s.beta : ex.lime.surrogate.coef;
  return normv(v) > 1e-10 ? v : ex.lime.surrogate.coef;
}

export function chooseAxes(ds: LabDataset, ex: Explanation, mode: AxisMode, features: number[], count: number): Axes {
  const ax = chooseAxesRaw(ds, ex, mode, features, count);
  const d = ds.features.length;
  while (ax.dirs.length < count) {
    // fewer independent directions than axes (d < count): extrude along an empty axis
    ax.dirs.push(new Array<number>(d).fill(0));
    ax.names.push("(no further dimension)");
  }
  return ax;
}

function chooseAxesRaw(ds: LabDataset, ex: Explanation, mode: AxisMode, features: number[], count: number): Axes {
  const d = ds.features.length;
  const pc = principalComponents(ds);
  const pcName = (k: number) => `PC${k + 1} (${(pc.explained[k] * 100).toFixed(0)}%)`;
  if (mode === "features") {
    const idx = [...new Set(features.slice(0, count).map((i) => Math.min(i, d - 1)))];
    return { dirs: idx.map((i) => Array.from({ length: d }, (_, j) => (j === i ? 1 : 0))), names: idx.map((i) => ds.features[i]) };
  }
  if (mode === "pca" || d < 2) {
    const dirs = orthonormalize(pc.vectors.slice(0, count), d);
    return { dirs, names: dirs.map((_, k) => pcName(k)) };
  }
  const dirs = orthonormalize([explanationDirection(ex), ...pc.vectors.slice(0, count)], d).slice(0, count);
  return { dirs, names: ["β: explanation direction", "PC1 ⟂ β", "PC2 ⟂ β", "PC3 ⟂ β"].slice(0, count) };
}

/** Raw feature vector at standardised offset sum_k c_k * dirs_k from x0. */
function toRaw(ds: LabDataset, x0: number[], dirs: number[][], c: number[]): number[] {
  return x0.map((v, j) => {
    let z = 0;
    for (let k = 0; k < dirs.length; k++) z += c[k] * dirs[k][j];
    return v + z * ds.scale[j];
  });
}

function coords(ds: LabDataset, x0: number[], dirs: number[][], x: number[]): number[] {
  const z = x.map((v, j) => (v - x0[j]) / ds.scale[j]);
  return dirs.map((a) => dotv(a, z));
}

function walkPath(ex: Explanation): { rows: number[][]; values: number[]; upEnd: number; downEnd: number } {
  const up = ex.at.walkUp;
  const down = ex.at.walkDown;
  const dr = down ? down.points.slice(1).reverse() : [];
  const dv = down ? down.values.slice(1).reverse() : [];
  const rows = [...dr, ex.x0, ...(up ? up.points.slice(1) : [])];
  const values = [...dv, ex.f0, ...(up ? up.values.slice(1) : [])];
  return { rows, values, downEnd: 0, upEnd: rows.length - 1 };
}

function linspace(a: number, b: number, n: number) {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

// ---------------------------------------------------------------------------------------------
// Scene output
// ---------------------------------------------------------------------------------------------

export interface Label {
  text: string;
  pos: number[];
  kind: "axis" | "note" | "x0";
  appearAt?: number;
}

export interface Caption {
  t: number;
  text: string;
}

export interface BuiltScene {
  labels: Label[];
  captions: Caption[];
  duration: number;
  pickObject: number;
  pickRows: number[];
  tourAxisLabels: { j: number; text: string }[];
  sweep: { start: number; dur: number; nt: number; range: number; name: string } | null;
}

export interface SceneOptions {
  mode: ViewMode;
  axisMode: AxisMode;
  features: number[];
  showData: boolean;
  showDelta: boolean;
  levels: boolean;
  sweep: boolean;
  positive: string;
}

const empty = (): BuiltScene => ({ labels: [], captions: [], duration: 0, pickObject: -1, pickRows: [], tourAxisLabels: [], sweep: null });

function dataPoints(viz: Viz, ds: LabDataset, pts: number[][], keep: number[], radius: number) {
  const xyz = new Float32Array(keep.length * 3);
  const rgba = new Float32Array(keep.length * 4);
  keep.forEach((i, k) => {
    xyz.set(pts[i], k * 3);
    rgba.set(ds.labels[i] > 0.5 ? COLORS.pos : COLORS.neg, k * 4);
  });
  return viz.points(xyz, rgba, radius, true);
}

function singlePoint(viz: Viz, p: number[], c: RGBA, radius: number) {
  return viz.points(Float32Array.from(p), Float32Array.from(c), radius);
}

// ---------------------------------------------------------------------------------------------
// Surface view
// ---------------------------------------------------------------------------------------------

function buildSurface(viz: Viz, ds: LabDataset, model: LabModel, ex: Explanation, ax: Axes, o: SceneOptions): BuiltScene {
  const out = empty();
  const n = 57;
  const us = linspace(-R, R, n);
  const hm = new Float32Array(n * n);
  const hl = new Float32Array(n * n);
  const ha = new Float32Array(n * n);
  const hd = ex.delta && o.showDelta ? new Float32Array(n * n) : null;
  const [a1, a2] = ax.dirs;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const x = toRaw(ds, ex.x0, [a1, a2], [us[i], us[j]]);
      hm[j * n + i] = model.fast(x);
      hl[j * n + i] = ex.lime.surrogate.predict(x);
      ha[j * n + i] = ex.at.surrogate.predict(x);
      if (hd && ex.delta) hd[j * n + i] = ex.delta.surrogate.predict(x);
    }
  const box: [number, number, number, number] = [-R, R, -R, R];
  viz.clip(true, -R, R, -R, R, -0.03, H * 1.04);
  const floor = viz.floorGrid(box, 0, 1, 0.006, COLORS.grid);
  const frame = viz.box([-R, -R, 0], [R, R, H], 0.009, COLORS.box);
  const surf = viz.surface(hm, null, n, n, box, H, COLORS.model, true);
  const lime = viz.surface(hl, null, n, n, box, H, withAlpha(COLORS.lime, 0.55));
  const atl = viz.surface(hl, ha, n, n, box, H, withAlpha(COLORS.atl, 0.82));
  const x0 = singlePoint(viz, [0, 0, ex.f0 * H], COLORS.x0, 0.11);
  viz.scaleOrigin(x0, [0, 0, ex.f0 * H]);

  const w = walkPath(ex);
  const wxyz = w.rows.flatMap((r, k) => [...coords(ds, ex.x0, [a1, a2], r), w.values[k] * H]);
  const walkTube = viz.tube(wxyz, 0.028, COLORS.atl);
  const walkDots = viz.points(Float32Array.from(wxyz), Float32Array.from(w.rows.flatMap(() => COLORS.atl)), 0.045);
  const tanh = ex.at.surrogate instanceof TanhSurrogate;
  const asym = tanh ? [viz.planeZ(ex.at.lo * H, box, withAlpha(COLORS.atl, 0.13)), viz.planeZ(ex.at.hi * H, box, withAlpha(COLORS.atl, 0.13))] : [];

  for (const id of [floor, frame]) viz.set(id, PROP.ALPHA, 0);
  viz.set(surf, PROP.REVEAL, 0);
  for (const id of [lime, walkTube]) viz.set(id, PROP.REVEAL, 0);
  for (const id of [x0, walkDots, atl, ...asym]) viz.set(id, PROP.ALPHA, 0);

  let t = 0;
  viz.tween(floor, PROP.ALPHA, 0, 1, t, 0.8);
  viz.tween(frame, PROP.ALPHA, 0, 1, t, 0.8);
  viz.tween(surf, PROP.REVEAL, 0, 1.01, t + 0.4, 2.2);
  out.captions.push({ t: 0, text: `The black box: P(${o.positive}) as height over a 2-D slice through x₀.` });
  t = 2.8;
  viz.tween(x0, PROP.ALPHA, 0, 1, t, 0.4);
  viz.tween(x0, PROP.SCALE, 2.4, 1, t, 0.9);
  viz.tween(x0, PROP.EMISSIVE, 0, 0.9, t, 1.4, EASE.THERE_AND_BACK);
  out.captions.push({ t, text: "x₀ — the instance being explained." });
  t = 4.1;
  viz.tween(lime, PROP.REVEAL, 0, 1.01, t, 1.6);
  out.captions.push({ t, text: "LIME fits a kernel-weighted plane: faithful only right next to x₀." });
  t = 6.1;
  viz.tween(walkTube, PROP.REVEAL, 0, 1.01, t, 2.0, EASE.LINEAR);
  viz.tween(walkDots, PROP.ALPHA, 0, 1, t, 0.6);
  out.captions.push({ t, text: "ε-walk: fixed steps up to the nearest local maximum and down to the nearest local minimum." });
  t = 8.4;
  for (const id of asym) viz.tween(id, PROP.ALPHA, 0, 1, t, 0.8);
  if (tanh) out.captions.push({ t, text: `Their values (${ex.at.rawLo.toFixed(3)} and ${ex.at.rawHi.toFixed(3)}) become the asymptotes of tanh⁻¹.` });
  t = 9.6;
  viz.tween(atl, PROP.ALPHA, 0, 1, t, 0.4);
  viz.tween(lime, PROP.ALPHA, 1, 0.22, t + 0.2, 1.0);
  viz.tween(atl, PROP.MORPH, 0, 1, t + 0.5, 2.3);
  viz.tween(surf, PROP.ALPHA, 1, 0.62, t + 0.3, 1.2);
  out.captions.push({ t, text: "Transform: the plane bends into ArcTanh-LIME's bounded S-surface." });
  t = 12.6;
  viz.tween(CAMERA, CAM.THETA, -2.25, -1.25, t, 3.2);
  viz.tween(CAMERA, CAM.PHI, 1.12, 1.32, t, 3.2);
  out.captions.push({ t, text: "Compare the orange sheet and the blue plane against the grey surface away from x₀." });
  out.duration = 16;
  if (hd && ex.delta) {
    const anchors = ex.delta.anchors.slice(1).map((a, k) => [...coords(ds, ex.x0, [a1, a2], a), model.predict(a) * H, k]);
    const ap = viz.points(Float32Array.from(anchors.flatMap((a) => a.slice(0, 3))),
      Float32Array.from(anchors.flatMap(() => COLORS.x0)), 0.06);
    const blend = viz.surface(hd, null, n, n, box, H, withAlpha(COLORS.blend, 0.7));
    viz.set(ap, PROP.ALPHA, 0);
    viz.set(blend, PROP.REVEAL, 0);
    t = 16;
    viz.tween(ap, PROP.ALPHA, 0, 1, t, 0.6);
    viz.tween(blend, PROP.REVEAL, 0, 1.01, t + 0.4, 2.0);
    viz.tween(atl, PROP.ALPHA, 1, 0.35, t + 0.4, 1.0);
    out.captions.push({ t, text: `±δ anchors (${ex.delta.anchors.length - 1}, projected) and their blended, partially-global surrogate (light orange).` });
    out.duration = 19;
  }
  if (o.showData) {
    const pts = ds.rows.map((r, i) => [...coords(ds, ex.x0, [a1, a2], r), ds.labels[i] * H]);
    const keep = pts.map((_, i) => i).filter((i) => Math.abs(pts[i][0]) <= R && Math.abs(pts[i][1]) <= R);
    out.pickObject = dataPoints(viz, ds, pts, keep, 0.045);
    out.pickRows = keep;
  }
  viz.camera(-2.25, 1.12, 11.5, [0, 0, H * 0.42]);
  viz.ortho(false);
  out.labels.push(
    { text: `${ax.names[0]} →`, pos: [R + 0.55, 0, 0], kind: "axis" },
    { text: `${ax.names[1]} →`, pos: [0, R + 0.55, 0], kind: "axis" },
    { text: "P = 1", pos: [-R, R, H + 0.2], kind: "axis" },
    { text: "x₀", pos: [0, 0, ex.f0 * H + 0.4], kind: "x0", appearAt: 2.8 },
  );
  if (ex.at.walkUp && ex.at.walkUp.points.length > 1)
    out.labels.push({ text: "local max", pos: [...coords(ds, ex.x0, [a1, a2], w.rows[w.upEnd]), w.values[w.upEnd] * H + 0.35], kind: "note", appearAt: 7.6 });
  if (ex.at.walkDown && ex.at.walkDown.points.length > 1)
    out.labels.push({ text: "local min", pos: [...coords(ds, ex.x0, [a1, a2], w.rows[w.downEnd]), w.values[w.downEnd] * H - 0.35], kind: "note", appearAt: 7.6 });
  return out;
}

// ---------------------------------------------------------------------------------------------
// Feature-space views
// ---------------------------------------------------------------------------------------------

function field3(ds: LabDataset, ex: Explanation, dirs: number[][], n: number, fns: ((x: number[]) => number)[], offset4: number[] | null) {
  const us = linspace(-R, R, n);
  const fields = fns.map(() => new Float32Array(n * n * n));
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const c = [us[i], us[j], us[k]];
        let x = toRaw(ds, ex.x0, dirs, c);
        if (offset4) x = x.map((v, q) => v + offset4[q]);
        const idx = (k * n + j) * n + i;
        fns.forEach((fn, q) => (fields[q][idx] = fn(x)));
      }
  return fields;
}

function crosses(field: Float32Array, level: number): boolean {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of field) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo < level && hi > level;
}

/** Index of the shown row whose prediction is closest to 0.5 (an informative default instance). */
export function boundaryRow(ds: LabDataset, model: LabModel): number {
  let best = 0;
  let gap = Infinity;
  ds.rows.forEach((r, i) => {
    const g = Math.abs(model.fast(r) - 0.5);
    if (g < gap) {
      gap = g;
      best = i;
    }
  });
  return best;
}

function buildSpace3d(viz: Viz, ds: LabDataset, model: LabModel, ex: Explanation, ax: Axes, o: SceneOptions): BuiltScene {
  const out = empty();
  const dirs = ax.dirs.slice(0, 3);
  const n = 24;
  const lo = [-R, -R, -R];
  const hi = [R, R, R];
  const fns = [model.fast, (x: number[]) => ex.lime.surrogate.predict(x), (x: number[]) => ex.at.surrogate.predict(x)];
  const colors: RGBA[] = [withAlpha(COLORS.model, 0.5), withAlpha(COLORS.lime, 0.42), withAlpha(COLORS.atl, 0.5)];
  viz.clip(false);
  const frame = viz.box(lo, hi, 0.009, COLORS.box);
  const pts = ds.rows.map((r) => coords(ds, ex.x0, dirs, r));
  const keep = pts.map((_, i) => i).filter((i) => pts[i].every((c) => Math.abs(c) <= R));
  const cloud = dataPoints(viz, ds, pts, keep, 0.055);
  viz.scaleOrigin(cloud, [0, 0, 0]);
  out.pickObject = cloud;
  out.pickRows = keep;
  const x0 = singlePoint(viz, [0, 0, 0], COLORS.x0, 0.12);
  const w = walkPath(ex);
  const wxyz = w.rows.flatMap((r) => coords(ds, ex.x0, dirs, r));
  const walk = viz.tube(wxyz, 0.03, COLORS.atl);

  let isos: number[] = [];
  let noCrossing = false;
  let sweepInfo: BuiltScene["sweep"] = null;
  if (o.sweep && ds.features.length > 3) {
    const pc = principalComponents(ds);
    const [a4] = orthonormalize([...dirs, ...pc.vectors.slice(0, 4)], ds.features.length).slice(3, 4);
    const nt = 9;
    const ts = linspace(-2, 2, nt);
    const per = n * n * n;
    const all = fns.map(() => new Float32Array(nt * per));
    ts.forEach((tv, ti) => {
      const off = a4.map((v, j) => tv * v * ds.scale[j]);
      field3(ds, ex, dirs, n, fns, off).forEach((f, q) => all[q].set(f, ti * per));
    });
    noCrossing = !crosses(all[0], 0.5);
    isos = all.map((f, q) => viz.isosurface4d(f, nt, [n, n, n], 0.5, lo, hi, colors[q]));
    sweepInfo = { start: 10.2, dur: 9, nt, range: 2, name: "the next principal direction ⟂ the axes" };
  } else {
    const f = field3(ds, ex, dirs, n, fns, null);
    noCrossing = !crosses(f[0], 0.5);
    isos = f.map((fld, q) => viz.isosurface(fld, [n, n, n], 0.5, lo, hi, colors[q]));
    if (o.levels)
      f.forEach((fld, q) => {
        for (const lv of [0.25, 0.75]) {
          const id = viz.isosurface(fld, [n, n, n], lv, lo, hi, withAlpha(colors[q], 0.16));
          viz.set(id, PROP.ALPHA, 0);
          viz.tween(id, PROP.ALPHA, 0, 1, 9.6, 1.0);
        }
      });
  }
  viz.set(frame, PROP.ALPHA, 0);
  viz.set(cloud, PROP.ALPHA, 0);
  viz.set(x0, PROP.ALPHA, 0);
  viz.set(walk, PROP.REVEAL, 0);
  for (const id of isos) viz.set(id, PROP.REVEAL, 0);

  viz.tween(frame, PROP.ALPHA, 0, 1, 0, 0.8);
  viz.tween(cloud, PROP.ALPHA, 0, 1, 0.3, 0.8);
  viz.tween(cloud, PROP.SCALE, 0.2, 1, 0.3, 1.4);
  out.captions.push({ t: 0, text: `Rows projected onto the three axes (standardised units around x₀); pink = ${o.positive}.` });
  viz.tween(isos[0], PROP.REVEAL, 0, 1.01, 1.9, 2.0);
  out.captions.push({ t: 1.9, text: noCrossing
    ? "The model never reaches P = 0.5 inside this 3-D slice — click a point nearer the boundary (or pick another view) to see it."
    : "Grey: where the model's P = 0.5, on the 3-D slice through x₀ (the other features held at x₀)." });
  viz.tween(x0, PROP.ALPHA, 0, 1, 4.1, 0.4);
  viz.tween(x0, PROP.EMISSIVE, 0, 0.9, 4.1, 1.4, EASE.THERE_AND_BACK);
  viz.tween(isos[1], PROP.REVEAL, 0, 1.01, 5.0, 1.5);
  out.captions.push({ t: 5.0, text: "Blue: LIME's decision plane (where the linear surrogate equals 0.5)." });
  viz.tween(isos[2], PROP.REVEAL, 0, 1.01, 6.7, 1.5);
  out.captions.push({ t: 6.7, text: "Orange: ArcTanh-LIME's decision plane — also flat; its gain is in the values, not the 0.5 level set." });
  viz.tween(walk, PROP.REVEAL, 0, 1.01, 8.4, 1.6, EASE.LINEAR);
  out.captions.push({ t: 8.4, text: "The ε-walk, projected: it climbs straight across the boundary." });
  out.duration = 10.2;
  if (sweepInfo) {
    // centre (the slice through x0) -> +2 sigma -> -2 sigma -> centre
    const { start, dur, nt: k } = sweepInfo;
    const mid = (k - 1) / 2;
    for (const id of isos) {
      viz.tween(id, PROP.TIME, mid, k - 1, start, dur / 4);
      viz.tween(id, PROP.TIME, k - 1, 0, start + dur / 4, dur / 2);
      viz.tween(id, PROP.TIME, 0, mid, start + (3 * dur) / 4, dur / 4);
    }
    out.captions.push({ t: sweepInfo.start, text: "Sweeping a 4th dimension: the slice moves ±2σ along a fourth direction, and every boundary moves with it." });
    out.duration = sweepInfo.start + sweepInfo.dur;
  }
  out.sweep = sweepInfo;
  viz.camera(-2.3, 1.15, 16.5, [0, 0, 0]);
  viz.ortho(false);
  out.labels.push(
    { text: `${ax.names[0]} →`, pos: [R + 0.5, -R, -R], kind: "axis" },
    { text: `${ax.names[1]} →`, pos: [-R, R + 0.5, -R], kind: "axis" },
    { text: `${ax.names[2]} ↑`, pos: [-R, -R, R + 0.35], kind: "axis" },
    { text: "x₀", pos: [0, 0, 0.42], kind: "x0", appearAt: 4.1 },
  );
  return out;
}

function buildSpace2d(viz: Viz, ds: LabDataset, model: LabModel, ex: Explanation, ax: Axes, o: SceneOptions): BuiltScene {
  const out = empty();
  const dirs = ax.dirs.slice(0, 2);
  const n = 121;
  const us = linspace(-R, R, n);
  const fields = [new Float32Array(n * n), new Float32Array(n * n), new Float32Array(n * n)];
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const x = toRaw(ds, ex.x0, dirs, [us[i], us[j]]);
      fields[0][j * n + i] = model.fast(x);
      fields[1][j * n + i] = ex.lime.surrogate.predict(x);
      fields[2][j * n + i] = ex.at.surrogate.predict(x);
    }
  const box: [number, number, number, number] = [-R, R, -R, R];
  viz.clip(false);
  const grid = viz.floorGrid(box, -0.02, 1, 0.006, COLORS.grid);
  const frame = viz.box([-R, -R, -0.02], [R, R, -0.02], 0.01, COLORS.box);
  const colors = [COLORS.model, COLORS.lime, COLORS.atl];
  const lines = fields.map((f, q) => viz.isolines(f, n, n, 0.5, box, 0.02 + q * 0.01, 0.028, colors[q]));
  const extra: number[] = [];
  if (o.levels)
    fields.forEach((f, q) => {
      for (const lv of [0.25, 0.75]) extra.push(viz.isolines(f, n, n, lv, box, 0.015, 0.012, withAlpha(colors[q], 0.6)));
    });
  const pts = ds.rows.map((r) => [...coords(ds, ex.x0, dirs, r), 0]);
  const keep = pts.map((_, i) => i).filter((i) => Math.abs(pts[i][0]) <= R && Math.abs(pts[i][1]) <= R);
  const cloud = dataPoints(viz, ds, pts, keep, 0.045);
  out.pickObject = cloud;
  out.pickRows = keep;
  const x0 = singlePoint(viz, [0, 0, 0.06], COLORS.x0, 0.1);
  const w = walkPath(ex);
  const walk = viz.tube(w.rows.flatMap((r) => [...coords(ds, ex.x0, dirs, r), 0.05]), 0.022, COLORS.atl);
  for (const id of [grid, frame, cloud, x0, ...extra]) viz.set(id, PROP.ALPHA, 0);
  for (const id of [...lines, walk]) viz.set(id, PROP.REVEAL, 0);
  viz.tween(grid, PROP.ALPHA, 0, 1, 0, 0.6);
  viz.tween(frame, PROP.ALPHA, 0, 1, 0, 0.6);
  viz.tween(cloud, PROP.ALPHA, 0, 1, 0.2, 0.8);
  out.captions.push({ t: 0, text: `Rows projected onto two axes around x₀; pink = ${o.positive}.` });
  viz.tween(lines[0], PROP.REVEAL, 0, 1.01, 1.0, 1.6);
  out.captions.push({ t: 1.0, text: "White: the model's P = 0.5 contour on this slice." });
  viz.tween(x0, PROP.ALPHA, 0, 1, 2.6, 0.4);
  viz.tween(x0, PROP.EMISSIVE, 0, 0.9, 2.6, 1.2, EASE.THERE_AND_BACK);
  viz.tween(lines[1], PROP.REVEAL, 0, 1.01, 3.3, 1.2);
  viz.tween(lines[2], PROP.REVEAL, 0, 1.01, 4.6, 1.2);
  out.captions.push({ t: 3.3, text: "Blue: LIME's 0.5 line. Orange: ArcTanh-LIME's 0.5 line." });
  for (const id of extra) viz.tween(id, PROP.ALPHA, 0, 1, 6.0, 0.8);
  if (extra.length) out.captions.push({ t: 6.0, text: "Thin lines: the 0.25 and 0.75 levels — the tanh surrogate's levels bunch up like the model's; LIME's are evenly spaced." });
  viz.tween(walk, PROP.REVEAL, 0, 1.01, 6.2, 1.5, EASE.LINEAR);
  out.duration = 8;
  viz.camera(-Math.PI / 2, 0.0005, 20, [0, 0, 0]);
  viz.ortho(true, R * 1.12);
  out.labels.push(
    { text: `${ax.names[0]} →`, pos: [R - 0.2, -R - 0.32, 0], kind: "axis" },
    { text: `${ax.names[1]} ↑`, pos: [-R - 0.1, R + 0.25, 0], kind: "axis" },
    { text: "x₀", pos: [0.32, 0.32, 0.1], kind: "x0", appearAt: 2.6 },
  );
  return out;
}

// ---------------------------------------------------------------------------------------------
// Grand tour
// ---------------------------------------------------------------------------------------------

function buildTour(viz: Viz, ds: LabDataset, ex: Explanation, o: SceneOptions): BuiltScene {
  const out = empty();
  const d = ds.features.length;
  viz.clip(false);
  viz.ortho(false);
  if (d < 3) {
    out.captions.push({ t: 0, text: "The grand tour needs at least three features — try the 2-D or 3-D views." });
    return out;
  }
  const pc = principalComponents(ds);
  const rng = new Rng(2024);
  const frames: number[][][] = [orthonormalize(pc.vectors.slice(0, 3), d)];
  for (let k = 0; k < 5; k++) {
    const rand = Array.from({ length: 3 }, () => Array.from({ length: d }, () => rng.normal()));
    frames.push(orthonormalize(rand, d));
  }
  const keyframes = Float32Array.from(frames.flatMap((f) => f.flat()));
  const rows = standardRows(ds, ds.mean);
  const x0z = ex.x0.map((v, j) => (v - ds.mean[j]) / ds.scale[j]);
  const X = Float32Array.from([...rows.flat(), ...x0z]);
  const rgba = Float32Array.from([...ds.labels.flatMap((l) => (l > 0.5 ? COLORS.pos : COLORS.neg)), ...COLORS.x0]);
  const cloud = viz.tourBegin(X, rows.length + 1, d, rgba, 0.06, keyframes, frames.length, 1);
  out.pickObject = cloud;
  out.pickRows = rows.map((_, i) => i);
  const w = walkPath(ex);
  viz.tourWalk(Float32Array.from(w.rows.flatMap((r) => r.map((v, j) => (v - ds.mean[j]) / ds.scale[j]))), w.rows.length, COLORS.atl);
  const lime = ex.lime.attribution;
  const atl = ex.at.attribution;
  const big = Math.max(normv(lime), normv(atl), 1e-12);
  viz.tourArrows(Float32Array.from(x0z), Float32Array.from([...lime, ...atl].map((v) => (v / big) * 2.8)), 2,
    Float32Array.from([...COLORS.lime, ...COLORS.atl]));
  viz.tourAxes(3.4);
  viz.tourSpeed(0.11);
  const top = atl.map((v, j) => [Math.abs(v), j]).sort((a, b) => b[0] - a[0]).slice(0, Math.min(6, d));
  out.tourAxisLabels = top.map(([, j]) => ({ j, text: ds.features[j] }));
  out.captions.push({ t: 0, text: `Grand tour through all ${d} features. Arrows at x₀: LIME (blue) vs ArcTanh-LIME (orange); orange path: the ε-walk; spokes: feature axes.` });
  out.duration = 0;
  viz.camera(-2.0, 1.2, 12, [0, 0, 0]);
  return out;
}

export function buildScene(viz: Viz, ds: LabDataset, model: LabModel, ex: Explanation, o: SceneOptions): BuiltScene {
  viz.clear();
  viz.setTime(0);
  viz.play(true, 1);
  const count = o.mode === "surface" || o.mode === "space2d" ? 2 : 3;
  const ax = chooseAxes(ds, ex, o.axisMode, o.features, count);
  if (o.mode === "surface") return buildSurface(viz, ds, model, ex, ax, o);
  if (o.mode === "space3d") return buildSpace3d(viz, ds, model, ex, ax, o);
  if (o.mode === "space2d") return buildSpace2d(viz, ds, model, ex, ax, o);
  return buildTour(viz, ds, ex, o);
}

/** Local fidelity of the two surrogates on fresh points at radius r (standardised) around x0. */
export function localFidelity(ds: LabDataset, model: LabModel, ex: Explanation, r: number, m = 200) {
  const rng = new Rng(99);
  const d = ds.features.length;
  let el = 0;
  let ea = 0;
  for (let k = 0; k < m; k++) {
    const u = Array.from({ length: d }, () => rng.normal());
    const nu = normv(u) || 1;
    const x = ex.x0.map((v, j) => Math.min(Math.max(v + (r * u[j] * ds.scale[j]) / nu, ds.bounds[j][0]), ds.bounds[j][1]));
    const f = model.predict(x);
    el += (ex.lime.surrogate.predict(x) - f) ** 2;
    ea += (ex.at.surrogate.predict(x) - f) ** 2;
  }
  return { lime: el / m, arctanh: ea / m };
}

export type { Surrogate };
