/**
 * Datasets for the 3-D lab: bundled presets (exported by `python3 -m experiments.export_web`)
 * and arbitrary CSV files, which are parsed, encoded and modelled entirely in the browser.
 */
import { gbmPredictor, mlpPredictor, Rng } from "@/lib/core/index.ts";
import type { GbmWeights, MlpWeights } from "@/lib/core/index.ts";
import { trainMlp } from "./train";

export interface LabModel {
  key: string;
  title: string;
  /** exact (parity-tested) predictor, used by the explainers */
  predict: (x: number[]) => number;
  /** same model with plain floating-point sums, for the hundreds of thousands of grid evaluations */
  fast: (x: number[]) => number;
  smooth: boolean;
}

export function fastPredictor(w: MlpWeights | GbmWeights): (x: number[]) => number {
  const sig = (v: number) => (v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v)));
  if (w.type === "gradient_boosting") {
    const g = w as GbmWeights;
    return (x) => {
      let s = 0;
      for (const t of g.trees) {
        let node = 0;
        while (t.feature[node] >= 0) node = x[t.feature[node]] <= t.threshold[node] ? t.left[node] : t.right[node];
        s += t.value[node];
      }
      return sig(g.f0 + g.learning_rate * s);
    };
  }
  const m = w as MlpWeights;
  const last = m.W.length - 1;
  return (x) => {
    let a = x.map((v, j) => (v - m.x_mean[j]) / m.x_std[j]);
    for (let layer = 0; layer <= last; layer++) {
      const W = m.W[layer];
      const b = m.b[layer];
      const z = new Array<number>(W.length);
      for (let i = 0; i < W.length; i++) {
        let s = b[i];
        const row = W[i];
        for (let j = 0; j < row.length; j++) s += row[j] * a[j];
        z[i] = layer === last ? s : Math.tanh(s);
      }
      a = z;
    }
    return m.output === "sigmoid" ? sig(a[0]) : m.y_mean + m.y_std * a[0];
  };
}

export interface LabDataset {
  key: string;
  title: string;
  attribution: string;
  features: string[];
  mean: number[];
  scale: number[];
  bounds: [number, number][];
  rows: number[][];
  labels: number[];
  positive: string;
  models: LabModel[];
  notes: string[];
}

interface PresetJson {
  key: string;
  title: string;
  attribution: string;
  features: string[];
  mean: number[];
  scale: number[];
  bounds: [number, number][];
  rows: number[][];
  labels: number[];
  positive: string;
  models: Record<string, MlpWeights | GbmWeights>;
  model_titles: Record<string, string>;
}

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export async function loadPresets(): Promise<LabDataset[]> {
  const res = await fetch(`${BASE}/lab/presets.json`);
  if (!res.ok) throw new Error("could not load the bundled datasets");
  const json = (await res.json()) as { datasets: PresetJson[] };
  return json.datasets.map((p) => ({
    key: p.key,
    title: p.title,
    attribution: p.attribution,
    features: p.features,
    mean: p.mean,
    scale: p.scale,
    bounds: p.bounds,
    rows: p.rows,
    labels: p.labels,
    positive: p.positive,
    notes: [],
    models: Object.entries(p.models).map(([key, w]) => ({
      key,
      title: p.model_titles[key] ?? key,
      predict: w.type === "gradient_boosting" ? gbmPredictor(w as GbmWeights) : mlpPredictor(w as MlpWeights),
      fast: fastPredictor(w),
      smooth: w.type !== "gradient_boosting",
    })),
  }));
}

// ---------------------------------------------------------------------------------------------
// CSV upload
// ---------------------------------------------------------------------------------------------

/** RFC-4180-ish CSV parser (quotes, escaped quotes, CRLF); delimiter sniffed from the header. */
export function parseCsv(text: string): { header: string[]; cells: string[][] } {
  const firstLine = text.slice(0, text.indexOf("\n") === -1 ? text.length : text.indexOf("\n"));
  const delim = [",", ";", "\t"].reduce((best, d) => (firstLine.split(d).length > firstLine.split(best).length ? d : best), ",");
  const records: string[][] = [];
  let field = "";
  let record: string[] = [];
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === delim) {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      record.push(field);
      if (record.some((c) => c.trim() !== "")) records.push(record);
      record = [];
      field = "";
    } else field += ch;
  }
  record.push(field);
  if (record.some((c) => c.trim() !== "")) records.push(record);
  if (records.length < 2) throw new Error("the file needs a header row and at least one data row");
  const header = records[0].map((h) => h.trim());
  return { header, cells: records.slice(1).map((r) => header.map((_, j) => (r[j] ?? "").trim())) };
}

const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-"]);
const isMissing = (s: string) => MISSING.has(s.toLowerCase());
const asNumber = (s: string) => (s.trim() === "" ? NaN : Number(s));
const POSITIVE = new Set(["1", "yes", "true", "positive", "pos", "y", "t", "diabetic", "diabetes", "tested_positive"]);

export interface PreparedTable {
  features: string[];
  X: number[][];
  y: number[];
  positive: string;
  notes: string[];
}

/** Encode features (numeric / binary / one-hot, median-imputed) and binarise the target column. */
export function prepareTable(header: string[], cells: string[][], target: number): PreparedTable {
  const notes: string[] = [];
  const rawTarget = cells.map((r) => r[target]);
  const keep = rawTarget.map((v) => !isMissing(v));
  const rows = cells.filter((_, i) => keep[i]);
  if (rows.length < cells.length) notes.push(`${cells.length - rows.length} rows without a target were dropped.`);
  const tvals = rows.map((r) => r[target]);
  const levels = [...new Set(tvals)];
  let y: number[];
  let positive: string;
  const numericTarget = tvals.every((v) => Number.isFinite(asNumber(v)));
  if (levels.length < 2) throw new Error("the target column has a single value");
  if (levels.length === 2) {
    const named = levels.find((l) => POSITIVE.has(l.toLowerCase()));
    const counts = levels.map((l) => tvals.filter((v) => v === l).length);
    positive = named ?? (numericTarget ? levels.sort((a, b) => asNumber(a) - asNumber(b))[1] : levels[counts[0] <= counts[1] ? 0 : 1]);
    y = tvals.map((v) => (v === positive ? 1 : 0));
  } else if (numericTarget) {
    const sorted = tvals.map(asNumber).sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)];
    positive = `${header[target]} > ${med}`;
    y = tvals.map((v) => (asNumber(v) > med ? 1 : 0));
    notes.push(`Numeric target binarised at its median (${med}).`);
  } else {
    const freq = new Map<string, number>();
    for (const v of tvals) freq.set(v, (freq.get(v) ?? 0) + 1);
    positive = [...freq.entries()].sort((a, b) => b[1] - a[1])[0][0];
    y = tvals.map((v) => (v === positive ? 1 : 0));
    notes.push(`Multi-class target: "${positive}" (most frequent) vs the rest.`);
  }

  const features: string[] = [];
  const columns: number[][] = [];
  header.forEach((name, j) => {
    if (j === target) return;
    const col = rows.map((r) => r[j]);
    const present = col.filter((v) => !isMissing(v));
    if (present.length === 0) {
      notes.push(`Column "${name}" is empty and was dropped.`);
      return;
    }
    const numeric = present.filter((v) => Number.isFinite(asNumber(v))).length >= 0.95 * present.length;
    if (numeric) {
      const vals = present.map(asNumber).filter(Number.isFinite).sort((a, b) => a - b);
      const med = vals[Math.floor(vals.length / 2)];
      const missing = col.filter(isMissing).length;
      if (missing) notes.push(`"${name}": ${missing} missing values set to the median (${med}).`);
      features.push(name);
      columns.push(col.map((v) => (isMissing(v) || !Number.isFinite(asNumber(v)) ? med : asNumber(v))));
      return;
    }
    const lv = [...new Set(present)];
    if (lv.length === 2) {
      const one = lv.find((l) => POSITIVE.has(l.toLowerCase())) ?? lv.sort()[1];
      features.push(`${name}=${one}`);
      columns.push(col.map((v) => (v === one ? 1 : 0)));
    } else if (lv.length <= 12) {
      for (const level of lv.sort()) {
        features.push(`${name}=${level}`);
        columns.push(col.map((v) => (v === level ? 1 : 0)));
      }
      notes.push(`"${name}" one-hot encoded (${lv.length} levels).`);
    } else {
      notes.push(`"${name}" has ${lv.length} text values and was dropped.`);
    }
  });
  if (!features.length) throw new Error("no usable feature columns");
  let X = rows.map((_, i) => columns.map((c) => c[i]));
  if (X.length > 4000) {
    const rng = new Rng(4000);
    const order = X.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const k = rng.randint(i + 1);
      [order[i], order[k]] = [order[k], order[i]];
    }
    const pick = order.slice(0, 4000);
    notes.push(`Random sample of 4,000 of ${X.length.toLocaleString("en-US")} rows used.`);
    y = pick.map((i) => y[i]);
    X = pick.map((i) => X[i]);
  }
  return { features, X, y, positive, notes };
}

export function columnStats(X: number[][]) {
  const d = X[0].length;
  const n = X.length;
  const mean = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const scale = mean.map((m, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / n) || 1);
  const bounds = mean.map((_, j) => [Math.min(...X.map((r) => r[j])), Math.max(...X.map((r) => r[j]))] as [number, number]);
  return { mean, scale, bounds };
}

/** Build a dataset from a parsed table: train an MLP in the browser (80/20 split) as the black box. */
export function datasetFromTable(name: string, t: PreparedTable, onProgress?: (epoch: number) => void): {
  dataset: LabDataset;
  accuracy: number;
} {
  const rng = new Rng(77);
  const order = t.X.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const k = rng.randint(i + 1);
    [order[i], order[k]] = [order[k], order[i]];
  }
  const nTrain = Math.max(1, Math.round(order.length * 0.8));
  const tr = order.slice(0, nTrain);
  const te = order.slice(nTrain);
  const Xtr = tr.map((i) => t.X[i]);
  const ytr = tr.map((i) => t.y[i]);
  const weights = trainMlp(Xtr, ytr, { hidden: 16, epochs: 60, lr: 0.01, seed: 5, onProgress });
  const predict = mlpPredictor(weights);
  const testIdx = te.length ? te : tr;
  const accuracy = testIdx.filter((i) => (predict(t.X[i]) > 0.5) === (t.y[i] > 0.5)).length / testIdx.length;
  const { mean, scale, bounds } = columnStats(Xtr);
  const show = [...te, ...tr].slice(0, 600);
  return {
    accuracy,
    dataset: {
      key: "upload",
      title: name,
      attribution: "Your file — parsed, modelled and explained locally in this browser tab; nothing is uploaded.",
      features: t.features,
      mean,
      scale,
      bounds,
      rows: show.map((i) => t.X[i]),
      labels: show.map((i) => t.y[i]),
      positive: t.positive,
      notes: [...t.notes, `MLP (${t.features.length}-16-1) trained here: held-out accuracy ${(accuracy * 100).toFixed(1)}%.`],
      models: [{ key: "mlp", title: "MLP (trained in your browser)", predict, fast: fastPredictor(weights), smooth: true }],
    },
  };
}

export { pca } from "./pca";
