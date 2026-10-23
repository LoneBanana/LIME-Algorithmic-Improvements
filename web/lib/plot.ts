/** Tiny plotting helpers (scales, ticks, paths) for the hand-written SVG charts. */

export interface Scale {
  (v: number): number;
  invert(p: number): number;
  domain: [number, number];
  range: [number, number];
}

export function scaleLinear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = (r1 - r0) / (d1 - d0 || 1);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.invert = (p: number) => d0 + (p - r0) / k;
  s.domain = domain;
  s.range = range;
  return s;
}

export function scaleLog(domain: [number, number], range: [number, number]): Scale {
  const l0 = Math.log10(domain[0]);
  const l1 = Math.log10(domain[1]);
  const lin = scaleLinear([l0, l1], range);
  const s = ((v: number) => lin(Math.log10(Math.max(v, 1e-300)))) as Scale;
  s.invert = (p: number) => 10 ** lin.invert(p);
  s.domain = domain;
  s.range = range;
  return s;
}

/** "Nice" linear ticks (steps of 1, 2 or 5 x 10^k). */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.07 ? 10 : norm >= 3.16 ? 5 : norm >= 1.41 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
  }
  return out;
}

/** Decade ticks for a log axis (adds 2x and 5x when the range is narrow). */
export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const out: number[] = [];
  const mult = hi - lo <= 2 ? [1, 2, 5] : [1];
  for (let e = lo; e <= hi; e++) {
    for (const m of mult) {
      const v = m * 10 ** e;
      if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
    }
  }
  return out;
}

export function fmtNum(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a < 1e-3) return v.toExponential(1).replace("e-", "e−");
  return (+v.toPrecision(digits)).toString().replace("-", "−");
}

export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (v === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace("e-", "e−").replace("e+", "e");
  return (+v.toPrecision(6)).toString().replace("-", "−");
}

/** SVG path through points; NaN / non-finite y values break the line. */
export function linePath(xs: number[], ys: number[]): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    pen = true;
  }
  return d;
}

export function linspace(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Pointer position in an SVG's viewBox coordinates. */
export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): { x: number; y: number } {
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: 0, y: 0 };
  const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
  return { x: p.x, y: p.y };
}

/** Linear interpolation between two hex colours. */
export function mixHex(a: string, b: string, t: number): [number, number, number] {
  const pa = hexToRgb(a);
  const pb = hexToRgb(b);
  return [0, 1, 2].map((i) => Math.round(pa[i] + (pb[i] - pa[i]) * t)) as [number, number, number];
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "").trim();
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Ratios: two decimals (three significant digits below 0.1). */
export function fmtRatio(v: number): string {
  if (!Number.isFinite(v)) return "–";
  if (v >= 10) return `${v.toFixed(0)}×`;
  if (v < 0.1) return `${v.toPrecision(2)}×`;
  return `${v.toFixed(2)}×`;
}

/**
 * A paired difference worth highlighting: Wilcoxon p < 0.05, means at least 5% apart, and the rank test
 * (sign of z, negative = A lower) pointing the same way as the ratio of means.
 */
export function meaningful(ratio: number | null, p: number | null, z: number | null): boolean {
  if (ratio === null || p === null || z === null) return false;
  return p < 0.05 && Math.abs(ratio - 1) >= 0.05 && z < 0 === ratio < 1;
}
