/** Port of atlime/arctanh.py: epsilon-walk to the local extrema + tanh^-1 regressor. */
import { defaultKernelWidth, limeKernel, sampleNeighborhood } from "./lime.ts";
import type { BlackBox, Neighborhood } from "./lime.ts";
import { solveSpd, weightedR2, weightedRidge } from "./linalg.ts";
import { sumprod } from "./numeric.ts";
import type { Rng } from "./rng.ts";
import { LinearSurrogate, standardize, TanhSurrogate } from "./surrogates.ts";
import { epsilonWalk, finiteDifferenceGradient } from "./walk.ts";
import type { Bounds, WalkResult } from "./walk.ts";

export interface ArcTanhConfig {
  numSamples: number;
  kernelWidth: number | null;
  alpha: number;
  epsilon: number;
  maxSteps: number;
  gradStep: number | null;
  tolRel: number;
  asymptotes: "walk" | "walk+samples" | "fixed";
  fixedBounds: [number, number] | null;
  margin: number;
  clip: number;
  fit: "link" | "output";
  pinOrigin: boolean;
  useWalkPoints: boolean;
  lmMaxIter: number;
}

export const DEFAULT_ARCTANH: ArcTanhConfig = {
  numSamples: 1000,
  kernelWidth: null,
  alpha: 1,
  epsilon: 0.1,
  maxSteps: 40,
  gradStep: null,
  tolRel: 1e-3,
  asymptotes: "walk",
  fixedBounds: null,
  margin: 0.05,
  clip: 1e-3,
  fit: "output",
  pinOrigin: false,
  useWalkPoints: true,
  lmMaxIter: 50,
};

export interface ArcTanhResult {
  surrogate: TanhSurrogate | LinearSurrogate;
  linkSurrogate: TanhSurrogate | LinearSurrogate;
  walkUp: WalkResult | null;
  walkDown: WalkResult | null;
  rawLo: number;
  rawHi: number;
  lo: number;
  hi: number;
  score: number;
  localPred: number;
  clippedFraction: number;
  fallback: string | null;
  neighborhood: Neighborhood;
  /** regression data actually used (LIME sample + walk path) */
  fitData: { Z: number[][]; y: number[]; w: number[] };
  nQueries: number;
  lmIterations: number;
  attribution: number[];
}

function arrayMin(v: number[]): number {
  let m = Infinity;
  for (const x of v) if (x < m) m = x;
  return m;
}

function arrayMax(v: number[]): number {
  let m = -Infinity;
  for (const x of v) if (x > m) m = x;
  return m;
}

function tanhLoss(
  Z: number[][], y: number[], w: number[], c: number, a: number, b0: number, beta: number[], penalty: number,
): number {
  let s = 0;
  for (let i = 0; i < Z.length; i++) {
    const r = y[i] - c - a * Math.tanh(b0 + sumprod(beta, Z[i]));
    s += w[i] * r * r;
  }
  return s + penalty * sumprod(beta, beta);
}

/** Levenberg-Marquardt for  min sum w (y - c - a tanh(b0 + beta.z))^2 + penalty ||beta||^2. */
export function fitTanhOutput(
  Z: number[][], y: number[], w: number[], c: number, a: number, beta0: number, betaInit: number[],
  penalty: number, pinOrigin: boolean, maxIter = 50,
): { beta0: number; beta: number[]; iterations: number; loss: number } {
  const d = betaInit.length;
  let b0 = beta0;
  let beta = betaInit.slice();
  let loss = tanhLoss(Z, y, w, c, a, b0, beta, penalty);
  let lam = 1e-3;
  const offset = pinOrigin ? 0 : 1;
  const p = d + offset;
  let iters = 0;
  for (iters = 1; iters <= maxIter; iters++) {
    const gram = Array.from({ length: p }, () => new Array<number>(p).fill(0));
    const rhs = new Array<number>(p).fill(0);
    for (let i = 0; i < Z.length; i++) {
      const z = Z[i];
      const t = Math.tanh(b0 + sumprod(beta, z));
      const s = a * (1 - t * t);
      const r = y[i] - c - a * t;
      const row = offset ? [1, ...z] : z.slice();
      const ws = w[i] * s;
      const ws2 = ws * s;
      for (let j = 0; j < p; j++) {
        const rj = row[j];
        rhs[j] += ws * r * rj;
        const gj = gram[j];
        const kVal = ws2 * rj;
        for (let k = j; k < p; k++) gj[k] += kVal * row[k];
      }
    }
    for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) gram[j][k] = gram[k][j];
    for (let j = 0; j < d; j++) {
      gram[j + offset][j + offset] += penalty;
      rhs[j + offset] -= penalty * beta[j];
    }
    let improved = false;
    while (lam <= 1e10) {
      const damped = gram.map((row) => row.slice());
      for (let j = 0; j < p; j++) damped[j][j] += lam * Math.max(gram[j][j], 1e-12);
      const delta = solveSpd(damped, rhs);
      const nb0 = offset ? b0 + delta[0] : b0;
      const nbeta = beta.map((bj, j) => bj + delta[j + offset]);
      const newLoss = tanhLoss(Z, y, w, c, a, nb0, nbeta, penalty);
      if (newLoss < loss) {
        const gain = loss - newLoss;
        b0 = nb0;
        beta = nbeta;
        loss = newLoss;
        lam = Math.max(lam / 3, 1e-12);
        improved = gain > 1e-12 * Math.max(loss, 1e-300);
        break;
      }
      lam *= 4;
    }
    if (!improved) break;
  }
  return { beta0: b0, beta, iterations: Math.min(iters, maxIter), loss };
}

export class ArcTanhLime {
  scale: number[];
  config: ArcTanhConfig;
  bounds: Bounds;

  constructor(scale: number[], config: Partial<ArcTanhConfig> = {}, bounds: Bounds = null) {
    this.scale = scale.slice();
    this.config = { ...DEFAULT_ARCTANH, ...config };
    this.bounds = bounds;
  }

  kernelWidth(): number {
    return this.config.kernelWidth ?? defaultKernelWidth(this.scale.length);
  }

  explain(f: BlackBox, x0: number[], rng: Rng): ArcTanhResult {
    const cfg = this.config;
    const d = x0.length;
    const nb = sampleNeighborhood(f, x0, this.scale, cfg.numSamples, this.kernelWidth(), rng);
    let nQueries = nb.y.length;
    const f0 = nb.y[0];

    // --- step 2: asymptotes from the epsilon-walks
    let walkUp: WalkResult | null = null;
    let walkDown: WalkResult | null = null;
    let rawLo: number;
    let rawHi: number;
    if (cfg.asymptotes === "fixed") {
      if (cfg.fixedBounds === null) throw new Error("asymptotes='fixed' needs fixedBounds");
      [rawLo, rawHi] = cfg.fixedBounds;
    } else {
      const h = cfg.gradStep ?? cfg.epsilon;
      // both walks share the gradient at x0; its queries are counted in the up walk
      const g0 = cfg.maxSteps > 0 ? finiteDifferenceGradient(f, x0, this.scale, h, this.bounds) : { grad: [], queries: 0 };
      walkUp = epsilonWalk(f, x0, f0, this.scale, 1, cfg.epsilon, cfg.maxSteps, h, cfg.tolRel, this.bounds, g0.grad);
      walkDown = epsilonWalk(f, x0, f0, this.scale, -1, cfg.epsilon, cfg.maxSteps, h, cfg.tolRel, this.bounds, g0.grad);
      walkUp.nQueries += g0.queries;
      nQueries += walkUp.nQueries + walkDown.nQueries;
      rawLo = walkDown.values[walkDown.values.length - 1];
      rawHi = walkUp.values[walkUp.values.length - 1];
      if (cfg.asymptotes === "walk+samples") {
        rawLo = Math.min(rawLo, arrayMin(nb.y));
        rawHi = Math.max(rawHi, arrayMax(nb.y));
      }
    }

    // --- regression data: LIME sample (+ walk path)
    const Z = nb.Z.map((z) => z.slice());
    const y = nb.y.slice();
    const w = nb.w.slice();
    const kw = this.kernelWidth();
    if (cfg.useWalkPoints) {
      for (const walk of [walkUp, walkDown]) {
        if (walk === null) continue;
        for (let k = 1; k < walk.points.length; k++) {
          const z = standardize(walk.points[k], x0, this.scale);
          Z.push(z);
          y.push(walk.values[k]);
          w.push(limeKernel(Math.sqrt(sumprod(z, z)), kw));
        }
      }
    }

    let lo = rawLo;
    let hi = rawHi;
    let fallback: string | null = null;
    const tiny = 1e-12 * Math.max(1, Math.abs(f0));
    if (hi - lo <= tiny) {
      lo = Math.min(arrayMin(y), lo);
      hi = Math.max(arrayMax(y), hi);
      fallback = "sample-range";
    }
    const base = { walkUp, walkDown, rawLo, rawHi, neighborhood: nb, fitData: { Z, y, w }, nQueries };
    if (hi - lo <= tiny) {
      const constant = new LinearSurrogate(x0, this.scale, f0, new Array<number>(d).fill(0));
      return {
        ...base, surrogate: constant, linkSurrogate: constant, lo, hi,
        score: weightedR2(nb.y, nb.y.map(() => f0), nb.w), localPred: f0, clippedFraction: 0,
        fallback: "constant", lmIterations: 0, attribution: new Array<number>(d).fill(0),
      };
    }
    const span = hi - lo;
    lo -= cfg.margin * span;
    hi += cfg.margin * span;
    const c = 0.5 * (hi + lo);
    const a = 0.5 * (hi - lo);

    // --- step 3: tanh^-1 (link-space) regression
    const lim = 1 - cfg.clip;
    const u = y.map((yi) => (yi - c) / a);
    const clippedFraction = u.filter((v) => Math.abs(v) > lim).length / u.length;
    const t = u.map((v) => Math.atanh(Math.max(-lim, Math.min(lim, v))));
    let b0: number;
    let beta: number[];
    if (cfg.pinOrigin) {
      b0 = Math.atanh(Math.max(-lim, Math.min(lim, (f0 - c) / a)));
      beta = weightedRidge(Z, t.map((ti) => ti - b0), w, cfg.alpha, false).beta;
    } else {
      const fit = weightedRidge(Z, t, w, cfg.alpha);
      beta = fit.beta;
      b0 = fit.intercept;
    }
    const linkSurrogate = new TanhSurrogate(x0, this.scale, lo, hi, b0, beta);

    // --- step 4: refinement in output space
    let surrogate = linkSurrogate;
    let lmIterations = 0;
    if (cfg.fit === "output") {
      const r = fitTanhOutput(Z, y, w, c, a, b0, beta, cfg.alpha * a * a, cfg.pinOrigin, cfg.lmMaxIter);
      surrogate = new TanhSurrogate(x0, this.scale, lo, hi, r.beta0, r.beta);
      lmIterations = r.iterations;
    }
    const pred = nb.Z.map((z) => surrogate.center + surrogate.halfRange * Math.tanh(surrogate.beta0 + sumprod(surrogate.beta, z)));
    return {
      ...base, surrogate, linkSurrogate, lo, hi, score: weightedR2(nb.y, pred, nb.w), localPred: pred[0],
      clippedFraction, fallback, lmIterations, attribution: surrogate.gradient(x0),
    };
  }
}
