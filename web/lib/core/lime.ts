/** Port of atlime/lime.py (continuous tabular LIME; the web demos use all features). */
import { weightedR2, weightedRidge } from "./linalg.ts";
import { sumprod } from "./numeric.ts";
import type { Rng } from "./rng.ts";
import { LinearSurrogate } from "./surrogates.ts";

export type BlackBox = (x: number[]) => number;

export function defaultKernelWidth(nFeatures: number): number {
  return 0.75 * Math.sqrt(nFeatures);
}

export function limeKernel(distance: number, width: number): number {
  return Math.sqrt(Math.exp(-(distance * distance) / (width * width)));
}

export interface Neighborhood {
  points: number[][];
  Z: number[][];
  y: number[];
  w: number[];
}

export function sampleNeighborhood(
  f: BlackBox,
  x0: number[],
  scale: number[],
  numSamples: number,
  kernelWidth: number,
  rng: Rng,
): Neighborhood {
  const d = x0.length;
  const points = [x0.slice()];
  const Z = [new Array<number>(d).fill(0)];
  for (let k = 0; k < numSamples - 1; k++) {
    const z: number[] = [];
    for (let j = 0; j < d; j++) z.push(rng.normal());
    Z.push(z);
    points.push(x0.map((xi, j) => xi + z[j] * scale[j]));
  }
  const y = points.map((p) => f(p));
  const w = Z.map((z) => limeKernel(Math.sqrt(sumprod(z, z)), kernelWidth));
  return { points, Z, y, w };
}

export interface LimeConfig {
  numSamples: number;
  kernelWidth: number | null;
  alpha: number;
}

export const DEFAULT_LIME: LimeConfig = { numSamples: 1000, kernelWidth: null, alpha: 1 };

export interface LimeResult {
  surrogate: LinearSurrogate;
  score: number;
  localPred: number;
  neighborhood: Neighborhood;
  nQueries: number;
  attribution: number[];
}

export class LimeTabular {
  scale: number[];
  config: LimeConfig;

  constructor(scale: number[], config: Partial<LimeConfig> = {}) {
    this.scale = scale.slice();
    this.config = { ...DEFAULT_LIME, ...config };
  }

  kernelWidth(): number {
    return this.config.kernelWidth ?? defaultKernelWidth(this.scale.length);
  }

  neighborhood(f: BlackBox, x0: number[], rng: Rng): Neighborhood {
    return sampleNeighborhood(f, x0, this.scale, this.config.numSamples, this.kernelWidth(), rng);
  }

  fit(x0: number[], nb: Neighborhood): LimeResult {
    const { beta, intercept } = weightedRidge(nb.Z, nb.y, nb.w, this.config.alpha);
    const surrogate = new LinearSurrogate(x0, this.scale, intercept, beta);
    const pred = nb.Z.map((z) => intercept + sumprod(beta, z));
    return {
      surrogate,
      score: weightedR2(nb.y, pred, nb.w),
      localPred: intercept,
      neighborhood: nb,
      nQueries: nb.y.length,
      attribution: beta.slice(),
    };
  }

  explain(f: BlackBox, x0: number[], rng: Rng): LimeResult {
    return this.fit(x0, this.neighborhood(f, x0, rng));
  }
}
