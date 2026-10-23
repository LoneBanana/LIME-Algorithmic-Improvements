/** Runs all four explainers on a 1-D curve with the same settings as the Python benchmarks. */
import {
  ArcTanhLime, DeltaAnchorEnsemble, LimeTabular, ONE_D_DOMAIN, ONE_D_FUNCTIONS, Rng,
} from "./core/index.ts";
import type { ArcTanhResult, Bounds, EnsembleResult, LimeResult, OneDName } from "./core/index.ts";
import { linspace } from "./plot.ts";

export interface OneDSettings {
  fn: OneDName;
  x0: number;
  epsilon: number;
  delta: number;
  numSamples: number;
  kernelWidth: number;
  seed: number;
}

export const DEFAULT_ONE_D: OneDSettings = {
  fn: "sine",
  x0: 0.6,
  epsilon: 0.1,
  delta: 1,
  numSamples: 1000,
  kernelWidth: 0.75,
  seed: 1,
};

export interface OneDRun {
  settings: OneDSettings;
  g: (x: number) => number;
  lime: LimeResult;
  arctanh: ArcTanhResult;
  limeDelta: EnsembleResult<LimeResult>;
  arctanhDelta: EnsembleResult<ArcTanhResult>;
  maxSteps: number;
}

export function runOneD(s: OneDSettings): OneDRun {
  const g = ONE_D_FUNCTIONS[s.fn];
  const f = (x: number[]) => g(x[0]);
  const bounds: Bounds = [ONE_D_DOMAIN];
  // max_steps = 40 at the default epsilon = 0.1, i.e. walks may travel up to 4 standard deviations
  const maxSteps = Math.max(1, Math.round(4 / s.epsilon));
  const limeCfg = { numSamples: s.numSamples, kernelWidth: s.kernelWidth };
  const atCfg = { numSamples: s.numSamples, kernelWidth: s.kernelWidth, epsilon: s.epsilon, maxSteps };
  const x0 = [s.x0];
  return {
    settings: s,
    g,
    maxSteps,
    lime: new LimeTabular([1], limeCfg).explain(f, x0, new Rng(s.seed)),
    arctanh: new ArcTanhLime([1], atCfg, bounds).explain(f, x0, new Rng(s.seed)),
    limeDelta: new DeltaAnchorEnsemble(new LimeTabular([1], limeCfg), { delta: s.delta }, bounds)
      .explain(f, x0, new Rng(s.seed)),
    arctanhDelta: new DeltaAnchorEnsemble(new ArcTanhLime([1], atCfg, bounds), { delta: s.delta }, bounds)
      .explain(f, x0, new Rng(s.seed)),
  };
}

export type MethodKey = "lime" | "arctanh" | "limeDelta" | "arctanhDelta";

export const METHOD_META: Record<MethodKey, { label: string; color: string; dashed: boolean }> = {
  lime: { label: "LIME", color: "var(--lime)", dashed: false },
  arctanh: { label: "ArcTanh-LIME", color: "var(--atl)", dashed: false },
  limeDelta: { label: "LIME-δ", color: "var(--lime)", dashed: true },
  arctanhDelta: { label: "ArcTanh-LIME-δ", color: "var(--atl)", dashed: true },
};

export function predictor(run: OneDRun, key: MethodKey): (x: number) => number {
  const s = run[key].surrogate;
  return (x: number) => s.predict([x]);
}

/** Mean squared error of a surrogate on a grid of +/- radius around x0 (clipped to the domain). */
export function windowMse(run: OneDRun, key: MethodKey, radius: number): number {
  const [lo, hi] = ONE_D_DOMAIN;
  const xs = linspace(Math.max(lo, run.settings.x0 - radius), Math.min(hi, run.settings.x0 + radius), 201);
  const p = predictor(run, key);
  let s = 0;
  for (const x of xs) s += (p(x) - run.g(x)) ** 2;
  return s / xs.length;
}
