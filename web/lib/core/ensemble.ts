/** Port of atlime/ensemble.py: the +/- delta anchor ensemble. */
import type { BlackBox } from "./lime.ts";
import type { Rng } from "./rng.ts";
import { BlendedSurrogate, LinearSurrogate, TanhSurrogate } from "./surrogates.ts";
import type { Bounds } from "./walk.ts";

export interface DeltaConfig {
  delta: number;
  rings: number;
  bandwidth: number | null;
}

export const DEFAULT_DELTA: DeltaConfig = { delta: 1, rings: 1, bandwidth: null };

interface BaseResult {
  surrogate: LinearSurrogate | TanhSurrogate;
  nQueries: number;
  attribution: number[];
}

interface BaseExplainer<R extends BaseResult> {
  scale: number[];
  explain(f: BlackBox, x: number[], rng: Rng): R;
}

export interface EnsembleResult<R extends BaseResult> {
  surrogate: BlendedSurrogate;
  members: R[];
  anchors: number[][];
  nQueries: number;
  /** the explanation at x0 is the x0 anchor's own */
  attribution: number[];
}

/** x0 plus x0 +/- k*delta*scale_j*e_j for every feature j and k = 1..rings (axes layout). */
export function anchorPoints(x0: number[], scale: number[], cfg: DeltaConfig, bounds: Bounds = null): number[][] {
  const d = x0.length;
  const dirs: number[][] = [];
  for (let j = 0; j < d; j++) {
    for (const s of [1, -1]) {
      const e = new Array<number>(d).fill(0);
      e[j] = s;
      dirs.push(e);
    }
  }
  const anchors = [x0.slice()];
  const seen = new Set([x0.join(",")]);
  for (let k = 1; k <= cfg.rings; k++) {
    for (const e of dirs) {
      let a = x0.map((xi, j) => xi + k * cfg.delta * e[j] * scale[j]);
      if (bounds !== null) a = a.map((v, j) => Math.min(Math.max(v, bounds[j][0]), bounds[j][1]));
      const key = a.join(",");
      if (!seen.has(key)) {
        seen.add(key);
        anchors.push(a);
      }
    }
  }
  return anchors;
}

export class DeltaAnchorEnsemble<R extends BaseResult> {
  base: BaseExplainer<R>;
  config: DeltaConfig;
  bounds: Bounds;

  constructor(base: BaseExplainer<R>, config: Partial<DeltaConfig> = {}, bounds: Bounds = null) {
    this.base = base;
    this.config = { ...DEFAULT_DELTA, ...config };
    this.bounds = bounds;
  }

  explain(f: BlackBox, x0: number[], rng: Rng): EnsembleResult<R> {
    const cfg = this.config;
    const scale = this.base.scale;
    const anchors = anchorPoints(x0, scale, cfg, this.bounds);
    const members: R[] = [];
    let nQueries = 0;
    anchors.forEach((a, i) => {
      const res = this.base.explain(f, a, i === 0 ? rng : rng.spawn(i));
      members.push(res);
      nQueries += res.nQueries;
    });
    const bandwidth = cfg.bandwidth ?? 0.5 * cfg.delta;
    const surrogate = new BlendedSurrogate(x0, scale, anchors, members.map((m) => m.surrogate), bandwidth);
    return { surrogate, members, anchors, nQueries, attribution: members[0].attribution };
  }
}
