/** Port of atlime/surrogates.py. All surrogates work in standardised offsets z = (x - anchor) / scale. */
import { fsum, sumprod } from "./numeric.ts";

export function standardize(x: number[], anchor: number[], scale: number[]): number[] {
  return x.map((xi, j) => (xi - anchor[j]) / scale[j]);
}

export interface Surrogate {
  kind: "linear" | "tanh" | "blend";
  predict(x: number[]): number;
  gradient(x: number[]): number[];
}

/** g(x) = intercept + coef . z  (classic LIME). */
export class LinearSurrogate implements Surrogate {
  readonly kind = "linear" as const;
  anchor: number[];
  scale: number[];
  intercept: number;
  coef: number[];

  constructor(anchor: number[], scale: number[], intercept: number, coef: number[]) {
    this.anchor = anchor.slice();
    this.scale = scale.slice();
    this.intercept = intercept;
    this.coef = coef.slice();
  }

  predict(x: number[]): number {
    return this.intercept + sumprod(this.coef, standardize(x, this.anchor, this.scale));
  }

  gradient(): number[] {
    return this.coef.slice();
  }
}

/** g(x) = c + a tanh(beta0 + beta . z);  link space: artanh((g - c)/a) = beta0 + beta . z. */
export class TanhSurrogate implements Surrogate {
  readonly kind = "tanh" as const;
  anchor: number[];
  scale: number[];
  lo: number;
  hi: number;
  beta0: number;
  beta: number[];

  constructor(anchor: number[], scale: number[], lo: number, hi: number, beta0: number, beta: number[]) {
    if (!(hi > lo)) throw new Error("TanhSurrogate needs hi > lo");
    this.anchor = anchor.slice();
    this.scale = scale.slice();
    this.lo = lo;
    this.hi = hi;
    this.beta0 = beta0;
    this.beta = beta.slice();
  }

  get center(): number {
    return 0.5 * (this.hi + this.lo);
  }

  get halfRange(): number {
    return 0.5 * (this.hi - this.lo);
  }

  eta(x: number[]): number {
    return this.beta0 + sumprod(this.beta, standardize(x, this.anchor, this.scale));
  }

  predict(x: number[]): number {
    return this.center + this.halfRange * Math.tanh(this.eta(x));
  }

  gradient(x: number[]): number[] {
    const t = Math.tanh(this.eta(x));
    const s = this.halfRange * (1 - t * t);
    return this.beta.map((b) => s * b);
  }
}

/** Gaussian partition-of-unity blend of anchor surrogates (the +/- delta ensemble). */
export class BlendedSurrogate implements Surrogate {
  readonly kind = "blend" as const;
  center: number[];
  scale: number[];
  anchors: number[][];
  members: (LinearSurrogate | TanhSurrogate)[];
  bandwidth: number;
  private anchorZ: number[][];

  constructor(
    center: number[],
    scale: number[],
    anchors: number[][],
    members: (LinearSurrogate | TanhSurrogate)[],
    bandwidth: number,
  ) {
    if (anchors.length !== members.length || members.length === 0) {
      throw new Error("BlendedSurrogate needs one member per anchor");
    }
    this.center = center.slice();
    this.scale = scale.slice();
    this.anchors = anchors.map((a) => a.slice());
    this.members = members.slice();
    this.bandwidth = bandwidth;
    this.anchorZ = this.anchors.map((a) => standardize(a, this.center, this.scale));
  }

  weights(x: number[]): number[] {
    const z = standardize(x, this.center, this.scale);
    const d2 = this.anchorZ.map((za) => fsum(z.map((zi, j) => (zi - za[j]) ** 2)));
    const m = Math.min(...d2);
    const inv = 1 / (2 * this.bandwidth * this.bandwidth);
    const v = d2.map((e) => Math.exp(-(e - m) * inv));
    const total = fsum(v);
    return v.map((vi) => vi / total);
  }

  predict(x: number[]): number {
    const p = this.weights(x);
    const terms: number[] = [];
    for (let i = 0; i < p.length; i++) if (p[i] > 0) terms.push(p[i] * this.members[i].predict(x));
    return fsum(terms);
  }

  gradient(x: number[]): number[] {
    const p = this.weights(x);
    const d = this.center.length;
    const zbar = Array.from({ length: d }, (_, j) => fsum(p.map((pi, i) => pi * this.anchorZ[i][j])));
    const out = new Array<number>(d).fill(0);
    const invH2 = 1 / (this.bandwidth * this.bandwidth);
    for (let i = 0; i < p.length; i++) {
      const pi = p[i];
      if (pi === 0) continue;
      const g = this.members[i];
      const gx = g.predict(x);
      const grad = g.gradient(x);
      const za = this.anchorZ[i];
      for (let j = 0; j < d; j++) out[j] += pi * grad[j] + invH2 * pi * gx * (za[j] - zbar[j]);
    }
    return out;
  }
}
