/**
 * In-browser training of the black box for uploaded data: a one-hidden-layer tanh MLP with
 * a sigmoid output, cross-entropy and Adam — the same model and optimiser as
 * atlime.models.MLP, producing weights in the same format (so mlpPredictor can run it).
 */
import { Rng, sigmoid } from "@/lib/core/index.ts";
import type { MlpWeights } from "@/lib/core/index.ts";

export function trainMlp(
  X: number[][],
  y: number[],
  opts: { hidden: number; epochs: number; lr: number; seed: number; batch?: number; onProgress?: (epoch: number) => void },
): MlpWeights {
  const n = X.length;
  const d = X[0].length;
  const H = opts.hidden;
  const batch = opts.batch ?? 32;
  const xMean = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const xStd = xMean.map((m, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / n) || 1);
  const Xs = X.map((r) => r.map((v, j) => (v - xMean[j]) / xStd[j]));
  const rng = new Rng(opts.seed);
  const glorot = (fanIn: number, fanOut: number) => {
    const lim = Math.sqrt(6 / (fanIn + fanOut));
    return Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => (2 * rng.uniform() - 1) * lim));
  };
  const W1 = glorot(d, H);
  const b1 = new Array<number>(H).fill(0);
  const W2 = glorot(H, 1);
  const b2 = [0];
  const params = [W1.flat(), b1, W2.flat(), b2];
  const m = params.map((p) => new Array<number>(p.length).fill(0));
  const v = params.map((p) => new Array<number>(p.length).fill(0));
  const order = X.map((_, i) => i);
  let step = 0;
  const h = new Array<number>(H);
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const k = rng.randint(i + 1);
      [order[i], order[k]] = [order[k], order[i]];
    }
    for (let start = 0; start < n; start += batch) {
      const g = params.map((p) => new Array<number>(p.length).fill(0));
      const idx = order.slice(start, start + batch);
      for (const r of idx) {
        const x = Xs[r];
        for (let a = 0; a < H; a++) {
          let s = b1[a];
          for (let j = 0; j < d; j++) s += W1[a][j] * x[j];
          h[a] = Math.tanh(s);
        }
        let o = b2[0];
        for (let a = 0; a < H; a++) o += W2[0][a] * h[a];
        const delta = sigmoid(o) - y[r];
        for (let a = 0; a < H; a++) {
          g[2][a] += delta * h[a];
          const da = delta * W2[0][a] * (1 - h[a] * h[a]);
          g[1][a] += da;
          for (let j = 0; j < d; j++) g[0][a * d + j] += da * x[j];
        }
        g[3][0] += delta;
      }
      step++;
      const c1 = 1 - 0.9 ** step;
      const c2 = 1 - 0.999 ** step;
      params.forEach((p, k) => {
        for (let i = 0; i < p.length; i++) {
          const gi = g[k][i] / idx.length;
          m[k][i] = 0.9 * m[k][i] + 0.1 * gi;
          v[k][i] = 0.999 * v[k][i] + 0.001 * gi * gi;
          p[i] -= (opts.lr * (m[k][i] / c1)) / (Math.sqrt(v[k][i] / c2) + 1e-8);
        }
      });
      for (let a = 0; a < H; a++) {
        for (let j = 0; j < d; j++) W1[a][j] = params[0][a * d + j];
        b1[a] = params[1][a];
        W2[0][a] = params[2][a];
      }
      b2[0] = params[3][0];
    }
    opts.onProgress?.(epoch + 1);
  }
  return { type: "mlp", sizes: [d, H, 1], output: "sigmoid", W: [W1, W2], b: [b1, b2], x_mean: xMean, x_std: xStd,
    y_mean: 0, y_std: 1 };
}
