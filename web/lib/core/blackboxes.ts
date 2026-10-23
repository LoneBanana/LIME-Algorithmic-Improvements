/** The black boxes used by the demos: the 1-D curves of atlime/datasets.py and MLP inference. */
import { sigmoid, sumprod } from "./numeric.ts";

export const ONE_D_FUNCTIONS = {
  logistic: (x: number) => sigmoid(2 * x),
  sine: (x: number) => Math.sin(x),
  bumps: (x: number) =>
    1.0 * Math.exp(-((x + 3) ** 2) / 1.0) +
    0.6 * Math.exp(-((x - 0.5) ** 2) / 2.0) +
    0.9 * Math.exp(-((x - 3.5) ** 2) / 0.6),
  staircase: (x: number) => (sigmoid(3 * (x + 3)) + sigmoid(3 * x) + sigmoid(3 * (x - 3))) / 3,
  ripple: (x: number) => 0.15 * x + 0.25 * Math.sin(2 * x),
} as const;

export type OneDName = keyof typeof ONE_D_FUNCTIONS;
export const ONE_D_DOMAIN: [number, number] = [-6, 6];

export interface MlpWeights {
  type: "mlp";
  sizes: number[];
  output: "sigmoid" | "linear";
  W: number[][][];
  b: number[][];
  x_mean: number[];
  x_std: number[];
  y_mean: number;
  y_std: number;
}

/** Forward pass identical to atlime.models.MLP.predict. */
export function mlpPredictor(m: MlpWeights): (x: number[]) => number {
  const last = m.W.length - 1;
  return (x: number[]) => {
    let a = x.map((v, j) => (v - m.x_mean[j]) / m.x_std[j]);
    for (let layer = 0; layer <= last; layer++) {
      const W = m.W[layer];
      const b = m.b[layer];
      const z = W.map((row, i) => b[i] + sumprod(row, a));
      a = layer === last ? z : z.map((v) => Math.tanh(v));
    }
    const o = a[0];
    return m.output === "sigmoid" ? sigmoid(o) : m.y_mean + m.y_std * o;
  };
}
