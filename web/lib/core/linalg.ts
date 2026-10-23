/** Port of atlime/linalg.py: Cholesky solver and weighted ridge regression. */
import { fsum, sumprod } from "./numeric.ts";

export type Matrix = number[][];

export function cholesky(a: Matrix): Matrix | null {
  const n = a.length;
  const low: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const rowI = low[i];
    for (let j = 0; j <= i; j++) {
      const rowJ = low[j];
      const s = a[i][j] - sumprod(rowI, rowJ, j);
      if (i === j) {
        if (s <= 0 || !Number.isFinite(s)) return null;
        rowI[i] = Math.sqrt(s);
      } else {
        rowI[j] = s / rowJ[j];
      }
    }
  }
  return low;
}

/** Solve A x = b for symmetric positive (semi-)definite A, regularising if needed. */
export function solveSpd(a: Matrix, b: number[]): number[] {
  const n = a.length;
  if (n === 0) return [];
  let low = cholesky(a);
  if (low === null) {
    let diag = 0;
    for (let i = 0; i < n; i++) diag += Math.abs(a[i][i]);
    const scale = diag / n || 1;
    let bump = 1e-12 * scale;
    while (low === null) {
      if (bump > 1e6 * scale) throw new Error("solveSpd: matrix could not be regularised");
      const shifted = a.map((row) => row.slice());
      for (let i = 0; i < n; i++) shifted[i][i] += bump;
      low = cholesky(shifted);
      bump *= 10;
    }
  }
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) y[i] = (b[i] - sumprod(low[i], y, i)) / low[i][i];
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= low[k][i] * x[k];
    x[i] = s / low[i][i];
  }
  return x;
}

function columns(rows: number[][], d: number): Matrix {
  const cols: Matrix = Array.from({ length: d }, () => new Array<number>(rows.length));
  for (let i = 0; i < rows.length; i++) for (let j = 0; j < d; j++) cols[j][i] = rows[i][j];
  return cols;
}

/** min  sum_i w_i (y_i - b - x_i.beta)^2 + alpha ||beta||^2  (intercept unpenalised). */
export function weightedRidge(
  rows: number[][],
  y: number[],
  w: number[],
  alpha = 1,
  fitIntercept = true,
): { beta: number[]; intercept: number } {
  const n = y.length;
  const d = n ? rows[0].length : 0;
  const sw = fsum(w);
  if (sw <= 0) throw new Error("weightedRidge: weights must have a positive sum");
  let cols = columns(rows, d);
  let xMean: number[];
  let yMean: number;
  let yc: number[];
  if (fitIntercept) {
    xMean = cols.map((c) => sumprod(w, c) / sw);
    yMean = sumprod(w, y) / sw;
    cols = cols.map((c, j) => c.map((v) => v - xMean[j]));
    yc = y.map((v) => v - yMean);
  } else {
    xMean = new Array<number>(d).fill(0);
    yMean = 0;
    yc = y.slice();
  }
  if (d === 0) return { beta: [], intercept: yMean };
  const wcols = cols.map((c) => c.map((v, i) => w[i] * v));
  const gram: Matrix = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let j = 0; j < d; j++) {
    for (let k = j; k < d; k++) {
      const v = sumprod(wcols[j], cols[k]);
      gram[j][k] = v;
      gram[k][j] = v;
    }
    gram[j][j] += alpha;
  }
  const rhs = wcols.map((wc) => sumprod(wc, yc));
  const beta = solveSpd(gram, rhs);
  return { beta, intercept: yMean - sumprod(beta, xMean) };
}

export function weightedR2(y: number[], yHat: number[], w: number[]): number {
  const sw = fsum(w);
  const yMean = sumprod(w, y) / sw;
  const ssRes = fsum(y.map((a, i) => w[i] * (a - yHat[i]) ** 2));
  const ssTot = fsum(y.map((a, i) => w[i] * (a - yMean) ** 2));
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}
