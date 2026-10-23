/** PCA of standardised rows via the cyclic Jacobi eigen-decomposition of their covariance (no dependencies). */

export function pca(Z: number[][]): { vectors: number[][]; explained: number[] } {
  const d = Z[0].length;
  const n = Z.length;
  const mu = Array.from({ length: d }, (_, j) => Z.reduce((s, r) => s + r[j], 0) / n);
  const A = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const r of Z)
    for (let i = 0; i < d; i++) for (let j = i; j < d; j++) A[i][j] += ((r[i] - mu[i]) * (r[j] - mu[j])) / n;
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) A[i][j] = A[j][i];
  const V: number[][] = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += A[p][q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < d; p++)
      for (let q = p + 1; q < d; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < d; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
  }
  const order = A.map((_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  const total = order.reduce((s, i) => s + Math.max(A[i][i], 0), 0) || 1;
  return {
    vectors: order.map((i) => V.map((row) => row[i])),
    explained: order.map((i) => Math.max(A[i][i], 0) / total),
  };
}
