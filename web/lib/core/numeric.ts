/**
 * Accurate summation primitives mirroring the Python reference implementation:
 *   fsum    <-> math.fsum     (Shewchuk's exact partials + CPython's final rounding)
 *   sumprod <-> math.sumprod  (dot product accumulated in doubled precision)
 * Using the same numerics in both languages is what lets the parity test
 * (scripts/parity.ts) hold the port to ~1e-9 relative agreement.
 */

const SPLITTER = 134217729; // 2^27 + 1, Veltkamp splitting constant

/** Dot product in doubled precision (Ogita-Rump-Oishi "Dot2" with Dekker's TwoProduct). */
export function sumprod(a: ArrayLike<number>, b: ArrayLike<number>, n = Math.min(a.length, b.length)): number {
  let p = 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    const h = x * y;
    let c = SPLITTER * x;
    const xh = c - (c - x);
    const xl = x - xh;
    c = SPLITTER * y;
    const yh = c - (c - y);
    const yl = y - yh;
    const r = xl * yl - (((h - xh * yh) - xl * yh) - xh * yl);
    const t = p + h;
    const z = t - p;
    const q = (p - (t - z)) + (h - z);
    p = t;
    s += q + r;
  }
  return p + s;
}

/** Correctly rounded sum (Python's math.fsum). */
export function fsum(values: Iterable<number>): number {
  const partials: number[] = [];
  for (const value of values) {
    let x = value;
    let i = 0;
    for (let j = 0; j < partials.length; j++) {
      let y = partials[j];
      if (Math.abs(x) < Math.abs(y)) {
        const t = x;
        x = y;
        y = t;
      }
      const hi = x + y;
      const lo = y - (hi - x);
      if (lo !== 0) partials[i++] = lo;
      x = hi;
    }
    partials.length = i;
    partials.push(x);
  }
  let n = partials.length;
  let hi = 0;
  if (n > 0) {
    hi = partials[--n];
    let lo = 0;
    while (n > 0) {
      const x = hi;
      const y = partials[--n];
      hi = x + y;
      const yr = hi - x;
      lo = y - yr;
      if (lo !== 0) break;
    }
    // half-even rounding across partials, as in CPython
    if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
      const y = lo * 2;
      const x = hi + y;
      const yr = x - hi;
      if (y === yr) hi = x;
    }
  }
  return hi;
}

export function sigmoid(v: number): number {
  if (v >= 0) return 1 / (1 + Math.exp(-v));
  const e = Math.exp(v);
  return e / (1 + e);
}
