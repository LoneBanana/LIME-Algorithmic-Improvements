/** Port of atlime/walk.py: the epsilon-walk to the nearest local maximum / minimum. */
import { sumprod } from "./numeric.ts";
import type { BlackBox } from "./lime.ts";

export type Bounds = [number, number][] | null;
export type StopReason = "extremum" | "flat" | "bounds" | "budget";

export interface WalkResult {
  direction: 1 | -1;
  points: number[][];
  values: number[];
  stopReason: StopReason;
  nQueries: number;
}

function clip(x: number[], bounds: Bounds): number[] {
  if (bounds === null) return x;
  return x.map((v, j) => Math.min(Math.max(v, bounds[j][0]), bounds[j][1]));
}

/** Central differences in standardised units (one-sided at a bound). */
export function finiteDifferenceGradient(
  f: BlackBox,
  x: number[],
  scale: number[],
  h: number,
  bounds: Bounds = null,
): { grad: number[]; queries: number } {
  const grad: number[] = [];
  let queries = 0;
  for (let j = 0; j < x.length; j++) {
    const sj = scale[j];
    let up = x[j] + h * sj;
    let dn = x[j] - h * sj;
    if (bounds !== null) {
      up = Math.min(up, bounds[j][1]);
      dn = Math.max(dn, bounds[j][0]);
    }
    const width = (up - dn) / sj;
    if (width <= 0) {
      grad.push(0);
      continue;
    }
    const xp = x.slice();
    const xm = x.slice();
    xp[j] = up;
    xm[j] = dn;
    grad.push((f(xp) - f(xm)) / width);
    queries += 2;
  }
  return { grad, queries };
}

/**
 * Fixed-length steps of size epsilon (standardised units) along the steepest
 * ascent (direction = +1) or descent (-1). A step is accepted only if it moves
 * f by more than tolRel * |f(x_k) - f(x0)| in the walking direction. grad0 is an
 * optional precomputed gradient at x0 (shared by the up and down walks; its
 * queries are not counted here).
 */
export function epsilonWalk(
  f: BlackBox,
  x0: number[],
  f0: number,
  scale: number[],
  direction: 1 | -1,
  epsilon: number,
  maxSteps: number,
  gradStep: number,
  tolRel: number,
  bounds: Bounds = null,
  grad0: number[] | null = null,
): WalkResult {
  let x = x0.slice();
  let fx = f0;
  const points = [x.slice()];
  const values = [fx];
  let queries = 0;
  let stopReason: StopReason = "budget";
  for (let k = 0; k < maxSteps; k++) {
    let grad: number[];
    if (k === 0 && grad0 !== null) {
      grad = grad0.slice();
    } else {
      const fd = finiteDifferenceGradient(f, x, scale, gradStep, bounds);
      grad = fd.grad;
      queries += fd.queries;
    }
    const gnorm = Math.sqrt(sumprod(grad, grad));
    if (gnorm <= 1e-12) {
      stopReason = "flat";
      break;
    }
    const step = (epsilon * direction) / gnorm;
    const xNew = clip(
      x.map((xi, j) => xi + step * grad[j] * scale[j]),
      bounds,
    );
    if (xNew.every((v, j) => v === x[j])) {
      stopReason = "bounds";
      break;
    }
    const fNew = f(xNew);
    queries += 1;
    if (direction * (fNew - fx) <= tolRel * Math.abs(fx - f0)) {
      stopReason = "extremum";
      break;
    }
    x = xNew;
    fx = fNew;
    points.push(x.slice());
    values.push(fx);
  }
  return { direction, points, values, stopReason, nQueries: queries };
}
