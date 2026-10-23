"""The epsilon-walk: climb (or descend) the black box's response surface.

Starting from the explained instance, take fixed-length steps of width
``epsilon`` (measured in standard deviations of each feature) along the
direction of steepest ascent (descent) until the next step stops improving
the output (see ``epsilon_walk`` for the exact rule). The last point reached is the nearest local
maximum (minimum) along the steepest-ascent curve through the instance; its
output value becomes an asymptote of the tanh^-1 regressor.

The black box is only *queried*: the ascent direction is a central
finite-difference gradient, which keeps the whole procedure model-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

BlackBox = Callable[[Sequence[float]], float]
Bounds = Sequence[tuple[float, float]] | None


@dataclass
class WalkResult:
    direction: int                 # +1 ascent (towards a local max), -1 descent (towards a local min)
    points: list[list[float]]      # points[0] is the start, points[-1] the extremum estimate
    values: list[float]
    stop_reason: str               # 'extremum' | 'flat' | 'bounds' | 'budget'
    n_queries: int

    @property
    def end_point(self) -> list[float]:
        return self.points[-1]

    @property
    def end_value(self) -> float:
        return self.values[-1]

    @property
    def steps(self) -> int:
        return len(self.points) - 1

    def to_dict(self) -> dict:
        return {"direction": self.direction, "points": self.points, "values": self.values,
                "stop_reason": self.stop_reason, "n_queries": self.n_queries}


def _clip(x: list[float], bounds: Bounds) -> list[float]:
    if bounds is None:
        return x
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(x, bounds)]


def finite_difference_gradient(f: BlackBox, x: Sequence[float], scale: Sequence[float], h: float,
                               bounds: Bounds = None) -> tuple[list[float], int]:
    """Central differences in standardised units (one-sided at a bound).

    Returns (gradient, number of black-box queries).
    """
    grad = []
    queries = 0
    for j, (xj, sj) in enumerate(zip(x, scale)):
        up = xj + h * sj
        dn = xj - h * sj
        if bounds is not None:
            lo, hi = bounds[j]
            up = min(up, hi)
            dn = max(dn, lo)
        width = (up - dn) / sj
        if width <= 0.0:
            grad.append(0.0)
            continue
        xp = list(x)
        xm = list(x)
        xp[j] = up
        xm[j] = dn
        grad.append((f(xp) - f(xm)) / width)
        queries += 2
    return grad, queries


def epsilon_walk(f: BlackBox, x0: Sequence[float], f0: float, scale: Sequence[float], direction: int,
                 epsilon: float, max_steps: int, grad_step: float, tol_rel: float,
                 bounds: Bounds = None, grad0: Sequence[float] | None = None) -> WalkResult:
    """Walk from x0 with steps of length ``epsilon`` (standardised units).

    direction=+1 climbs to the nearest local maximum, -1 descends to the
    nearest local minimum. A step is accepted only if it moves f in the
    walking direction by more than ``tol_rel * |f(x_k) - f(x0)|``, i.e. by
    more than a small fraction of the climb achieved so far. The first step
    therefore only needs a strict improvement (a start deep inside a
    saturated region still walks), while a walk that has climbed a sigmoid
    stops once its tail has flattened out. Stops when
      * a step is not accepted -> 'extremum' (we stepped onto/over the peak,
        or the curve has levelled off)
      * the gradient vanishes  -> 'flat'
      * a bound blocks every move -> 'bounds'
      * ``max_steps`` steps were taken -> 'budget'
    The extremum estimate is the last *accepted* point. ``grad0`` is an
    optional, already computed gradient at x0 (the up and down walks share
    it, saving 2d queries); its queries are not counted here.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    x = list(x0)
    fx = float(f0)
    points = [list(x)]
    values = [fx]
    queries = 0
    reason = "budget"
    for k in range(max_steps):
        if k == 0 and grad0 is not None:
            grad = list(grad0)
        else:
            grad, q = finite_difference_gradient(f, x, scale, grad_step, bounds)
            queries += q
        gnorm = math.sqrt(math.sumprod(grad, grad))
        if gnorm <= 1e-12:
            reason = "flat"
            break
        step = epsilon * direction / gnorm
        x_new = _clip([xi + step * gi * si for xi, gi, si in zip(x, grad, scale)], bounds)
        if all(a == b for a, b in zip(x_new, x)):
            reason = "bounds"
            break
        f_new = float(f(x_new))
        queries += 1
        if direction * (f_new - fx) <= tol_rel * abs(fx - f0):
            reason = "extremum"
            break
        x, fx = x_new, f_new
        points.append(list(x))
        values.append(fx)
    return WalkResult(direction, points, values, reason, queries)
