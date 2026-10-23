"""Evaluation metrics and paired statistics (pure Python).

Fidelity is measured on black-box queries drawn independently of the
explainers (in 1-D a few coincide exactly with walk points x0 +/- k*epsilon or
with ensemble anchors; checked to move sphere MSE by at most ~1.4%):

* ``sphere``       - points at exact standardised distance r from x0
                     (uniform random directions), for r on a grid. This is
                     the "how far does the explanation stay faithful" curve.
* ``neighborhood`` - a held-out LIME-distributed sample with LIME kernel
                     weights (the locality LIME itself optimises).
* ``global``       - the held-out test set of the dataset.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .lime import lime_kernel
from .rng import Rng

BlackBox = Callable[[Sequence[float]], float]


# -- evaluation sets -----------------------------------------------------------

def unit_directions(d: int, m: int, rng: Rng) -> list[list[float]]:
    if d == 1:  # exactly balanced in 1-D
        return [[1.0 if i % 2 == 0 else -1.0] for i in range(m)]
    out = []
    for _ in range(m):
        v = [rng.normal() for _ in range(d)]
        n = math.sqrt(math.sumprod(v, v))
        out.append([c / n for c in v])
    return out


def clip_point(x: Sequence[float], bounds) -> list[float]:
    if bounds is None:
        return list(x)
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(x, bounds)]


def sphere_set(f: BlackBox, x0: Sequence[float], scale: Sequence[float], radii: Sequence[float], m: int,
               rng: Rng, bounds=None) -> dict:
    """Points at standardised radius r around x0 (clipped to ``bounds`` if given)."""
    d = len(x0)
    out = {"radii": list(radii), "points": [], "values": []}
    for r in radii:
        dirs = unit_directions(d, m, rng)
        pts = [clip_point([xi + r * ui * si for xi, ui, si in zip(x0, u, scale)], bounds) for u in dirs]
        out["points"].append(pts)
        out["values"].append([f(p) for p in pts])
    return out


def neighborhood_set(f: BlackBox, x0: Sequence[float], scale: Sequence[float], m: int, kernel_width: float,
                     rng: Rng) -> dict:
    d = len(x0)
    pts, w = [], []
    for _ in range(m):
        z = [rng.normal() for _ in range(d)]
        pts.append([xi + zi * si for xi, zi, si in zip(x0, z, scale)])
        w.append(lime_kernel(math.sqrt(math.sumprod(z, z)), kernel_width))
    return {"points": pts, "values": [f(p) for p in pts], "weights": w}


# -- errors ----------------------------------------------------------------------

def mse(surrogate, points: Sequence[Sequence[float]], values: Sequence[float]) -> float:
    return math.fsum((surrogate.predict(p) - v) ** 2 for p, v in zip(points, values)) / len(values)


def weighted_mse(surrogate, points, values, weights) -> float:
    return (math.fsum(w * (surrogate.predict(p) - v) ** 2 for p, v, w in zip(points, values, weights))
            / math.fsum(weights))


def sphere_mse(surrogate, sphere: dict) -> list[float]:
    return [mse(surrogate, pts, vals) for pts, vals in zip(sphere["points"], sphere["values"])]


def true_gradient(f: BlackBox, x: Sequence[float], scale: Sequence[float], h: float = 1e-4) -> list[float]:
    """Central-difference gradient in standardised units (for smooth black boxes)."""
    g = []
    for j in range(len(x)):
        xp, xm = list(x), list(x)
        xp[j] += h * scale[j]
        xm[j] -= h * scale[j]
        g.append((f(xp) - f(xm)) / (2.0 * h))
    return g


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(math.sumprod(a, a))
    nb = math.sqrt(math.sumprod(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.sumprod(a, b) / (na * nb)


# -- descriptive statistics -------------------------------------------------------

def mean(v: Sequence[float]) -> float:
    return math.fsum(v) / len(v)


def median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(v: Sequence[float], q: float) -> float:
    s = sorted(v)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def std(v: Sequence[float]) -> float:
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


# -- paired comparisons -------------------------------------------------------------

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test of a - b.

    Normal approximation; zero differences are dropped (Wilcoxon's zero
    method) and tied |differences| get average ranks with the usual variance
    correction.
    """
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "W_plus": 0.0, "z": 0.0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        r = 0.5 * (i + j) + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_plus = math.fsum(r for r, dv in zip(ranks, diffs) if dv > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return {"n": n, "W_plus": w_plus, "z": 0.0, "p": 1.0}
    z = (w_plus - mu) / math.sqrt(var)
    return {"n": n, "W_plus": w_plus, "z": z, "p": 2.0 * (1.0 - normal_cdf(abs(z)))}


def bootstrap_mean_diff_ci(a: Sequence[float], b: Sequence[float], rng: Rng, n_boot: int = 2000,
                           level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(a - b) over paired instances."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    stats = []
    for _ in range(n_boot):
        stats.append(math.fsum(diffs[rng.randint(n)] for _ in range(n)) / n)
    alpha = (1.0 - level) / 2.0
    return quantile(stats, alpha), quantile(stats, 1.0 - alpha)


def win_rate(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of instances where a < b (ties count one half)."""
    score = 0.0
    for x, y in zip(a, b):
        score += 1.0 if x < y else 0.5 if x == y else 0.0
    return score / len(a)
