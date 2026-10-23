"""Minimal dense linear algebra in pure Python (no NumPy).

Only what the explainers need: a Cholesky solver for symmetric positive
definite systems, weighted ridge regression with an unpenalised intercept
(the estimator LIME uses: scikit-learn's ``Ridge(alpha, fit_intercept=True)``
fitted with ``sample_weight``) and the weighted coefficient of determination.

Data matrices are stored row-major as ``list[list[float]]`` (one row per
sample). Internally the regressions work column-wise so that every inner
product runs through ``math.sumprod`` (C speed, extended precision).
"""

from __future__ import annotations

import math
from typing import Sequence

Vector = list[float]
Matrix = list[list[float]]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sumprod(a, b)


def norm(a: Sequence[float]) -> float:
    return math.sqrt(math.sumprod(a, a))


def cholesky(a: Matrix) -> Matrix | None:
    """Lower-triangular L with L L^T = A, or None if A is not positive definite."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = low[i]
        for j in range(i + 1):
            row_j = low[j]
            s = a[i][j] - math.sumprod(row_i[:j], row_j[:j])
            if i == j:
                if s <= 0.0 or not math.isfinite(s):
                    return None
                row_i[i] = math.sqrt(s)
            else:
                row_i[j] = s / row_j[j]
    return low


def solve_spd(a: Matrix, b: Sequence[float]) -> Vector:
    """Solve A x = b for symmetric positive (semi-)definite A.

    If the Cholesky factorisation fails (singular / numerically indefinite
    matrix) a growing multiple of the mean diagonal is added until it
    succeeds, i.e. the solve degrades gracefully into a ridge solve.
    """
    n = len(a)
    if n == 0:
        return []
    low = cholesky(a)
    if low is None:
        scale = sum(abs(a[i][i]) for i in range(n)) / n or 1.0
        bump = 1e-12 * scale
        while low is None:
            if bump > 1e6 * scale:
                raise ValueError("solve_spd: matrix could not be regularised")
            shifted = [row[:] for row in a]
            for i in range(n):
                shifted[i][i] += bump
            low = cholesky(shifted)
            bump *= 10.0
    # forward substitution L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - math.sumprod(low[i][:i], y[:i])) / low[i][i]
    # back substitution L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= low[k][i] * x[k]
        x[i] = s / low[i][i]
    return x


def columns(rows: Sequence[Sequence[float]], d: int) -> Matrix:
    """Transpose a row-major matrix with d columns."""
    if not rows:
        return [[] for _ in range(d)]
    return [list(c) for c in zip(*rows)]


def weighted_ridge(
    rows: Sequence[Sequence[float]],
    y: Sequence[float],
    w: Sequence[float],
    alpha: float = 1.0,
    fit_intercept: bool = True,
) -> tuple[Vector, float]:
    """Minimise  sum_i w_i (y_i - b - x_i . beta)^2 + alpha * ||beta||^2.

    The intercept b is not penalised. Returns ``(beta, b)``. Identical to
    scikit-learn's ``Ridge(alpha=alpha).fit(X, y, sample_weight=w)``.
    """
    n = len(y)
    d = len(rows[0]) if n else 0
    sw = math.fsum(w)
    if sw <= 0.0:
        raise ValueError("weighted_ridge: weights must have a positive sum")
    cols = columns(rows, d)
    if fit_intercept:
        x_mean = [math.sumprod(w, c) / sw for c in cols]
        y_mean = math.sumprod(w, y) / sw
        cols = [[v - m for v in c] for c, m in zip(cols, x_mean)]
        yc = [v - y_mean for v in y]
    else:
        x_mean = [0.0] * d
        y_mean = 0.0
        yc = list(y)
    if d == 0:
        return [], y_mean
    wcols = [[wi * v for wi, v in zip(w, c)] for c in cols]
    gram = [[0.0] * d for _ in range(d)]
    for j in range(d):
        for k in range(j, d):
            gram[j][k] = gram[k][j] = math.sumprod(wcols[j], cols[k])
        gram[j][j] += alpha
    rhs = [math.sumprod(wc, yc) for wc in wcols]
    beta = solve_spd(gram, rhs)
    return beta, y_mean - math.sumprod(beta, x_mean)


def weighted_r2(y: Sequence[float], y_hat: Sequence[float], w: Sequence[float]) -> float:
    """Weighted coefficient of determination (scikit-learn's ``r2_score``)."""
    sw = math.fsum(w)
    y_mean = math.sumprod(w, y) / sw
    ss_res = math.fsum(wi * (a - b) ** 2 for wi, a, b in zip(w, y, y_hat))
    ss_tot = math.fsum(wi * (a - y_mean) ** 2 for wi, a in zip(w, y))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot
