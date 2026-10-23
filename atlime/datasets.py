"""Synthetic datasets and analytic black boxes (all generated from scratch).

Every function here is deterministic given an ``Rng``. No real-world data is
used anywhere in this project.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .models import sigmoid
from .rng import Rng

# ---------------------------------------------------------------------------
# 1-D response curves (used for the illustrations and experiment E1).
# The input is treated as already standardised (feature scale = 1).
# ---------------------------------------------------------------------------


def _logistic(x: float) -> float:
    return sigmoid(2.0 * x)


def _sine(x: float) -> float:
    return math.sin(x)


def _bumps(x: float) -> float:
    return (1.0 * math.exp(-((x + 3.0) ** 2) / 1.0)
            + 0.6 * math.exp(-((x - 0.5) ** 2) / 2.0)
            + 0.9 * math.exp(-((x - 3.5) ** 2) / 0.6))


def _staircase(x: float) -> float:
    return (sigmoid(3.0 * (x + 3.0)) + sigmoid(3.0 * x) + sigmoid(3.0 * (x - 3.0))) / 3.0


def _ripple(x: float) -> float:
    return 0.15 * x + 0.25 * math.sin(2.0 * x)


ONE_D_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "logistic": _logistic,
    "sine": _sine,
    "bumps": _bumps,
    "staircase": _staircase,
    "ripple": _ripple,
}
ONE_D_DOMAIN = (-6.0, 6.0)


def one_d_black_box(name: str) -> Callable[[Sequence[float]], float]:
    g = ONE_D_FUNCTIONS[name]
    return lambda x: g(x[0])


# ---------------------------------------------------------------------------
# Two moons (2-D classification)
# ---------------------------------------------------------------------------


def make_moons(n: int, noise: float, rng: Rng) -> tuple[list[list[float]], list[float]]:
    """Same construction as scikit-learn's ``make_moons`` (re-implemented)."""
    n_out = n // 2
    n_in = n - n_out
    X: list[list[float]] = []
    y: list[float] = []
    for i in range(n_out):
        t = math.pi * i / max(n_out - 1, 1)
        X.append([math.cos(t), math.sin(t)])
        y.append(0.0)
    for i in range(n_in):
        t = math.pi * i / max(n_in - 1, 1)
        X.append([1.0 - math.cos(t), 1.0 - math.sin(t) - 0.5])
        y.append(1.0)
    X = [[v + noise * rng.normal() for v in row] for row in X]
    order = list(range(n))
    rng.shuffle(order)
    return [X[i] for i in order], [y[i] for i in order]


# ---------------------------------------------------------------------------
# Friedman #1 (regression, d = 10, five informative features)
# ---------------------------------------------------------------------------


def friedman1(x: Sequence[float]) -> float:
    return (10.0 * math.sin(math.pi * x[0] * x[1]) + 20.0 * (x[2] - 0.5) ** 2
            + 10.0 * x[3] + 5.0 * x[4])


def make_friedman1(n: int, noise: float, rng: Rng, d: int = 10) -> tuple[list[list[float]], list[float]]:
    X = [[rng.uniform() for _ in range(d)] for _ in range(n)]
    y = [friedman1(r) + noise * rng.normal() for r in X]
    return X, y


# ---------------------------------------------------------------------------
# Synthetic tabular classification (d = 8, five informative features)
# ---------------------------------------------------------------------------


def synth_logit(x: Sequence[float]) -> float:
    return (2.0 * math.sin(1.5 * x[0]) + 1.5 * x[1] * x[2] + 2.0 * math.tanh(2.0 * x[3])
            - 0.8 * x[4] * x[4] + 0.8)


def synth_prob(x: Sequence[float]) -> float:
    return sigmoid(synth_logit(x))


def make_synth_classification(n: int, rng: Rng, d: int = 8) -> tuple[list[list[float]], list[float]]:
    X = [[rng.normal() for _ in range(d)] for _ in range(n)]
    y = [1.0 if rng.uniform() < synth_prob(r) else 0.0 for r in X]
    return X, y


def split(X: list, y: list, n_train: int) -> tuple[list, list, list, list]:
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
