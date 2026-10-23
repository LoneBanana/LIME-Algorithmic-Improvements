"""LIME for tabular data, implemented from scratch.

Follows the reference implementation of Ribeiro, Singh & Guestrin (2016)
(``lime.lime_tabular.LimeTabularExplainer`` + ``lime.lime_base.LimeBase``) in
its continuous mode, i.e. ``discretize_continuous=False`` and
``sample_around_instance=True``:

1. perturb the instance with Gaussian noise scaled by the training standard
   deviation of each feature (the first sample is the instance itself);
2. query the black box on every perturbation;
3. weight each perturbation with the exponential kernel
   ``sqrt(exp(-d^2 / width^2))`` of its Euclidean distance d in standardised
   space, ``width = 0.75 * sqrt(n_features)`` by default;
4. optionally select K features (forward selection / highest weights);
5. fit a weighted ridge regression (alpha = 1) on the standardised features.

The returned explanation is the linear surrogate; its coefficients are the
feature attributions (per standard deviation of each feature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .linalg import weighted_r2, weighted_ridge
from .rng import Rng
from .surrogates import LinearSurrogate

BlackBox = Callable[[Sequence[float]], float]


def default_kernel_width(n_features: int) -> float:
    return 0.75 * math.sqrt(n_features)


def lime_kernel(distance: float, width: float) -> float:
    return math.sqrt(math.exp(-(distance * distance) / (width * width)))


def training_scale(rows: Sequence[Sequence[float]]) -> list[float]:
    """Per-feature population standard deviation (StandardScaler semantics:
    zero-variance features get scale 1)."""
    n = len(rows)
    d = len(rows[0])
    scale = []
    for j in range(d):
        col = [r[j] for r in rows]
        m = math.fsum(col) / n
        s = math.sqrt(math.fsum((v - m) ** 2 for v in col) / n)
        scale.append(s if s > 0.0 else 1.0)
    return scale


def training_bounds(rows: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    d = len(rows[0])
    return [(min(r[j] for r in rows), max(r[j] for r in rows)) for j in range(d)]


@dataclass
class Neighborhood:
    """Perturbation sample: original-space points, standardised offsets Z
    (Z[0] is the instance itself), black-box outputs y and kernel weights w."""

    points: list[list[float]]
    Z: list[list[float]]
    y: list[float]
    w: list[float]


def sample_neighborhood(f: BlackBox, x0: Sequence[float], scale: Sequence[float], num_samples: int,
                        kernel_width: float, rng: Rng) -> Neighborhood:
    d = len(x0)
    points = [list(x0)]
    Z = [[0.0] * d]
    for _ in range(num_samples - 1):
        z = [rng.normal() for _ in range(d)]
        Z.append(z)
        points.append([xi + zi * si for xi, zi, si in zip(x0, z, scale)])
    y = [float(f(p)) for p in points]
    w = [lime_kernel(math.sqrt(math.sumprod(z, z)), kernel_width) for z in Z]
    return Neighborhood(points, Z, y, w)


def _subset(Z: Sequence[Sequence[float]], feats: Sequence[int]) -> list[list[float]]:
    return [[z[j] for j in feats] for z in Z]


def select_features(Z: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float], method: str,
                    num_features: int) -> list[int]:
    d = len(Z[0])
    k = min(num_features, d)
    if method == "none" or k >= d:
        return list(range(d))
    if method == "auto":
        method = "forward_selection" if k <= 6 else "highest_weights"
    if method == "highest_weights":
        # LIME ranks by |coef_j * z0_j| with z0 the instance in training-mean
        # centred units; for continuous features that ranks features by how far
        # the instance is from the training mean. We rank by |coef_j| instead.
        coef, _ = weighted_ridge(Z, y, w, alpha=0.01)
        order = sorted(range(d), key=lambda j: abs(coef[j]), reverse=True)
        return sorted(order[:k])
    if method == "forward_selection":
        used: list[int] = []
        for _ in range(k):
            best, best_score = -1, -math.inf
            for j in range(d):
                if j in used:
                    continue
                cand = used + [j]
                rows = _subset(Z, cand)
                coef, b = weighted_ridge(rows, y, w, alpha=0.0)
                pred = [b + math.sumprod(coef, r) for r in rows]
                score = weighted_r2(y, pred, w)
                if score > best_score:
                    best, best_score = j, score
            used.append(best)
        return sorted(used)
    raise ValueError(f"unknown feature selection method: {method}")


@dataclass
class LimeConfig:
    num_samples: int = 1000
    kernel_width: float | None = None   # default 0.75 * sqrt(n_features)
    alpha: float = 1.0                  # ridge penalty (LIME uses Ridge(alpha=1))
    feature_selection: str = "none"     # 'none' | 'auto' | 'forward_selection' | 'highest_weights'
    num_features: int | None = None
    pin_intercept: bool = False         # NOT standard LIME: force g(x0) = f(x0) (ablation only)


@dataclass
class LimeResult:
    surrogate: LinearSurrogate
    score: float                 # weighted R^2 on the perturbation sample
    local_pred: float            # surrogate prediction at the instance
    used_features: list[int]
    neighborhood: Neighborhood
    n_queries: int
    extras: dict = field(default_factory=dict)

    @property
    def attribution(self) -> list[float]:
        return self.surrogate.coef


class LimeTabular:
    """The baseline explainer (one linear surrogate per instance)."""

    def __init__(self, scale: Sequence[float], config: LimeConfig | None = None) -> None:
        self.scale = list(scale)
        self.config = config or LimeConfig()

    @classmethod
    def from_training_data(cls, rows: Sequence[Sequence[float]], config: LimeConfig | None = None) -> "LimeTabular":
        return cls(training_scale(rows), config)

    def kernel_width(self) -> float:
        return self.config.kernel_width or default_kernel_width(len(self.scale))

    def neighborhood(self, f: BlackBox, x0: Sequence[float], rng: Rng) -> Neighborhood:
        return sample_neighborhood(f, x0, self.scale, self.config.num_samples, self.kernel_width(), rng)

    def fit(self, x0: Sequence[float], nb: Neighborhood) -> LimeResult:
        cfg = self.config
        d = len(x0)
        feats = select_features(nb.Z, nb.y, nb.w, cfg.feature_selection, cfg.num_features or d)
        if cfg.pin_intercept:
            intercept = nb.y[0]
            coef_sel, _ = weighted_ridge(_subset(nb.Z, feats), [v - intercept for v in nb.y], nb.w, alpha=cfg.alpha,
                                         fit_intercept=False)
        else:
            coef_sel, intercept = weighted_ridge(_subset(nb.Z, feats), nb.y, nb.w, alpha=cfg.alpha)
        coef = [0.0] * d
        for j, c in zip(feats, coef_sel):
            coef[j] = c
        surrogate = LinearSurrogate(x0, self.scale, intercept, coef)
        pred = [intercept + math.sumprod(coef, z) for z in nb.Z]
        return LimeResult(surrogate, weighted_r2(nb.y, pred, nb.w), intercept, feats, nb, len(nb.y))

    def explain(self, f: BlackBox, x0: Sequence[float], rng: Rng) -> LimeResult:
        return self.fit(x0, self.neighborhood(f, x0, rng))
