"""Interpretable surrogate models returned by the explainers.

All surrogates work in *standardised offsets* from their anchor point,

    z_j = (x_j - anchor_j) / scale_j ,

where ``scale_j`` is the training-data standard deviation of feature j (the
same scaling LIME applies before fitting). Gradients are therefore reported
"per standard deviation of the feature", the unit of LIME's coefficients.
"""

from __future__ import annotations

import math
from typing import Sequence


def standardize(x: Sequence[float], anchor: Sequence[float], scale: Sequence[float]) -> list[float]:
    return [(xi - ai) / si for xi, ai, si in zip(x, anchor, scale)]


class LinearSurrogate:
    """g(x) = intercept + coef . z   (classic LIME)."""

    kind = "linear"

    def __init__(self, anchor: Sequence[float], scale: Sequence[float], intercept: float, coef: Sequence[float]) -> None:
        self.anchor = list(anchor)
        self.scale = list(scale)
        self.intercept = float(intercept)
        self.coef = list(coef)

    def predict(self, x: Sequence[float]) -> float:
        return self.intercept + math.sumprod(self.coef, standardize(x, self.anchor, self.scale))

    def gradient(self, x: Sequence[float]) -> list[float]:
        return list(self.coef)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "anchor": self.anchor, "scale": self.scale,
                "intercept": self.intercept, "coef": self.coef}


class TanhSurrogate:
    """g(x) = c + a * tanh(beta0 + beta . z)   with  c = (hi+lo)/2,  a = (hi-lo)/2.

    Equivalently, in *link space*:  artanh((g - c) / a) = beta0 + beta . z,
    i.e. a linear model on the tanh^-1-transformed output whose vertical
    asymptotes sit at the output values ``lo`` and ``hi`` (the local minimum
    and maximum found by the epsilon-walk).
    """

    kind = "tanh"

    def __init__(self, anchor: Sequence[float], scale: Sequence[float], lo: float, hi: float,
                 beta0: float, beta: Sequence[float]) -> None:
        if not hi > lo:
            raise ValueError("TanhSurrogate needs hi > lo")
        self.anchor = list(anchor)
        self.scale = list(scale)
        self.lo = float(lo)
        self.hi = float(hi)
        self.beta0 = float(beta0)
        self.beta = list(beta)

    @property
    def center(self) -> float:
        return 0.5 * (self.hi + self.lo)

    @property
    def half_range(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def eta(self, x: Sequence[float]) -> float:
        return self.beta0 + math.sumprod(self.beta, standardize(x, self.anchor, self.scale))

    def predict(self, x: Sequence[float]) -> float:
        return self.center + self.half_range * math.tanh(self.eta(x))

    def gradient(self, x: Sequence[float]) -> list[float]:
        t = math.tanh(self.eta(x))
        s = self.half_range * (1.0 - t * t)
        return [s * b for b in self.beta]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "anchor": self.anchor, "scale": self.scale, "lo": self.lo,
                "hi": self.hi, "beta0": self.beta0, "beta": self.beta}


class BlendedSurrogate:
    """Partition-of-unity blend of member surrogates fitted at different anchors.

        G(x) = sum_a p_a(x) g_a(x),   p_a(x) = v_a(x) / sum_b v_b(x),
        v_a(x) = exp(-||z(x) - z_a||^2 / (2 h^2)),

    with z measured in standard deviations around ``center`` (the explained
    instance) and h the blending bandwidth.
    """

    kind = "blend"

    def __init__(self, center: Sequence[float], scale: Sequence[float], anchors: Sequence[Sequence[float]],
                 members: Sequence, bandwidth: float) -> None:
        if len(anchors) != len(members) or not members:
            raise ValueError("BlendedSurrogate needs one member per anchor")
        self.center = list(center)
        self.scale = list(scale)
        self.anchors = [list(a) for a in anchors]
        self.members = list(members)
        self.bandwidth = float(bandwidth)
        self._anchor_z = [standardize(a, self.center, self.scale) for a in self.anchors]

    def weights(self, x: Sequence[float]) -> list[float]:
        z = standardize(x, self.center, self.scale)
        d2 = [math.fsum((zi - ai) ** 2 for zi, ai in zip(z, za)) for za in self._anchor_z]
        m = min(d2)
        inv = 1.0 / (2.0 * self.bandwidth * self.bandwidth)
        v = [math.exp(-(e - m) * inv) for e in d2]  # shifted for numerical safety
        total = math.fsum(v)
        return [vi / total for vi in v]

    def predict(self, x: Sequence[float]) -> float:
        p = self.weights(x)
        return math.fsum(pi * g.predict(x) for pi, g in zip(p, self.members) if pi > 0.0)

    def gradient(self, x: Sequence[float]) -> list[float]:
        """Analytic gradient: sum_a p_a grad g_a + (1/h^2) sum_a p_a g_a (z_a - zbar)."""
        p = self.weights(x)
        d = len(self.center)
        zbar = [math.fsum(pi * za[j] for pi, za in zip(p, self._anchor_z)) for j in range(d)]
        out = [0.0] * d
        inv_h2 = 1.0 / (self.bandwidth * self.bandwidth)
        for pi, g, za in zip(p, self.members, self._anchor_z):
            if pi == 0.0:
                continue
            gx = g.predict(x)
            grad = g.gradient(x)
            for j in range(d):
                out[j] += pi * grad[j] + inv_h2 * pi * gx * (za[j] - zbar[j])
        return out

    def to_dict(self) -> dict:
        return {"kind": self.kind, "center": self.center, "scale": self.scale, "anchors": self.anchors,
                "bandwidth": self.bandwidth, "members": [m.to_dict() for m in self.members]}
