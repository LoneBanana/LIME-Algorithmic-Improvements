"""The +/- delta anchor ensemble ("partially-global" explanations).

Run a base explainer not only at the instance x0 but also at anchors
x0 +/- k*delta along every feature axis (k = 1..rings, delta in standard
deviations), then blend the anchor surrogates with a Gaussian partition of
unity (``BlendedSurrogate``). Each anchor is explained exactly like x0
(its own walk, its own asymptotes, its own tanh^-1 regressor), so the blend
covers the union of the anchors' neighbourhoods.

The same machinery with a linear base explainer ("LIME-delta") is used as an
ablation, to separate the effect of the tanh^-1 surrogate from the effect of
simply having more anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .rng import Rng
from .surrogates import BlendedSurrogate
from .walk import Bounds

BlackBox = Callable[[Sequence[float]], float]


@dataclass
class DeltaConfig:
    delta: float = 1.0               # anchor spacing, in standard deviations
    rings: int = 1                   # anchors at +/- k*delta for k = 1..rings
    bandwidth: float | None = None   # blending bandwidth; default delta / 2
    layout: str = "axes"             # 'axes' (2*d*rings anchors) | 'random' (n_random directions per ring)
    n_random: int = 8


@dataclass
class EnsembleResult:
    surrogate: BlendedSurrogate
    members: list                    # base-explainer results, members[0] is x0 itself
    anchors: list[list[float]]
    n_queries: int
    extras: dict = field(default_factory=dict)

    @property
    def attribution(self) -> list[float]:
        # The explanation *at x0* is the x0 anchor's own; the gradient of the blend
        # also contains d(weights)/dx terms that are artefacts of the blending.
        return self.members[0].attribution

    @property
    def blend_gradient(self) -> list[float]:
        return self.surrogate.gradient(self.surrogate.center)


def anchor_points(x0: Sequence[float], scale: Sequence[float], cfg: DeltaConfig, bounds: Bounds = None,
                  rng: Rng | None = None) -> list[list[float]]:
    d = len(x0)
    dirs: list[list[float]] = []
    if cfg.layout == "axes":
        for j in range(d):
            for s in (1.0, -1.0):
                e = [0.0] * d
                e[j] = s
                dirs.append(e)
    elif cfg.layout == "random":
        if rng is None:
            raise ValueError("layout='random' needs an rng")
        for _ in range(cfg.n_random):
            v = [rng.normal() for _ in range(d)]
            n = sum(c * c for c in v) ** 0.5
            dirs.append([c / n for c in v])
            dirs.append([-c / n for c in v])
    else:
        raise ValueError(f"unknown layout: {cfg.layout}")
    anchors = [list(x0)]
    seen = {tuple(x0)}
    for k in range(1, cfg.rings + 1):
        for e in dirs:
            a = [xi + k * cfg.delta * ei * si for xi, ei, si in zip(x0, e, scale)]
            if bounds is not None:
                a = [min(max(v, lo), hi) for v, (lo, hi) in zip(a, bounds)]
            key = tuple(a)
            if key not in seen:  # clipping can collapse anchors onto each other
                seen.add(key)
                anchors.append(a)
    return anchors


class DeltaAnchorEnsemble:
    """Wraps any base explainer exposing ``.scale`` and ``.explain(f, x, rng)``."""

    def __init__(self, base, config: DeltaConfig | None = None, bounds: Bounds = None) -> None:
        self.base = base
        self.config = config or DeltaConfig()
        self.bounds = list(bounds) if bounds is not None else None

    def explain(self, f: BlackBox, x0: Sequence[float], rng: Rng) -> EnsembleResult:
        cfg = self.config
        scale = self.base.scale
        anchors = anchor_points(x0, scale, cfg, self.bounds, rng)
        members = []
        n_queries = 0
        for i, a in enumerate(anchors):
            # anchor 0 reuses the caller's stream, so with the 'axes' layout it is identical to a
            # stand-alone explanation of x0 ('random' first draws its directions from the same stream)
            res = self.base.explain(f, a, rng if i == 0 else rng.spawn(i))
            members.append(res)
            n_queries += res.n_queries
        bandwidth = cfg.bandwidth if cfg.bandwidth is not None else 0.5 * cfg.delta
        surrogate = BlendedSurrogate(x0, scale, anchors, [m.surrogate for m in members], bandwidth)
        return EnsembleResult(surrogate, members, anchors, n_queries)
