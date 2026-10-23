"""ArcTanh-LIME: a tanh^-1 regressor anchored on the black box's local extrema.

Procedure for an instance x0 (all distances in standard deviations):

1. Sample the usual LIME neighbourhood around x0 and query the black box.
2. epsilon-walk *up* the response surface from x0 until the nearest local
   maximum, and *down* until the nearest local minimum (``walk.py``). Their
   output values are ``hi`` and ``lo``.
3. Treat ``lo`` / ``hi`` as the asymptotes of tanh^-1: map every observed
   output into (-1, 1) with  u = (y - c) / a,  c = (hi+lo)/2, a = (hi-lo)/2,
   and regress  t = artanh(u)  linearly on the standardised offsets z (with
   the LIME kernel weights). x0 is the origin of the regressor (z = 0).
       artanh((f(x) - c) / a)  ~  beta0 + beta . z
   which in output space is the bounded, S-shaped surrogate
       g(x) = c + a * tanh(beta0 + beta . z).
4. (default) Refine beta0, beta by weighted least squares *in output space*
   (Levenberg-Marquardt, initialised at the link-space solution), because
   errors in link space are inflated near the asymptotes.

The attribution reported for x0 is the local slope of g at x0,
a * (1 - tanh(beta0)^2) * beta, which has the same units as LIME's
coefficients; beta itself describes the whole S-curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .lime import Neighborhood, default_kernel_width, lime_kernel, sample_neighborhood
from .linalg import solve_spd, weighted_r2, weighted_ridge
from .rng import Rng
from .surrogates import LinearSurrogate, TanhSurrogate, standardize
from .walk import Bounds, WalkResult, epsilon_walk, finite_difference_gradient

BlackBox = Callable[[Sequence[float]], float]


@dataclass
class ArcTanhConfig:
    # LIME neighbourhood (identical meaning to LimeConfig)
    num_samples: int = 1000
    kernel_width: float | None = None
    alpha: float = 1.0
    # epsilon-walk
    epsilon: float = 0.1                # step width, in standard deviations
    max_steps: int = 40                 # per direction
    grad_step: float | None = None      # finite-difference step; default = epsilon
    tol_rel: float = 1e-3               # stop when a step gains < tol_rel * (climb so far)
    # asymptotes
    asymptotes: str = "walk"            # 'walk' | 'walk+samples' | 'fixed'
    fixed_bounds: tuple[float, float] | None = None
    margin: float = 0.05                # widen [lo, hi] by this fraction of hi - lo on each side
    clip: float = 1e-3                  # keep |u| <= 1 - clip before artanh
    # regression
    fit: str = "output"                 # 'link' (pure tanh^-1 regression) | 'output' (+ LM refinement)
    pin_origin: bool = False            # force g(x0) = f(x0)
    use_walk_points: bool = True        # add the walk path to the regression data
    lm_max_iter: int = 50


@dataclass
class ArcTanhResult:
    surrogate: TanhSurrogate | LinearSurrogate
    link_surrogate: TanhSurrogate | LinearSurrogate   # step-3 solution before refinement
    walk_up: WalkResult | None
    walk_down: WalkResult | None
    raw_lo: float
    raw_hi: float
    lo: float
    hi: float
    score: float                # weighted R^2 (output space) on the LIME neighbourhood
    local_pred: float
    clipped_fraction: float     # share of targets that fell outside (lo, hi)
    fallback: str | None
    neighborhood: Neighborhood
    fit_data: Neighborhood      # regression data actually used: LIME sample + walk path
    n_queries: int
    extras: dict = field(default_factory=dict)

    @property
    def attribution(self) -> list[float]:
        return self.surrogate.gradient(self.surrogate.anchor)


def _tanh_loss(Z, y, w, c, a, b0, beta, penalty) -> float:
    s = 0.0
    for z, yi, wi in zip(Z, y, w):
        r = yi - c - a * math.tanh(b0 + math.sumprod(beta, z))
        s += wi * r * r
    return s + penalty * math.sumprod(beta, beta)


def fit_tanh_output(Z: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float], c: float, a: float,
                    beta0: float, beta: Sequence[float], penalty: float, pin_origin: bool,
                    max_iter: int = 50) -> tuple[float, list[float], dict]:
    """Levenberg-Marquardt for  min sum w (y - c - a tanh(b0 + beta.z))^2 + penalty ||beta||^2."""
    d = len(beta)
    b0 = float(beta0)
    beta = list(beta)
    loss = _tanh_loss(Z, y, w, c, a, b0, beta, penalty)
    lam = 1e-3
    offset = 0 if pin_origin else 1
    p = d + offset
    iters = 0
    for iters in range(1, max_iter + 1):
        # Jacobian of the prediction: a * sech^2(eta) * [1, z]
        gram = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        for z, yi, wi in zip(Z, y, w):
            t = math.tanh(b0 + math.sumprod(beta, z))
            s = a * (1.0 - t * t)
            r = yi - c - a * t
            row = ([1.0] + list(z)) if offset else list(z)
            ws = wi * s
            ws2 = ws * s
            for j in range(p):
                rj = row[j]
                rhs[j] += ws * r * rj
                gj = gram[j]
                k_val = ws2 * rj
                for k in range(j, p):
                    gj[k] += k_val * row[k]
        for j in range(p):
            for k in range(j):
                gram[j][k] = gram[k][j]
        for j in range(d):
            gram[j + offset][j + offset] += penalty
            rhs[j + offset] -= penalty * beta[j]
        improved = False
        while lam <= 1e10:
            damped = [row[:] for row in gram]
            for j in range(p):
                damped[j][j] += lam * max(gram[j][j], 1e-12)
            delta = solve_spd(damped, rhs)
            nb0 = b0 + delta[0] if offset else b0
            nbeta = [bj + dj for bj, dj in zip(beta, delta[offset:])]
            new_loss = _tanh_loss(Z, y, w, c, a, nb0, nbeta, penalty)
            if new_loss < loss:
                gain = loss - new_loss
                b0, beta, loss = nb0, nbeta, new_loss
                lam = max(lam / 3.0, 1e-12)
                improved = gain > 1e-12 * max(loss, 1e-300)
                break
            lam *= 4.0
        if not improved:
            break
    return b0, beta, {"iterations": iters, "loss": loss}


class ArcTanhLime:
    """The proposed explainer: epsilon-walk to the local extrema + tanh^-1 regression."""

    def __init__(self, scale: Sequence[float], config: ArcTanhConfig | None = None, bounds: Bounds = None) -> None:
        self.scale = list(scale)
        self.config = config or ArcTanhConfig()
        self.bounds = list(bounds) if bounds is not None else None

    def kernel_width(self) -> float:
        return self.config.kernel_width or default_kernel_width(len(self.scale))

    def neighborhood(self, f: BlackBox, x0: Sequence[float], rng: Rng) -> Neighborhood:
        return sample_neighborhood(f, x0, self.scale, self.config.num_samples, self.kernel_width(), rng)

    def walks(self, f: BlackBox, x0: Sequence[float], f0: float) -> tuple[WalkResult, WalkResult]:
        """Both walks; they share the gradient at x0, whose queries are counted in the up walk."""
        cfg = self.config
        h = cfg.grad_step if cfg.grad_step is not None else cfg.epsilon
        g0, q0 = finite_difference_gradient(f, x0, self.scale, h, self.bounds) if cfg.max_steps > 0 else ([], 0)
        up = epsilon_walk(f, x0, f0, self.scale, +1, cfg.epsilon, cfg.max_steps, h, cfg.tol_rel, self.bounds, g0)
        down = epsilon_walk(f, x0, f0, self.scale, -1, cfg.epsilon, cfg.max_steps, h, cfg.tol_rel, self.bounds, g0)
        up.n_queries += q0
        return up, down

    def explain(self, f: BlackBox, x0: Sequence[float], rng: Rng, nb: Neighborhood | None = None) -> ArcTanhResult:
        cfg = self.config
        d = len(x0)
        n_queries = 0
        if nb is None:
            nb = self.neighborhood(f, x0, rng)
            n_queries += len(nb.y)
        f0 = nb.y[0]

        # --- step 2: asymptotes ------------------------------------------------
        up = down = None
        if cfg.asymptotes == "fixed":
            if cfg.fixed_bounds is None:
                raise ValueError("asymptotes='fixed' needs fixed_bounds")
            raw_lo, raw_hi = cfg.fixed_bounds
        else:
            up, down = self.walks(f, x0, f0)
            n_queries += up.n_queries + down.n_queries
            raw_lo, raw_hi = down.end_value, up.end_value
            if cfg.asymptotes == "walk+samples":
                raw_lo = min(raw_lo, min(nb.y))
                raw_hi = max(raw_hi, max(nb.y))
            elif cfg.asymptotes != "walk":
                raise ValueError(f"unknown asymptotes mode: {cfg.asymptotes}")

        # --- regression data: LIME sample (+ walk path) ------------------------
        Z = [list(z) for z in nb.Z]
        y = list(nb.y)
        w = list(nb.w)
        kw = self.kernel_width()
        if cfg.use_walk_points:
            for walk in (up, down):
                if walk is None:
                    continue
                for pt, val in zip(walk.points[1:], walk.values[1:]):
                    z = standardize(pt, x0, self.scale)
                    Z.append(z)
                    y.append(val)
                    w.append(lime_kernel(math.sqrt(math.sumprod(z, z)), kw))

        fit_data = Neighborhood([], Z, y, w)
        lo, hi = raw_lo, raw_hi
        fallback = None
        tiny = 1e-12 * max(1.0, abs(f0))
        if hi - lo <= tiny:
            # Both walks stalled immediately (flat start): use the observed range.
            lo, hi = min(min(y), lo), max(max(y), hi)
            fallback = "sample-range"
        if hi - lo <= tiny:
            constant = LinearSurrogate(x0, self.scale, f0, [0.0] * d)
            score = weighted_r2(nb.y, [f0] * len(nb.y), nb.w)
            return ArcTanhResult(constant, constant, up, down, raw_lo, raw_hi, lo, hi, score, f0, 0.0, "constant",
                                 nb, fit_data, n_queries)
        span = hi - lo
        lo -= cfg.margin * span
        hi += cfg.margin * span
        c = 0.5 * (hi + lo)
        a = 0.5 * (hi - lo)

        # --- step 3: tanh^-1 (link-space) regression ---------------------------
        lim = 1.0 - cfg.clip
        u = [(yi - c) / a for yi in y]
        clipped = sum(1 for v in u if abs(v) > lim) / len(u)
        t = [math.atanh(max(-lim, min(lim, v))) for v in u]
        if cfg.pin_origin:
            b0 = math.atanh(max(-lim, min(lim, (f0 - c) / a)))
            beta, _ = weighted_ridge(Z, [ti - b0 for ti in t], w, alpha=cfg.alpha, fit_intercept=False)
        else:
            beta, b0 = weighted_ridge(Z, t, w, alpha=cfg.alpha)
        link_sur = TanhSurrogate(x0, self.scale, lo, hi, b0, beta)

        # --- step 4: optional refinement in output space -----------------------
        surrogate = link_sur
        extras: dict = {}
        if cfg.fit == "output":
            # Ridge penalty expressed on the output-space slope a*beta, the
            # analogue of LIME's penalty on its (output-space) coefficients.
            b0r, betar, info = fit_tanh_output(Z, y, w, c, a, b0, beta, cfg.alpha * a * a, cfg.pin_origin,
                                               cfg.lm_max_iter)
            surrogate = TanhSurrogate(x0, self.scale, lo, hi, b0r, betar)
            extras["lm"] = info
        elif cfg.fit != "link":
            raise ValueError(f"unknown fit mode: {cfg.fit}")

        pred = [surrogate.center + surrogate.half_range * math.tanh(surrogate.beta0 + math.sumprod(surrogate.beta, z))
                for z in nb.Z]
        score = weighted_r2(nb.y, pred, nb.w)
        return ArcTanhResult(surrogate, link_sur, up, down, raw_lo, raw_hi, lo, hi, score, pred[0], clipped,
                             fallback, nb, fit_data, n_queries, extras)
