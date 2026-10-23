"""Shared experiment infrastructure: black-box settings, methods, per-instance evaluation."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from atlime import (ArcTanhConfig, ArcTanhLime, ArcTanhResult, DeltaAnchorEnsemble, DeltaConfig, LimeConfig,
                    LimeTabular, Rng, training_bounds, training_scale)
from atlime.datasets import (ONE_D_DOMAIN, ONE_D_FUNCTIONS, friedman1, make_friedman1, make_moons,
                             make_synth_classification, one_d_black_box, synth_prob)
from atlime.lime import default_kernel_width
from atlime.metrics import cosine, mse, neighborhood_set, sphere_mse, sphere_set, true_gradient, weighted_mse
from atlime.models import MLP, CountingBlackBox, RandomForest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"

RADII = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]   # standard deviations from the explained instance
SPHERE_POINTS = 100
NEIGHBORHOOD_POINTS = 500
GLOBAL_POINTS = 400
NUM_SAMPLES = 1000                              # LIME perturbations (also used by ArcTanh-LIME)


# ---------------------------------------------------------------------------
# Black-box settings
# ---------------------------------------------------------------------------

@dataclass
class Setting:
    name: str
    title: str
    f: Callable
    d: int
    scale: list[float]
    bounds: list[tuple[float, float]] | None
    test_X: list[list[float]]
    instances: list[list[float]]
    smooth: bool                    # gradient of the black box is meaningful
    output: str                     # 'probability' | 'regression' | 'curve'
    info: dict = field(default_factory=dict)

    def output_range(self) -> tuple[float, float]:
        """Range of the black box over the held-out data (the 'global' asymptotes ablation)."""
        if "output_range" not in self.info:
            vals = [self.f(x) for x in self.test_X]
            self.info["output_range"] = (min(vals), max(vals))
        return self.info["output_range"]


def _save(name: str, model) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / f"{name}.json").write_text(json.dumps(model.to_dict()))


def _load(name: str):
    data = json.loads((MODELS / f"{name}.json").read_text())
    return MLP.from_dict(data) if data["type"] == "mlp" else RandomForest.from_dict(data)


def _data_moons():
    X, y = make_moons(1000, 0.15, Rng(100))
    return X[:700], y[:700], X[700:], y[700:]


def _data_synth():
    X, y = make_synth_classification(4000, Rng(200))
    return X[:3000], y[:3000], X[3000:], y[3000:]


def _data_friedman():
    X, y = make_friedman1(3000, 1.0, Rng(300))
    return X[:2400], y[:2400], X[2400:], y[2400:]


def _accuracy(model, X, y) -> float:
    return sum((model(x) > 0.5) == (t > 0.5) for x, t in zip(X, y)) / len(X)


def _r2(model, X, y) -> float:
    m = sum(y) / len(y)
    ss_res = sum((model(x) - t) ** 2 for x, t in zip(X, y))
    ss_tot = sum((t - m) ** 2 for t in y)
    return 1.0 - ss_res / ss_tot


def prepare_models(verbose: bool = True) -> dict:
    """Train (or load cached) black boxes. Returns their held-out quality."""
    quality = {}
    path = MODELS / "quality.json"
    if path.exists():
        quality = json.loads(path.read_text())

    def have(name):
        return (MODELS / f"{name}.json").exists() and name in quality

    if not have("moons_mlp"):
        Xtr, ytr, Xte, yte = _data_moons()
        t0 = time.time()
        net = MLP([2, 24, 24, 1], "sigmoid", seed=1)
        net.fit(Xtr, ytr, epochs=250, batch_size=32, lr=0.01, seed=2)
        _save("moons_mlp", net)
        quality["moons_mlp"] = {"test_accuracy": _accuracy(net, Xte, yte), "train_seconds": time.time() - t0}
    if not have("synth_mlp"):
        Xtr, ytr, Xte, yte = _data_synth()
        t0 = time.time()
        net = MLP([8, 32, 1], "sigmoid", seed=3)
        net.fit(Xtr, ytr, epochs=80, batch_size=32, lr=0.005, seed=4)
        _save("synth_mlp", net)
        quality["synth_mlp"] = {"test_accuracy": _accuracy(net, Xte, yte), "train_seconds": time.time() - t0}
    if not have("synth_rf"):
        Xtr, ytr, Xte, yte = _data_synth()
        t0 = time.time()
        rf = RandomForest(n_trees=40, max_depth=10, min_samples_leaf=5, seed=5).fit(Xtr, ytr)
        _save("synth_rf", rf)
        quality["synth_rf"] = {"test_accuracy": _accuracy(rf, Xte, yte), "train_seconds": time.time() - t0}
    if not have("friedman_mlp"):
        Xtr, ytr, Xte, yte = _data_friedman()
        t0 = time.time()
        net = MLP([10, 32, 1], "linear", seed=6)
        net.fit(Xtr, ytr, epochs=150, batch_size=32, lr=0.005, seed=7)
        _save("friedman_mlp", net)
        quality["friedman_mlp"] = {"test_r2": _r2(net, Xte, yte), "train_seconds": time.time() - t0}
    if "synth_oracle" not in quality:
        _, _, Xte, yte = _data_synth()
        quality["synth_oracle"] = {"test_accuracy": _accuracy(synth_prob, Xte, yte)}
    if "friedman_oracle" not in quality:
        _, _, Xte, yte = _data_friedman()
        quality["friedman_oracle"] = {"test_r2": _r2(friedman1, Xte, yte)}
    MODELS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quality, indent=2))
    if verbose:
        for k, v in quality.items():
            print(f"  {k:16s} {v}")
    return quality


def _one_d_setting(fname: str) -> Setting:
    lo, hi = ONE_D_DOMAIN
    grid = [[lo + (hi - lo) * i / 240] for i in range(241)]
    instances = [[-5.0 + 10.0 * i / 59] for i in range(60)]
    return Setting(f"1d_{fname}", f"1-D {fname}", one_d_black_box(fname), 1, [1.0], [ONE_D_DOMAIN], grid,
                   instances, True, "curve")


_CACHE: dict[str, Setting] = {}


def get_setting(name: str) -> Setting:
    if name in _CACHE:
        return _CACHE[name]
    if name.startswith("1d_"):
        s = _one_d_setting(name[3:])
    elif name == "moons_mlp":
        Xtr, _, Xte, _ = _data_moons()
        # the global test set excludes the explained instances (moons only has 240 test points left)
        s = Setting(name, "Two moons - MLP (2-D)", _load("moons_mlp"), 2, training_scale(Xtr),
                    training_bounds(Xtr), Xte[60:60 + GLOBAL_POINTS], Xte[:60], True, "probability")
    elif name in ("synth_oracle", "synth_mlp", "synth_rf"):
        Xtr, _, Xte, _ = _data_synth()
        f = synth_prob if name == "synth_oracle" else _load(name)
        title = {"synth_oracle": "Synthetic 8-D - true probability", "synth_mlp": "Synthetic 8-D - MLP",
                 "synth_rf": "Synthetic 8-D - random forest"}[name]
        s = Setting(name, title, f, 8, training_scale(Xtr), training_bounds(Xtr), Xte[40:40 + GLOBAL_POINTS],
                    Xte[:40], name != "synth_rf", "probability")
    elif name in ("friedman_oracle", "friedman_mlp"):
        Xtr, _, Xte, _ = _data_friedman()
        f = friedman1 if name == "friedman_oracle" else _load(name)
        title = {"friedman_oracle": "Friedman #1 - true function", "friedman_mlp": "Friedman #1 - MLP"}[name]
        s = Setting(name, title, f, 10, training_scale(Xtr), [(0.0, 1.0)] * 10, Xte[40:40 + GLOBAL_POINTS],
                    Xte[:40], True, "regression")
    else:
        raise KeyError(name)
    _CACHE[name] = s
    return s


ONE_D = [f"1d_{k}" for k in ONE_D_FUNCTIONS]
TABULAR = ["moons_mlp", "synth_oracle", "synth_mlp", "synth_rf", "friedman_oracle", "friedman_mlp"]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

MAIN_METHODS = ["LIME", "LIME (matched budget)", "LIME (same data)", "ArcTanh-LIME", "LIME-δ", "ArcTanh-LIME-δ"]


def build(method: str, s: Setting, at_overrides: dict | None = None, delta_overrides: dict | None = None,
          num_samples: int = NUM_SAMPLES):
    at_cfg = replace(ArcTanhConfig(num_samples=num_samples), **(at_overrides or {}))
    delta_cfg = replace(DeltaConfig(), **(delta_overrides or {}))
    lime = LimeTabular(s.scale, LimeConfig(num_samples=num_samples))
    if method == "LIME":
        return lime
    if method == "LIME (pinned intercept)":
        return LimeTabular(s.scale, LimeConfig(num_samples=num_samples, pin_intercept=True))
    if method == "ArcTanh-LIME":
        return ArcTanhLime(s.scale, at_cfg, s.bounds)
    if method == "ArcTanh-LIME (unbounded walk)":
        return ArcTanhLime(s.scale, at_cfg, None)
    if method == "LIME-δ":
        return DeltaAnchorEnsemble(lime, delta_cfg, s.bounds)
    if method == "ArcTanh-LIME-δ":
        return DeltaAnchorEnsemble(ArcTanhLime(s.scale, at_cfg, s.bounds), delta_cfg, s.bounds)
    raise KeyError(method)


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(task: dict) -> dict:
    """task = {setting, index, seed, methods: [(label, method, at_overrides, delta_overrides)]}.

    All methods see the same random stream (so LIME and ArcTanh-LIME share
    their perturbation sample) and are scored on the same evaluation queries,
    which are drawn independently of the explainers.

    Baselines derived from the preceding "ArcTanh-LIME" run:
      * "LIME (matched budget)": LIME with N = ArcTanh-LIME's query count;
      * "LIME (same data)": LIME's weighted ridge fitted on exactly ArcTanh-LIME's
        regression data (LIME sample + walk path), i.e. only the surrogate differs.
    """
    s = get_setting(task["setting"])
    x0 = s.instances[task["index"]]
    seed = task["seed"]
    eval_rng = Rng(seed + 7919)
    sphere = sphere_set(s.f, x0, s.scale, RADII, SPHERE_POINTS, eval_rng, s.bounds)
    nbhd = neighborhood_set(s.f, x0, s.scale, NEIGHBORHOOD_POINTS, default_kernel_width(s.d), eval_rng)
    global_vals = [s.f(p) for p in s.test_X]
    grad = true_gradient(s.f, x0, s.scale) if s.smooth else None
    f0 = s.f(x0)
    out = {"setting": s.name, "index": task["index"], "x0": x0, "f0": f0, "methods": {}}
    queries_at = None
    last_at = None
    for label, method, at_over, delta_over in task["methods"]:
        f = CountingBlackBox(s.f)
        if method in ("LIME (matched budget)", "LIME (same data)") and last_at is None:
            raise ValueError(f"{method} must come after ArcTanh-LIME")
        t0 = time.perf_counter()
        if method == "LIME (same data)":
            res = LimeTabular(s.scale, LimeConfig(num_samples=NUM_SAMPLES)).fit(x0, last_at.fit_data)
            queries = queries_at  # no new queries: the same points ArcTanh-LIME paid for
        else:
            explainer = (build("LIME", s, num_samples=queries_at) if method == "LIME (matched budget)"
                         else build(method, s, at_over, delta_over))
            res = explainer.explain(f, x0, Rng(seed))
            queries = f.calls
        elapsed = time.perf_counter() - t0
        g = res.surrogate
        attribution = res.attribution
        rec = {
            "queries": queries,
            "seconds": elapsed,
            "sphere_mse": sphere_mse(g, sphere),
            "neighborhood_wmse": weighted_mse(g, nbhd["points"], nbhd["values"], nbhd["weights"]),
            "global_mse": mse(g, s.test_X, global_vals),
            "local_abs_error": abs(g.predict(x0) - f0),
            "attribution": attribution,
        }
        if grad is not None:
            rec["grad_cosine"] = cosine(attribution, grad)
            gn = math.sqrt(sum(v * v for v in grad))
            rec["grad_rel_error"] = (math.sqrt(sum((a - b) ** 2 for a, b in zip(attribution, grad))) / gn
                                     if gn > 0 else None)
            if hasattr(res, "blend_gradient"):
                rec["blend_grad_cosine"] = cosine(res.blend_gradient, grad)
        if isinstance(res, ArcTanhResult):
            rec.update({
                "lo": res.lo, "hi": res.hi, "raw_lo": res.raw_lo, "raw_hi": res.raw_hi,
                "walk_up_steps": res.walk_up.steps if res.walk_up else 0,
                "walk_down_steps": res.walk_down.steps if res.walk_down else 0,
                "walk_up_reason": res.walk_up.stop_reason if res.walk_up else None,
                "walk_down_reason": res.walk_down.stop_reason if res.walk_down else None,
                "clipped_fraction": res.clipped_fraction, "fallback": res.fallback,
            })
            if label == "ArcTanh-LIME":
                queries_at = f.calls
                last_at = res
        out["methods"][label] = rec
    return out


def main_tasks(setting: str, seed: int = 0) -> list[dict]:
    s = get_setting(setting)
    methods = [("LIME", "LIME", None, None), ("ArcTanh-LIME", "ArcTanh-LIME", None, None),
               ("LIME (same data)", "LIME (same data)", None, None),
               ("LIME (matched budget)", "LIME (matched budget)", None, None),
               ("LIME-δ", "LIME-δ", None, None), ("ArcTanh-LIME-δ", "ArcTanh-LIME-δ", None, None)]
    return [{"setting": setting, "index": i, "seed": 1000 * seed + i + 1, "methods": methods}
            for i in range(len(s.instances))]


def run_parallel(tasks: list[dict], workers: int | None = None, fn=evaluate_instance) -> list[dict]:
    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=1)
