"""Export everything the website needs into web/data/.

* parity.json   - fixtures that the TypeScript port must reproduce (npm run parity)
* moons.json    - the trained two-moons MLP + its data scale/bounds + a point sample
* results.json  - benchmark / ablation / stability summaries

    python3 -m experiments.export_web
"""

from __future__ import annotations

import json

from atlime import (ArcTanhConfig, ArcTanhLime, DeltaAnchorEnsemble, DeltaConfig, LimeConfig, LimeTabular, Rng,
                    training_bounds, training_scale)
from atlime.datasets import ONE_D_FUNCTIONS, one_d_black_box
from atlime.models import MLP

from .common import MODELS, RESULTS, ROOT, _data_moons

WEB_DATA = ROOT / "web" / "data"


def _walk(w) -> dict | None:
    if w is None:
        return None
    return {"steps": w.steps, "stop_reason": w.stop_reason, "end_value": w.end_value, "n_queries": w.n_queries}


def _arctanh(res, probes) -> dict:
    s, link = res.surrogate, res.link_surrogate
    return {
        "raw_lo": res.raw_lo, "raw_hi": res.raw_hi, "lo": res.lo, "hi": res.hi, "fallback": res.fallback,
        "beta0": getattr(s, "beta0", None), "beta": getattr(s, "beta", None),
        "link_beta0": getattr(link, "beta0", None), "link_beta": getattr(link, "beta", None),
        "score": res.score, "clipped_fraction": res.clipped_fraction, "n_queries": res.n_queries,
        "walk_up": _walk(res.walk_up), "walk_down": _walk(res.walk_down),
        "attribution": res.attribution, "predictions": [s.predict(p) for p in probes],
    }


def _lime(res, probes) -> dict:
    s = res.surrogate
    return {"intercept": s.intercept, "coef": s.coef, "score": res.score,
            "predictions": [s.predict(p) for p in probes]}


def parity_fixtures() -> dict:
    out: dict = {}
    r = Rng(123)
    out["uniform_seed123"] = [r.uniform() for _ in range(8)]
    r = Rng(7)
    out["normal_seed7"] = [r.normal() for _ in range(9)]
    out["spawn_99_3"] = Rng(99).spawn(3).uniform()

    probes_1d = [[-6.0 + 0.5 * i] for i in range(25)]
    bounds_1d = [(-6.0, 6.0)]
    cases = []
    for k, name in enumerate(ONE_D_FUNCTIONS):
        f = one_d_black_box(name)
        for j, x0 in enumerate([-3.7, -0.4, 1.9]):
            seed = 10 * k + j + 1
            lime = LimeTabular([1.0], LimeConfig(num_samples=300)).explain(f, [x0], Rng(seed))
            at = ArcTanhLime([1.0], ArcTanhConfig(num_samples=300), bounds_1d).explain(f, [x0], Rng(seed))
            cases.append({"function": name, "x0": [x0], "seed": seed, "num_samples": 300,
                          "lime": _lime(lime, probes_1d), "arctanh": _arctanh(at, probes_1d)})
    out["one_d"] = {"probes": probes_1d, "bounds": bounds_1d, "cases": cases}

    sine = one_d_black_box("sine")
    variants = {
        "link": {"fit": "link"},
        "pin": {"pin_origin": True},
        "walk+samples": {"asymptotes": "walk+samples"},
        "margin0": {"margin": 0.0},
        "eps0.3": {"epsilon": 0.3, "max_steps": 15},
    }
    out["variants"] = {}
    for key, over in variants.items():
        cfg = ArcTanhConfig(num_samples=250, **over)
        res = ArcTanhLime([1.0], cfg, bounds_1d).explain(sine, [0.7], Rng(5))
        out["variants"][key] = {"config": over, "result": _arctanh(res, probes_1d)}

    ens = DeltaAnchorEnsemble(ArcTanhLime([1.0], ArcTanhConfig(num_samples=200), bounds_1d),
                              DeltaConfig(delta=1.0), bounds_1d).explain(sine, [0.7], Rng(5))
    lens = DeltaAnchorEnsemble(LimeTabular([1.0], LimeConfig(num_samples=200)), DeltaConfig(delta=1.5, rings=2),
                               bounds_1d).explain(sine, [0.7], Rng(6))
    out["ensembles"] = {
        "arctanh": {"anchors": ens.anchors, "n_queries": ens.n_queries,
                    "predictions": [ens.surrogate.predict(p) for p in probes_1d],
                    "blend_gradient": [ens.surrogate.gradient(p) for p in ([0.2], [1.3], [2.9])]},
        "lime_rings2": {"anchors": lens.anchors, "n_queries": lens.n_queries,
                        "predictions": [lens.surrogate.predict(p) for p in probes_1d]},
    }

    Xtr, _, Xte, _ = _data_moons()
    net = MLP.from_dict(json.loads((MODELS / "moons_mlp.json").read_text()))
    scale, bounds = training_scale(Xtr), training_bounds(Xtr)
    probes_2d = [[-1.5 + 0.35 * i, -1.0 + 0.2 * j] for i in range(12) for j in range(13)]
    moons_cases = []
    for i, x0 in enumerate(Xte[:3]):
        lime = LimeTabular(scale, LimeConfig(num_samples=400)).explain(net, x0, Rng(100 + i))
        at = ArcTanhLime(scale, ArcTanhConfig(num_samples=400), bounds).explain(net, x0, Rng(100 + i))
        moons_cases.append({"x0": x0, "seed": 100 + i, "num_samples": 400, "lime": _lime(lime, probes_2d),
                            "arctanh": _arctanh(at, probes_2d)})
    out["moons"] = {"scale": scale, "bounds": bounds, "probes": probes_2d,
                    "mlp_outputs": [net(p) for p in probes_2d], "cases": moons_cases}
    return out


def moons_payload() -> dict:
    Xtr, ytr, Xte, _ = _data_moons()
    return {
        "model": json.loads((MODELS / "moons_mlp.json").read_text()),
        "scale": training_scale(Xtr),
        "bounds": training_bounds(Xtr),
        "train": [[round(x[0], 4), round(x[1], 4), int(t)] for x, t in zip(Xtr[:300], ytr[:300])],
        "test_instances": Xte[:60],
    }


def results_payload() -> dict:
    payload = {}
    for name in ("benchmarks", "ablations", "stability"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            payload[name] = json.loads(path.read_text())
    quality = MODELS / "quality.json"
    if quality.exists():
        payload["quality"] = json.loads(quality.read_text())
    return payload


def _r(v: float, digits: int = 5) -> float:
    return float(f"{v:.{digits}g}")


def lab_presets() -> dict:
    """Datasets + black boxes for the 3-D lab (fetched lazily by the page from public/lab/)."""
    from .common import _data_synth
    from .diabetes import ATTRIBUTION, MODEL_TITLES, TITLES, load, model

    def pack(key, title, attribution, names, Xtr, rows, labels, positive, models):
        n = len(Xtr)
        mean = [sum(r[j] for r in Xtr) / n for j in range(len(names))]
        return {"key": key, "title": title, "attribution": attribution, "features": names,
                "mean": [_r(v, 8) for v in mean], "scale": [_r(v, 8) for v in training_scale(Xtr)],
                "bounds": [[_r(lo, 8), _r(hi, 8)] for lo, hi in training_bounds(Xtr)],
                "rows": [[_r(v) for v in r] for r in rows], "labels": [int(v) for v in labels], "positive": positive,
                "models": {k: m.to_dict() for k, m in models.items()},
                "model_titles": {k: MODEL_TITLES.get(k, k) for k in models}}

    out = []
    positives = {"pima": "diabetes", "early": "positive", "brfss": "prediabetes or diabetes",
                 "efron": "high progression"}
    for key in ("pima", "early", "brfss", "efron"):
        d = load(key)
        rows = (d.X_test + d.X_train)[:500]
        labels = (d.y_test + d.y_train)[:500]
        out.append(pack(key, TITLES[key], ATTRIBUTION[key], d.names, d.X_train, rows, labels, positives[key],
                        {m: model(key, m) for m in ("mlp", "gbm", "lr")}))
    Xtr, ytr, Xte, yte = _data_moons()
    out.append(pack("moons", "Two moons (synthetic)", "Synthetic two-moons data generated by this project.",
                    ["x1", "x2"], Xtr, Xte[:300], yte[:300], "class 1",
                    {"mlp": MLP.from_dict(json.loads((MODELS / "moons_mlp.json").read_text()))}))
    Xtr, ytr, Xte, yte = _data_synth()
    out.append(pack("synth", "Synthetic 8-D (known ground truth)", "Synthetic data generated by this project.",
                    [f"x{j + 1}" for j in range(8)], Xtr, Xte[:400], yte[:400], "class 1",
                    {"mlp": MLP.from_dict(json.loads((MODELS / "synth_mlp.json").read_text()))}))
    return {"datasets": out}


def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    lab = ROOT / "web" / "public" / "lab"
    lab.mkdir(parents=True, exist_ok=True)
    (lab / "presets.json").write_text(json.dumps(lab_presets(), separators=(",", ":")))
    print(f"wrote web/public/lab/presets.json ({(lab / 'presets.json').stat().st_size / 1024:.0f} KiB)")
    diabetes = RESULTS / "diabetes" / "benchmarks.json"
    if diabetes.exists():
        (WEB_DATA / "diabetes.json").write_text(diabetes.read_text())
    (WEB_DATA / "parity.json").write_text(json.dumps(parity_fixtures()))
    (WEB_DATA / "moons.json").write_text(json.dumps(moons_payload()))
    (WEB_DATA / "results.json").write_text(json.dumps(results_payload()))
    for f in ("parity.json", "moons.json", "results.json", "diabetes.json"):
        print(f"wrote web/data/{f} ({(WEB_DATA / f).stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
