"""Real diabetes-risk datasets: download, parsing and preprocessing (standard library only).

    python3 -m experiments.diabetes download   # fetch the four raw files into data/raw/ (git-ignored)
    python3 -m experiments.diabetes summary    # parse them and print shapes / class balance
    python3 -m experiments.diabetes models     # train (or load) the 16 black boxes, print accuracy / AUC

Datasets (all binary "diabetes risk" targets; 1 = positive):

* ``pima``       Pima Indians Diabetes (OpenML 37): 768 women x 8 clinical features.
                 Zeros in glucose, blood pressure, skin thickness, insulin and BMI are
                 physiologically impossible and stand for "missing"; they are replaced by
                 the *training* median of the feature.
* ``early``      Early-stage diabetes risk prediction (UCI 529, Sylhet, Bangladesh):
                 520 patients x 16 features (age + 15 yes/no symptoms). Yes/Male = 1.
* ``brfss``      CDC Diabetes Health Indicators, BRFSS 2015 (UCI 891): 253,680 survey
                 responses x 21 indicators; target "prediabetes or diabetes". A seeded
                 random subsample of 6,000 rows is used (pure-Python models).
* ``efron``      Efron et al. (2004) diabetes data (scikit-learn's ``load_diabetes``):
                 442 patients x 10 baseline variables; the one-year progression score is
                 binarised at its median as "high-progression risk".

Features are used as continuous inputs (LIME's continuous mode), so perturbations of
binary features take non-binary values; the black boxes accept any real input.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from atlime import Rng, training_bounds, training_scale
from atlime.models import MLP, GradientBoosting, RandomForest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"   # trained on the downloaded data, so git-ignored as well
MANIFEST = ROOT / "results" / "diabetes" / "data_manifest.json"

SOURCES = {
    "pima": ("https://api.openml.org/data/v1/download/37", "dataset_37_diabetes.arff"),
    "early": ("https://archive.ics.uci.edu/static/public/529/data.csv", "early_stage_diabetes.csv"),
    "brfss": ("https://archive.ics.uci.edu/static/public/891/data.csv", "cdc_diabetes_health_indicators.csv"),
    "efron": ("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt", "diabetes.tab.txt"),
}

TITLES = {
    "pima": "Pima Indians",
    "early": "Early-stage (Sylhet)",
    "brfss": "CDC BRFSS 2015",
    "efron": "Efron progression",
}

ATTRIBUTION = {
    "pima": "Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases), "
            "via OpenML dataset 37.",
    "early": "Islam, Ferdousi, Rahman & Bushra (2020), Early Stage Diabetes Risk Prediction, UCI Machine Learning "
             "Repository dataset 529, CC BY 4.0.",
    "brfss": "CDC Behavioral Risk Factor Surveillance System 2015, Diabetes Health Indicators, UCI Machine Learning "
             "Repository dataset 891, CC BY 4.0.",
    "efron": "Efron, Hastie, Johnstone & Tibshirani (2004), Least Angle Regression, Annals of Statistics; data file "
             "from www4.stat.ncsu.edu/~boos/var.select.",
}


def download(force: bool = False) -> dict:
    """Fetch the raw files (each into its own directory) and record size + SHA-256.

    If a manifest from an earlier download exists (the committed one describes the files behind
    the published results), any file whose hash differs from it is reported.
    """
    recorded = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest = {}
    for key, (url, name) in SOURCES.items():
        folder = RAW / key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        if force or not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "atlime-research/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
        data = path.read_bytes()
        manifest[key] = {"url": url, "file": f"data/raw/{key}/{name}", "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()}
        print(f"  {key:6s} {len(data):>10,d} bytes  {path.relative_to(ROOT)}")
        old = recorded.get(key, {}).get("sha256")
        if old and old != manifest[key]["sha256"]:
            print(f"  WARNING: {key} differs from the recorded SHA-256 {old[:12]}…; results may not reproduce exactly")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Parsers (each returns feature names, rows, binary labels)
# ---------------------------------------------------------------------------

def _path(key: str) -> Path:
    path = RAW / key / SOURCES[key][1]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python3 -m experiments.diabetes download` first")
    return path


def _parse_pima() -> tuple[list[str], list[list[float]], list[float]]:
    names, rows, y = [], [], []
    in_data = False
    for line in _path("pima").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            names.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            parts = [p.strip().strip("'\"") for p in line.split(",")]
            rows.append([float(v) for v in parts[:-1]])
            y.append(1.0 if parts[-1] == "tested_positive" else 0.0)
    return names[:-1], rows, y


def _read_csv(path: Path, sep: str = ",") -> tuple[list[str], list[list[str]]]:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    header = [h.strip().strip('"') for h in lines[0].split(sep)]
    return header, [[c.strip().strip('"') for c in l.split(sep)] for l in lines[1:]]


def _parse_early() -> tuple[list[str], list[list[float]], list[float]]:
    header, cells = _read_csv(_path("early"))
    target = header.index("class")
    names = [h for i, h in enumerate(header) if i != target]
    code = {"yes": 1.0, "no": 0.0, "male": 1.0, "female": 0.0}
    rows, y = [], []
    for row in cells:
        feats = [float(v) if k.lower() == "age" else code[v.lower()] for k, v in zip(header, row) if k != "class"]
        rows.append(feats)
        y.append(1.0 if row[target].lower() == "positive" else 0.0)
    return names, rows, y


def _parse_brfss() -> tuple[list[str], list[list[float]], list[float]]:
    header, cells = _read_csv(_path("brfss"))
    target = header.index("Diabetes_binary")
    keep = [i for i, h in enumerate(header) if h not in ("ID", "Diabetes_binary")]
    rng = Rng(891)
    order = list(range(len(cells)))
    rng.shuffle(order)
    rows, y = [], []
    for i in order[:6000]:  # seeded subsample: pure-Python models cannot train on 253k rows quickly
        rows.append([float(cells[i][k]) for k in keep])
        y.append(float(cells[i][target]))
    return [header[k] for k in keep], rows, y


def _parse_efron() -> tuple[list[str], list[list[float]], list[float]]:
    header, cells = _read_csv(_path("efron"), sep="\t")
    values = [[float(v) for v in row] for row in cells]
    score = sorted(r[-1] for r in values)
    n = len(score)
    med = score[n // 2] if n % 2 else 0.5 * (score[n // 2 - 1] + score[n // 2])
    return header[:-1], [r[:-1] for r in values], [1.0 if r[-1] > med else 0.0 for r in values]


PARSERS = {"pima": _parse_pima, "early": _parse_early, "brfss": _parse_brfss, "efron": _parse_efron}
PIMA_ZERO_IS_MISSING = ("plas", "pres", "skin", "insu", "mass")


@dataclass
class DiabetesData:
    key: str
    names: list[str]
    X_train: list[list[float]]
    y_train: list[float]
    X_test: list[list[float]]
    y_test: list[float]


_DATA: dict[str, DiabetesData] = {}


def load(key: str) -> DiabetesData:
    """Parsed, shuffled (seeded) and split data: 75/25 for the small sets, 5,000/1,000 for BRFSS."""
    if key in _DATA:
        return _DATA[key]
    names, rows, y = PARSERS[key]()
    order = list(range(len(rows)))
    Rng(1000 + list(SOURCES).index(key)).shuffle(order)
    rows = [rows[i] for i in order]
    y = [y[i] for i in order]
    n_train = 5000 if key == "brfss" else int(round(0.75 * len(rows)))
    Xtr, ytr, Xte, yte = rows[:n_train], y[:n_train], rows[n_train:], y[n_train:]
    if key == "pima":  # impute impossible zeros with training medians (never looks at the test split)
        for name in PIMA_ZERO_IS_MISSING:
            j = names.index(name)
            vals = sorted(r[j] for r in Xtr if r[j] != 0.0)
            med = vals[len(vals) // 2]
            for r in Xtr + Xte:
                if r[j] == 0.0:
                    r[j] = med
    data = DiabetesData(key, names, Xtr, ytr, Xte, yte)
    _DATA[key] = data
    return data


# ---------------------------------------------------------------------------
# Black boxes: logistic regression, MLP, random forest, gradient boosting
# ---------------------------------------------------------------------------

MODELS = ("lr", "mlp", "rf", "gbm")
MODEL_TITLES = {"lr": "logistic regression", "mlp": "MLP", "rf": "random forest", "gbm": "gradient boosting"}
SMOOTH = {"lr": True, "mlp": True, "rf": False, "gbm": False}


def _train(key: str, model: str, d: DiabetesData):
    dim = len(d.names)
    big = key == "brfss"
    if model == "lr":
        net = MLP([dim, 1], "sigmoid", seed=11)
        net.fit(d.X_train, d.y_train, epochs=40 if big else 200, batch_size=32, lr=0.01, seed=12)
        return net
    if model == "mlp":
        net = MLP([dim, 32, 1], "sigmoid", seed=13)
        net.fit(d.X_train, d.y_train, epochs=40 if big else 150, batch_size=32, lr=0.005, weight_decay=1e-4,
                seed=14)
        return net
    if model == "rf":
        return RandomForest(n_trees=40, max_depth=8, min_samples_leaf=5, seed=15).fit(d.X_train, d.y_train)
    if model == "gbm":
        return GradientBoosting(n_trees=100, learning_rate=0.1, max_depth=3, min_samples_leaf=10,
                                seed=16).fit(d.X_train, d.y_train)
    raise KeyError(model)


def _from_dict(data: dict):
    return {"mlp": MLP, "random_forest": RandomForest, "gradient_boosting": GradientBoosting}[data["type"]].from_dict(data)


def auc(scores: list[float], labels: list[float]) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) statistic, ties averaged."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = sum(1 for v in labels if v > 0.5)
    neg = len(labels) - pos
    rank_pos = sum(r for r, v in zip(ranks, labels) if v > 0.5)
    return (rank_pos - pos * (pos + 1) / 2) / (pos * neg)


def model(key: str, name: str):
    path = MODEL_DIR / f"{key}_{name}.json"
    if path.exists():
        return _from_dict(json.loads(path.read_text()))
    d = load(key)
    t0 = time.time()
    m = _train(key, name, d)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m.to_dict()))
    print(f"    trained {key}/{name} in {time.time() - t0:.1f}s")
    return m


def prepare_models() -> dict:
    """Train/load all 16 black boxes and report held-out accuracy and AUC."""
    quality = {}
    for key in SOURCES:
        d = load(key)
        for name in MODELS:
            m = model(key, name)
            p = [m(x) for x in d.X_test]
            acc = sum((q > 0.5) == (t > 0.5) for q, t in zip(p, d.y_test)) / len(p)
            quality[f"{key}_{name}"] = {"test_accuracy": acc, "test_auc": auc(p, d.y_test),
                                        "base_rate": sum(d.y_test) / len(d.y_test)}
    return quality


def make_setting(name: str):
    """Benchmark setting ``dia_<dataset>_<model>`` for experiments.common."""
    from .common import GLOBAL_POINTS, Setting
    _, key, mname = name.split("_")
    d = load(key)
    title = f"{TITLES[key]} · {MODEL_TITLES[mname]}"
    return Setting(name, title, model(key, mname), len(d.names), training_scale(d.X_train),
                   training_bounds(d.X_train), d.X_test[40:40 + GLOBAL_POINTS], d.X_test[:40], SMOOTH[mname],
                   "probability", {"features": d.names})


def setting_names() -> list[str]:
    return [f"dia_{k}_{m}" for k in SOURCES for m in MODELS]


def summary() -> None:
    for key in SOURCES:
        d = load(key)
        pos = sum(d.y_train + d.y_test) / (len(d.y_train) + len(d.y_test))
        print(f"  {key:6s} train {len(d.X_train):5d}  test {len(d.X_test):5d}  features {len(d.names):2d}  "
              f"positive {pos:.3f}  {d.names}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "download":
        download(force="--force" in sys.argv)
    elif cmd == "models":
        for k, v in prepare_models().items():
            print(f"  {k:12s} acc {v['test_accuracy']:.3f}  AUC {v['test_auc']:.3f}  base rate {v['base_rate']:.3f}")
    else:
        summary()
