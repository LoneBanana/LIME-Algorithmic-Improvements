"""Aggregation of per-instance results into summary statistics and Markdown tables."""

from __future__ import annotations

import math

from atlime import Rng
from atlime.metrics import bootstrap_mean_diff_ci, mean, median, quantile, wilcoxon_signed_rank, win_rate

from .common import RADII


def _metric_vectors(records: list[dict], label: str) -> dict[str, list]:
    out: dict[str, list] = {f"r{r:g}": [] for r in RADII}
    out.update({"neighborhood_wmse": [], "global_mse": [], "local_abs_error": [], "queries": [], "seconds": [],
                "grad_cosine": []})
    for rec in records:
        m = rec["methods"][label]
        for r, v in zip(RADII, m["sphere_mse"]):
            out[f"r{r:g}"].append(v)
        for k in ("neighborhood_wmse", "global_mse", "local_abs_error", "queries", "seconds"):
            out[k].append(m[k])
        if m.get("grad_cosine") is not None:
            out["grad_cosine"].append(m["grad_cosine"])
    return out


def describe(v: list[float]) -> dict:
    if not v:
        return {}
    return {"mean": mean(v), "median": median(v), "q25": quantile(v, 0.25), "q75": quantile(v, 0.75), "n": len(v)}


def paired(a: list[float], b: list[float], seed: int = 0) -> dict:
    """Compare method A against B on paired instances (lower is better)."""
    lo, hi = bootstrap_mean_diff_ci(a, b, Rng(seed), n_boot=2000)
    w = wilcoxon_signed_rank(a, b)
    ratio = [x / y for x, y in zip(a, b) if y > 0]
    return {"mean_diff": mean([x - y for x, y in zip(a, b)]), "ci95": [lo, hi], "win_rate": win_rate(a, b),
            "wilcoxon_p": w["p"], "wilcoxon_z": w["z"], "median_ratio": median(ratio) if ratio else None,
            "mean_ratio": mean(a) / mean(b) if mean(b) > 0 else None}


def summarize(records: list[dict], comparisons: list[tuple[str, str]]) -> dict:
    labels = list(records[0]["methods"].keys())
    vecs = {lab: _metric_vectors(records, lab) for lab in labels}
    summary = {"n_instances": len(records), "methods": {}, "comparisons": {}}
    for lab in labels:
        summary["methods"][lab] = {k: describe(v) for k, v in vecs[lab].items()}
    for i, (a, b) in enumerate(comparisons):
        if a not in vecs or b not in vecs:
            continue
        comp = {}
        for k in [f"r{r:g}" for r in RADII] + ["neighborhood_wmse", "global_mse", "local_abs_error"]:
            comp[k] = paired(vecs[a][k], vecs[b][k], seed=i)
        summary["comparisons"][f"{a} vs {b}"] = comp
    # diagnostics of the walk
    diag = {"up_steps": [], "down_steps": [], "reasons": {}, "fallbacks": {}, "clipped": []}
    for rec in records:
        for lab, m in rec["methods"].items():
            if "walk_up_steps" not in m or lab != "ArcTanh-LIME":
                continue
            diag["up_steps"].append(m["walk_up_steps"])
            diag["down_steps"].append(m["walk_down_steps"])
            for reason in (m["walk_up_reason"], m["walk_down_reason"]):
                diag["reasons"][reason] = diag["reasons"].get(reason, 0) + 1
            fb = m["fallback"] or "none"
            diag["fallbacks"][fb] = diag["fallbacks"].get(fb, 0) + 1
            diag["clipped"].append(m["clipped_fraction"])
    if diag["up_steps"]:
        summary["walk"] = {"up_steps": describe(diag["up_steps"]), "down_steps": describe(diag["down_steps"]),
                           "stop_reasons": diag["reasons"], "fallbacks": diag["fallbacks"],
                           "clipped_fraction": describe(diag["clipped"])}
    return summary


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) < 10 ** (-digits):
        return f"{v:.1e}"
    return f"{v:.{digits}g}"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def method_table(summary: dict, radii_cols: list[float]) -> str:
    cols = [f"r={r:g}" for r in radii_cols]
    head = "| Method | Queries (median) | " + " | ".join(f"MSE {c}" for c in cols) + \
           " | Neighbourhood wMSE | Global MSE | |g(x0)−f(x0)| | cos(attr, ∇f) |"
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for lab, m in summary["methods"].items():
        row = [lab, fmt(m["queries"]["median"], 4)]
        row += [fmt(m[f"r{r:g}"]["mean"]) for r in radii_cols]
        row += [fmt(m["neighborhood_wmse"]["mean"]), fmt(m["global_mse"]["mean"]),
                fmt(m["local_abs_error"]["mean"])]
        row.append(fmt(m["grad_cosine"]["mean"]) if m.get("grad_cosine") else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def comparison_table(summary: dict, radii_cols: list[float]) -> str:
    cols = [f"r{r:g}" for r in radii_cols] + ["neighborhood_wmse", "global_mse"]
    names = [f"r={r:g}" for r in radii_cols] + ["neighbourhood", "global"]
    head = "| Comparison (A vs B) | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for comp_name, comp in summary["comparisons"].items():
        cells = []
        for k in cols:
            c = comp[k]
            ratio = c["mean_ratio"]
            cells.append(f"{fmt(ratio, 2)}× · {c['win_rate'] * 100:.0f}% · p={pval(c['wilcoxon_p'])}")
        lines.append(f"| {comp_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
