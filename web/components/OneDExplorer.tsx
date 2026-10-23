"use client";

import { useDeferredValue, useMemo, useState } from "react";
import type { OneDName } from "@/lib/core/index.ts";
import { DEFAULT_ONE_D, METHOD_META, runOneD, windowMse } from "@/lib/explain1d";
import type { MethodKey, OneDSettings } from "@/lib/explain1d";
import { fmtNum } from "@/lib/plot";
import ErrorChart from "./ErrorChart";
import LinkSpaceChart from "./LinkSpaceChart";
import OneDChart, { ALL_LAYERS } from "./OneDChart";
import type { Layers } from "./OneDChart";

const FUNCTIONS: { key: OneDName; label: string }[] = [
  { key: "logistic", label: "logistic" },
  { key: "sine", label: "sine" },
  { key: "bumps", label: "bumps" },
  { key: "staircase", label: "staircase" },
  { key: "ripple", label: "trend + ripple" },
];

const LAYER_CHIPS: { key: keyof Layers; label: string; keyClass: string; color: string }[] = [
  { key: "samples", label: "LIME samples", keyClass: "key tick", color: "var(--muted)" },
  { key: "lime", label: "LIME", keyClass: "key", color: "var(--lime)" },
  { key: "walk", label: "ε-walk", keyClass: "key dot", color: "var(--atl)" },
  { key: "asymptotes", label: "asymptotes", keyClass: "key dashed", color: "var(--atl)" },
  { key: "arctanh", label: "ArcTanh-LIME", keyClass: "key", color: "var(--atl)" },
  { key: "anchors", label: "±δ anchors", keyClass: "key", color: "var(--ink-2)" },
  { key: "limeDelta", label: "LIME-δ", keyClass: "key dashed", color: "var(--lime)" },
  { key: "arctanhDelta", label: "ArcTanh-LIME-δ", keyClass: "key dashed", color: "var(--atl)" },
];

function Slider(props: {
  label: string; value: number; min: number; max: number; step: number; format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  const { label, value, min, max, step, format, onChange } = props;
  return (
    <label className="control">
      <span>
        {label} <span className="value">{format ? format(value) : value}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export default function OneDExplorer() {
  const [settings, setSettings] = useState<OneDSettings>(DEFAULT_ONE_D);
  const [layers, setLayers] = useState<Layers>({ ...ALL_LAYERS, anchors: false, limeDelta: false });
  const deferred = useDeferredValue(settings);
  const run = useMemo(() => runOneD(deferred), [deferred]);
  const set = (patch: Partial<OneDSettings>) => setSettings((s) => ({ ...s, ...patch }));
  const methods: MethodKey[] = (["lime", "arctanh", "limeDelta", "arctanhDelta"] as MethodKey[]).filter((k) => layers[k]);
  const at = run.arctanh;
  const queries: Record<MethodKey, number> = {
    lime: run.lime.nQueries,
    arctanh: at.nQueries,
    limeDelta: run.limeDelta.nQueries,
    arctanhDelta: run.arctanhDelta.nQueries,
  };
  const tanh = "beta0" in at.surrogate ? at.surrogate : null;

  return (
    <figure className="card" aria-label="Interactive 1-D playground">
      <div className="controls">
        <div className="control" style={{ minWidth: 0 }}>
          <span>Black box f(x)</span>
          <div className="seg" role="group" aria-label="Black box">
            {FUNCTIONS.map((fn) => (
              <button key={fn.key} type="button" aria-pressed={settings.fn === fn.key} onClick={() => set({ fn: fn.key })}>
                {fn.label}
              </button>
            ))}
          </div>
        </div>
        <Slider label="ε, walk step (σ)" value={settings.epsilon} min={0.02} max={0.5} step={0.01}
          onChange={(v) => set({ epsilon: v })} />
        <Slider label="δ, anchor spacing (σ)" value={settings.delta} min={0.25} max={3} step={0.05}
          onChange={(v) => set({ delta: v })} />
        <Slider label="kernel width (σ)" value={settings.kernelWidth} min={0.25} max={3} step={0.05}
          onChange={(v) => set({ kernelWidth: v })} />
        <Slider label="samples N" value={settings.numSamples} min={100} max={2000} step={100}
          onChange={(v) => set({ numSamples: v })} />
        <button type="button" className="btn" onClick={() => set({ seed: settings.seed + 1 })}>
          Resample
        </button>
      </div>
      <div className="chips" role="group" aria-label="Layers">
        {LAYER_CHIPS.map((c) => (
          <button key={c.key} type="button" className="chip" aria-pressed={layers[c.key]}
            onClick={() => setLayers((l) => ({ ...l, [c.key]: !l[c.key] }))}>
            <span className={c.keyClass} style={{ color: c.color }} />
            {c.label}
          </button>
        ))}
      </div>
      <div style={{ marginTop: 10 }}>
        <OneDChart run={run} layers={layers} onX0={(x0) => set({ x0 })}
          ariaLabel={`Black box ${settings.fn} with its LIME line and ArcTanh-LIME curve at x0 = ${settings.x0}`} />
      </div>
      <p className="caption" style={{ marginTop: 4 }}>
        <b>Drag anywhere on the plot</b> (or focus x₀ and use the arrow keys) to move the explained instance. Black
        line: the black box. Grey ticks on the axis: LIME&apos;s perturbation sample (Gaussian, 1σ), with its density
        shaded. Everything is recomputed live by the TypeScript port of the Python code.
      </p>
      <div className="grid-main-side" style={{ marginTop: 18 }}>
        <div className="stack">
          <div>
            <p className="fig-title">Where each surrogate stays faithful</p>
            <p className="fig-sub">Absolute error |g(x) − f(x)| across the whole domain.</p>
            <ErrorChart run={run} methods={methods.length ? methods : ["lime", "arctanh"]} />
          </div>
          <div>
            <p className="fig-title">The tanh⁻¹ regressor&apos;s view (link space)</p>
            <p className="fig-sub">
              Outputs mapped through artanh((f − c)/a) with c ± a = the asymptotes; the fitted model is a straight
              line here.
            </p>
            <LinkSpaceChart run={run} />
          </div>
        </div>
        <div className="stack">
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Queries</th>
                  <th>MSE ±1σ</th>
                  <th>MSE ±3σ</th>
                </tr>
              </thead>
              <tbody>
                {(["lime", "arctanh", "limeDelta", "arctanhDelta"] as MethodKey[]).map((k) => (
                  <tr key={k}>
                    <td>
                      <span className={METHOD_META[k].dashed ? "key dashed" : "key"}
                        style={{ color: METHOD_META[k].color, marginRight: 7, verticalAlign: "middle" }} />
                      {METHOD_META[k].label}
                    </td>
                    <td>{queries[k].toLocaleString("en-US")}</td>
                    <td>{fmtNum(windowMse(run, k, 1))}</td>
                    <td>{fmtNum(windowMse(run, k, 3))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <dl className="readout">
            <dt>f(x₀)</dt>
            <dd>{fmtNum(run.g(settings.x0), 4)}</dd>
            <dt>walk up ({at.walkUp ? at.walkUp.points.length - 1 : 0} steps, {at.walkUp?.stopReason})</dt>
            <dd>→ {fmtNum(at.rawHi, 4)}</dd>
            <dt>walk down ({at.walkDown ? at.walkDown.points.length - 1 : 0} steps, {at.walkDown?.stopReason})</dt>
            <dd>→ {fmtNum(at.rawLo, 4)}</dd>
            <dt>asymptotes (5% margin)</dt>
            <dd>
              [{fmtNum(at.lo, 3)}, {fmtNum(at.hi, 3)}]
            </dd>
            {tanh && (
              <>
                <dt>β₀, β</dt>
                <dd>
                  {fmtNum(tanh.beta0, 3)}, {fmtNum(tanh.beta[0], 3)}
                </dd>
              </>
            )}
            <dt>slope at x₀: LIME vs ArcTanh</dt>
            <dd>
              {fmtNum(run.lime.attribution[0], 3)} vs {fmtNum(at.attribution[0], 3)}
            </dd>
            <dt>targets clipped</dt>
            <dd>{(at.clippedFraction * 100).toFixed(1)}%</dd>
            <dt>max steps per walk</dt>
            <dd>{run.maxSteps}</dd>
          </dl>
          {at.fallback && (
            <p className="small muted">Fallback used: {at.fallback} (both walks stalled immediately).</p>
          )}
        </div>
      </div>
    </figure>
  );
}
