"use client";

import { useEffect, useMemo, useState } from "react";
import { runOneD } from "@/lib/explain1d";
import type { OneDSettings } from "@/lib/explain1d";
import { fmtNum } from "@/lib/plot";
import { usePrefersReducedMotion } from "@/lib/theme";
import LinkSpaceChart from "./LinkSpaceChart";
import OneDChart, { ALL_LAYERS } from "./OneDChart";
import type { Layers } from "./OneDChart";

const NONE: Layers = Object.fromEntries(Object.keys(ALL_LAYERS).map((k) => [k, false])) as unknown as Layers;

const STEP_LAYERS: Layers[] = [
  { ...NONE, density: true, samples: true, lime: true },
  { ...NONE, density: true, samples: true, walk: true },
  { ...NONE, walk: true, asymptotes: true, arctanh: true, lime: true },
  { ...NONE, anchors: true, arctanh: true, arctanhDelta: true },
];

const START: OneDSettings = { fn: "bumps", x0: -2.2, epsilon: 0.1, delta: 1, numSamples: 1000, kernelWidth: 0.75, seed: 3 };

export default function MethodSteps() {
  const [step, setStep] = useState(0);
  const [settings, setSettings] = useState<OneDSettings>(START);
  const [reveal, setReveal] = useState(1);
  const [replay, setReplay] = useState(0);
  const reduced = usePrefersReducedMotion();
  const run = useMemo(() => runOneD(settings), [settings]);
  const at = run.arctanh;

  useEffect(() => {
    if (step !== 1 || reduced) {
      setReveal(1);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / 1600);
      setReveal(p);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    setReveal(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step, reduced, replay]);

  const upSteps = at.walkUp ? at.walkUp.points.length - 1 : 0;
  const downSteps = at.walkDown ? at.walkDown.points.length - 1 : 0;
  const tanh = "beta0" in at.surrogate ? at.surrogate : null;

  const steps = [
    {
      title: "Sample",
      body: (
        <p>
          Classic LIME perturbs x₀ = {fmtNum(settings.x0, 3)} with Gaussian noise (one training standard deviation),
          queries the black box at all {settings.numSamples.toLocaleString("en-US")} points, weights them with an
          exponential kernel and fits a weighted ridge line (blue). The line is faithful only very close to x₀: the
          black box bends away on both sides.
        </p>
      ),
    },
    {
      title: "Walk up & down",
      body: (
        <p>
          From x₀, take steps of fixed width ε = {settings.epsilon}σ along the steepest ascent until the next step
          stops climbing — the <b>nearest local maximum</b> ({upSteps} {upSteps === 1 ? "step" : "steps"}, f = {fmtNum(at.rawHi, 4)}) — and
          likewise downhill to the <b>nearest local minimum</b> ({downSteps} {downSteps === 1 ? "step" : "steps"}, f = {fmtNum(at.rawLo, 4)}).
          Directions come from finite differences, so the black box is only ever queried.{" "}
          <button type="button" className="btn" onClick={() => setReplay((r) => r + 1)} style={{ padding: "2px 9px" }}>
            Replay
          </button>
        </p>
      ),
    },
    {
      title: "Asymptotes → tanh⁻¹",
      body: (
        <p>
          The two extreme values become the asymptotes of tanh⁻¹: every output is mapped to u = (f − c)/a ∈ (−1, 1)
          with c ± a = [{fmtNum(at.lo, 3)}, {fmtNum(at.hi, 3)}] (5% margin), and artanh(u) is regressed linearly on x.
          In this <i>link space</i> (right) the S-shaped segment becomes a line; mapping back gives the orange curve
          g(x) = c + a·tanh(β₀ + β·(x − x₀)){tanh ? `, here β₀ = ${fmtNum(tanh.beta0, 3)}, β = ${fmtNum(tanh.beta[0], 3)}` : ""}.
        </p>
      ),
    },
    {
      title: "Repeat at ±δ, blend",
      body: (
        <p>
          The same walk-and-fit is run at the anchors x₀ ± δ (δ = {settings.delta}σ; in d dimensions ±δ along every
          feature axis). Their surrogates (thin) are blended with Gaussian weights centred on the anchors (bandwidth
          δ/2) into one <b>partially-global</b> surrogate (dashed), which follows the black box over a wider range —
          at {run.arctanhDelta.nQueries.toLocaleString("en-US")} queries instead of {at.nQueries.toLocaleString("en-US")}.
        </p>
      ),
    },
  ];

  return (
    <figure className="card" aria-label="The method in four steps">
      <div className="steps" role="tablist" aria-label="Steps of the method">
        {steps.map((s, i) => (
          <button key={s.title} type="button" role="tab" aria-selected={step === i} className="step-tab"
            onClick={() => setStep(i)}>
            <span className="num">STEP {i + 1}</span>
            <b>{s.title}</b>
          </button>
        ))}
      </div>
      <div className="step-body prose">{steps[step].body}</div>
      <div className={step === 2 ? "grid-2" : undefined}>
        <div>
          <OneDChart run={run} layers={STEP_LAYERS[step]} walkReveal={step === 1 ? reveal : 1}
            onX0={(x0) => setSettings((s) => ({ ...s, x0 }))}
            ariaLabel={`Step ${step + 1}: ${steps[step].title}`} />
        </div>
        {step === 2 && (
          <div>
            <LinkSpaceChart run={run} />
          </div>
        )}
      </div>
      <div className="legend" style={{ marginTop: 8 }}>
        <span><span className="key c-ink" /> black box f</span>
        {STEP_LAYERS[step].lime && <span><span className="key c-lime" /> LIME</span>}
        {STEP_LAYERS[step].walk && <span><span className="key dot c-atl" /> ε-walk</span>}
        {STEP_LAYERS[step].asymptotes && <span><span className="key dashed c-atl" /> asymptotes</span>}
        {STEP_LAYERS[step].arctanh && <span><span className="key c-atl" /> ArcTanh-LIME</span>}
        {STEP_LAYERS[step].arctanhDelta && <span><span className="key dashed c-atl" /> ArcTanh-LIME-δ (blend)</span>}
      </div>
      <figcaption>
        Black box: the &ldquo;bumps&rdquo; curve. Drag on the plot to move x₀ and watch every step update.
      </figcaption>
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button type="button" className="btn" disabled={step === 0} onClick={() => setStep((s) => s - 1)}>
          ← Previous
        </button>
        <button type="button" className="btn primary" disabled={step === 3} onClick={() => setStep((s) => s + 1)}>
          Next step →
        </button>
      </div>
    </figure>
  );
}
