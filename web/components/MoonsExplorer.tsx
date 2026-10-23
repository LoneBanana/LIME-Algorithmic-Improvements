"use client";

import { createContext, useContext, useEffect, useId, useMemo, useRef, useState } from "react";
import moons from "@/data/moons.json";
import { ArcTanhLime, DeltaAnchorEnsemble, LimeTabular, mlpPredictor, Rng } from "@/lib/core/index.ts";
import type { Bounds, MlpWeights, Surrogate } from "@/lib/core/index.ts";
import { useElementWidth } from "@/lib/hooks";
import { clamp, fmtNum, fmtTick, mixHex, niceTicks, scaleLinear, svgPoint } from "@/lib/plot";
import { useCssColors } from "@/lib/theme";

const net = mlpPredictor(moons.model as unknown as MlpWeights);
const SCALE = moons.scale as number[];
const BOUNDS = moons.bounds as unknown as Bounds;
const PAD = 0.18;
const XD: [number, number] = [BOUNDS![0][0] - PAD, BOUNDS![0][1] + PAD];
const YD: [number, number] = [BOUNDS![1][0] - PAD, BOUNDS![1][1] + PAD];
const NX = 132;
const NY = Math.round((NX * (YD[1] - YD[0])) / (XD[1] - XD[0]));
const CX = Array.from({ length: NX }, (_, i) => XD[0] + ((i + 0.5) * (XD[1] - XD[0])) / NX);
const CY = Array.from({ length: NY }, (_, j) => YD[0] + ((j + 0.5) * (YD[1] - YD[0])) / NY);

function gridOf(fn: (x: number[]) => number): Float64Array {
  const out = new Float64Array(NX * NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) out[j * NX + i] = fn([CX[i], CY[j]]);
  return out;
}

const PROB = gridOf(net);

/** Marching squares on the cell-centre grid; returns segments in data coordinates. */
function contour(v: Float64Array, level: number): number[][] {
  const segs: number[][] = [];
  const at = (i: number, j: number) => v[j * NX + i];
  const px = (gi: number) => XD[0] + ((gi + 0.5) * (XD[1] - XD[0])) / NX;
  const py = (gj: number) => YD[0] + ((gj + 0.5) * (YD[1] - YD[0])) / NY;
  for (let j = 0; j < NY - 1; j++) {
    for (let i = 0; i < NX - 1; i++) {
      const a = at(i, j);
      const b = at(i + 1, j);
      const c = at(i + 1, j + 1);
      const d = at(i, j + 1);
      const idx = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const bottom = [px(i + (level - a) / (b - a)), py(j)];
      const right = [px(i + 1), py(j + (level - b) / (c - b))];
      const top = [px(i + (level - d) / (c - d)), py(j + 1)];
      const left = [px(i), py(j + (level - a) / (d - a))];
      const add = (p: number[], q: number[]) => segs.push([p[0], p[1], q[0], q[1]]);
      switch (idx) {
        case 1: case 14: add(left, bottom); break;
        case 2: case 13: add(bottom, right); break;
        case 3: case 12: add(left, right); break;
        case 4: case 11: add(right, top); break;
        case 6: case 9: add(bottom, top); break;
        case 7: case 8: add(left, top); break;
        case 5: add(left, top); add(bottom, right); break;
        case 10: add(left, bottom); add(right, top); break;
      }
    }
  }
  return segs;
}

const BOUNDARY = contour(PROB, 0.5);

function paint(canvas: HTMLCanvasElement | null, values: Float64Array, color: (v: number) => [number, number, number]) {
  if (!canvas) return;
  canvas.width = NX;
  canvas.height = NY;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(NX, NY);
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const [r, g, b] = color(values[j * NX + i]);
      const k = ((NY - 1 - j) * NX + i) * 4;
      img.data[k] = r;
      img.data[k + 1] = g;
      img.data[k + 2] = b;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Found by scanning the trained MLP: two points with P(class 1) = 0.50 and two deep inside the moons.
const PRESETS: { label: string; x: number[] }[] = [
  { label: "on the boundary", x: [-0.27, 0.55] },
  { label: "inner bend", x: [1.02, -0.1] },
  { label: "upper moon", x: [-0.6, 0.75] },
  { label: "lower moon", x: [1.4, -0.35] },
];

type MapKey = "lime" | "arctanh" | "arctanhDelta";
const MAPS: { key: MapKey; label: string; keyClass: string }[] = [
  { key: "lime", label: "LIME", keyClass: "key c-lime" },
  { key: "arctanh", label: "ArcTanh-LIME", keyClass: "key c-atl" },
  { key: "arctanhDelta", label: "ArcTanh-LIME-δ", keyClass: "key dashed c-atl" },
];

const M = { l: 36, r: 10, t: 10, b: 26 };

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const PanelContext = createContext<{ x: (v: number) => number; y: (v: number) => number } | null>(null);

function usePanel() {
  const ctx = useContext(PanelContext);
  if (!ctx) throw new Error("usePanel outside Panel");
  return ctx;
}

function Panel(props: {
  width: number; children?: React.ReactNode; canvasRef: React.RefObject<HTMLCanvasElement | null>;
  onPick?: (x: number[]) => void; ticks?: boolean; label: string;
}) {
  const { width, children, canvasRef, onPick, ticks = true, label } = props;
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();
  const m = ticks ? M : { l: 4, r: 4, t: 4, b: 4 };
  const plotW = width - m.l - m.r;
  const plotH = (plotW * (YD[1] - YD[0])) / (XD[1] - XD[0]);
  const H = plotH + m.t + m.b;
  const x = scaleLinear(XD, [m.l, m.l + plotW]);
  const y = scaleLinear(YD, [m.t + plotH, m.t]);
  return (
    <div className="heat-wrap" style={{ height: H }}>
      <canvas ref={canvasRef} style={{ left: m.l, top: m.t, width: plotW, height: plotH }} aria-hidden="true" />
      <svg ref={svgRef} className="chart" width={width} height={H} viewBox={`0 0 ${width} ${H}`}
        style={{ position: "absolute", left: 0, top: 0 }} role="img" aria-label={label}
        onPointerDown={(e) => {
          if (!onPick || !svgRef.current) return;
          const p = svgPoint(svgRef.current, e.clientX, e.clientY);
          const xv = clamp(x.invert(p.x), BOUNDS![0][0], BOUNDS![0][1]);
          const yv = clamp(y.invert(p.y), BOUNDS![1][0], BOUNDS![1][1]);
          onPick([+xv.toFixed(3), +yv.toFixed(3)]);
        }}>
        {ticks && (
          <g>
            {niceTicks(XD[0], XD[1], 5).map((t) => (
              <text key={`x${t}`} x={x(t)} y={m.t + plotH + 16} textAnchor="middle">{fmtTick(t)}</text>
            ))}
            {niceTicks(YD[0], YD[1], 4).map((t) => (
              <text key={`y${t}`} x={m.l - 6} y={y(t) + 3.5} textAnchor="end">{fmtTick(t)}</text>
            ))}
          </g>
        )}
        <defs>
          <clipPath id={clipId}>
            <rect x={m.l} y={m.t} width={plotW} height={plotH} />
          </clipPath>
        </defs>
        <rect x={m.l} y={m.t} width={plotW} height={plotH} fill="none" stroke="var(--border)" />
        <g className={onPick ? "drag-surface xy" : undefined} clipPath={`url(#${clipId})`}>
          <rect x={m.l} y={m.t} width={plotW} height={plotH} fill="transparent" />
          {children && <PanelContext.Provider value={{ x, y }}>{children}</PanelContext.Provider>}
        </g>
      </svg>
    </div>
  );
}

function Boundary() {
  const { x, y } = usePanel();
  const d = BOUNDARY.map((s) => `M${x(s[0]).toFixed(1)},${y(s[1]).toFixed(1)}L${x(s[2]).toFixed(1)},${y(s[3]).toFixed(1)}`).join("");
  return <path d={d} stroke="var(--ink)" strokeWidth={1.4} opacity={0.75} fill="none" />;
}

function Instance({ x0, anchors }: { x0: number[]; anchors?: number[][] }) {
  const { x, y } = usePanel();
  return (
    <g pointerEvents="none">
      {anchors?.slice(1).map((a, i) => (
        <path key={i} d={`M${x(a[0])},${y(a[1]) - 6}l5,6l-5,6l-5,-6z`} fill="var(--surface)" stroke="var(--ink)"
          strokeWidth={1.4} />
      ))}
      <circle cx={x(x0[0])} cy={y(x0[1])} r={6.5} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2.5} />
    </g>
  );
}

function Ellipse({ x0, k }: { x0: number[]; k: number }) {
  const { x, y } = usePanel();
  const rx = Math.abs(x(x0[0] + k * SCALE[0]) - x(x0[0]));
  const ry = Math.abs(y(x0[1] + k * SCALE[1]) - y(x0[1]));
  return (
    <ellipse cx={x(x0[0])} cy={y(x0[1])} rx={rx} ry={ry} fill="none" stroke="var(--ink)" strokeWidth={1}
      strokeDasharray="3 3" opacity={0.6} pointerEvents="none" />
  );
}

function Walk({ points, kind }: { points: number[][]; kind: "up" | "down" }) {
  const { x, y } = usePanel();
  if (points.length < 2) return null;
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  const end = points[points.length - 1];
  return (
    <g pointerEvents="none">
      <path d={d} stroke="var(--atl)" strokeWidth={2.2} fill="none" strokeLinejoin="round" />
      <circle cx={x(end[0])} cy={y(end[1])} r={5} fill={kind === "up" ? "var(--atl)" : "var(--surface)"}
        stroke={kind === "up" ? "var(--surface)" : "var(--atl)"} strokeWidth={2} />
      <text x={x(end[0]) + 8} y={y(end[1]) + 4} className="label-strong halo">{kind === "up" ? "max" : "min"}</text>
    </g>
  );
}

function TrainingPoints() {
  const { x, y } = usePanel();
  return (
    <g pointerEvents="none" opacity={0.7}>
      {/* two-tone marks stay visible on both ends of the heat map, in either theme */}
      {(moons.train as number[][]).map((p, i) =>
        p[2] === 1 ? (
          <circle key={i} cx={x(p[0])} cy={y(p[1])} r={2.3} fill="#111" stroke="#fff" strokeWidth={0.8} />
        ) : (
          <circle key={i} cx={x(p[0])} cy={y(p[1])} r={2.3} fill="#fff" stroke="#111" strokeWidth={1} />
        ),
      )}
    </g>
  );
}

function windowStats(s: Surrogate, x0: number[], k: number): number {
  let sum = 0;
  let n = 0;
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const zx = (CX[i] - x0[0]) / SCALE[0];
      const zy = (CY[j] - x0[1]) / SCALE[1];
      if (zx * zx + zy * zy > k * k) continue;
      sum += (s.predict([CX[i], CY[j]]) - PROB[j * NX + i]) ** 2;
      n++;
    }
  }
  return n ? sum / n : NaN;
}

export default function MoonsExplorer() {
  const [x0, setX0] = useState<number[]>(PRESETS[0].x);
  const [delta, setDelta] = useState(1);
  const [seed, setSeed] = useState(1);
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(560);
  const [mapsRef, mapsWidth] = useElementWidth<HTMLDivElement>(760);
  const mainCanvas = useRef<HTMLCanvasElement>(null);
  const mapCanvases = { lime: useRef<HTMLCanvasElement>(null), arctanh: useRef<HTMLCanvasElement>(null),
    arctanhDelta: useRef<HTMLCanvasElement>(null) };
  const colors = useCssColors(["seq-lo", "seq-mid", "seq-hi", "err-lo", "err-hi"] as const);

  const run = useMemo(() => {
    const cfg = { numSamples: 1000 };
    const lime = new LimeTabular(SCALE, cfg).explain(net, x0, new Rng(seed));
    const arctanh = new ArcTanhLime(SCALE, cfg, BOUNDS).explain(net, x0, new Rng(seed));
    const arctanhDelta = new DeltaAnchorEnsemble(new ArcTanhLime(SCALE, cfg, BOUNDS), { delta }, BOUNDS)
      .explain(net, x0, new Rng(seed));
    const surrogates: Record<MapKey, Surrogate> = {
      lime: lime.surrogate, arctanh: arctanh.surrogate, arctanhDelta: arctanhDelta.surrogate,
    };
    const errors = {} as Record<MapKey, Float64Array>;
    let all: number[] = [];
    for (const k of Object.keys(surrogates) as MapKey[]) {
      const s = surrogates[k];
      const e = gridOf((p) => s.predict(p));
      for (let i = 0; i < e.length; i++) e[i] = Math.abs(e[i] - PROB[i]);
      errors[k] = e;
      all = all.concat(Array.from(e.filter((_, i) => i % 7 === 0)));
    }
    all.sort((a, b) => a - b);
    const cap = Math.max(all[Math.floor(0.97 * (all.length - 1))], 1e-3);
    const stats = Object.fromEntries((Object.keys(surrogates) as MapKey[]).map((k) => [k, {
      mse1: windowStats(surrogates[k], x0, 1), mse2: windowStats(surrogates[k], x0, 2),
    }])) as Record<MapKey, { mse1: number; mse2: number }>;
    const queries: Record<MapKey, number> = { lime: lime.nQueries, arctanh: arctanh.nQueries,
      arctanhDelta: arctanhDelta.nQueries };
    return { lime, arctanh, arctanhDelta, errors, cap, stats, queries };
  }, [x0, delta, seed]);

  useEffect(() => {
    if (!colors) return;
    const [lo, mid, hi] = [colors["seq-lo"], colors["seq-mid"], colors["seq-hi"]];
    paint(mainCanvas.current, PROB, (v) => (v < 0.5 ? mixHex(lo, mid, v / 0.5) : mixHex(mid, hi, (v - 0.5) / 0.5)));
  }, [colors, width]);

  useEffect(() => {
    if (!colors) return;
    for (const m of MAPS) {
      paint(mapCanvases[m.key].current, run.errors[m.key],
        (v) => mixHex(colors["err-lo"], colors["err-hi"], Math.min(1, v / run.cap)));
    }
  }, [colors, run, mapsWidth]);

  const at = run.arctanh;
  const mapWidth = mapsWidth >= 600 ? Math.floor((mapsWidth - 24) / 3) : mapsWidth;
  const f0 = net(x0);

  return (
    <figure className="card" aria-label="Two-moons explorer">
      <div className="controls">
        <div className="control" style={{ minWidth: 0 }}>
          <span>Instance</span>
          <div className="seg" role="group" aria-label="Preset instances">
            {PRESETS.map((p) => (
              <button key={p.label} type="button" aria-pressed={p.x === x0} onClick={() => setX0(p.x)}>{p.label}</button>
            ))}
            <button type="button" aria-pressed={false} onClick={() => {
              const pts = moons.test_instances as number[][];
              setX0(pts[Math.floor(Math.random() * pts.length)]);
            }}>random test point</button>
          </div>
        </div>
        <label className="control">
          <span>δ, anchor spacing (σ) <span className="value">{delta}</span></span>
          <input type="range" min={0.25} max={2} step={0.05} value={delta} onChange={(e) => setDelta(Number(e.target.value))} />
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>Resample</button>
      </div>
      <div className="grid-main-side">
        <div ref={wrapRef}>
          <p className="fig-title">MLP probability P(class 1) — click to choose x₀</p>
          <Panel width={width} canvasRef={mainCanvas} onPick={setX0}
            label="Two-moons MLP probability heat map with the epsilon-walk from the chosen instance">
            <TrainingPoints />
            <Boundary />
            {at.walkUp && <Walk points={at.walkUp.points} kind="up" />}
            {at.walkDown && <Walk points={at.walkDown.points} kind="down" />}
            <Instance x0={x0} anchors={run.arctanhDelta.anchors} />
          </Panel>
          <div className="legend" style={{ marginTop: 8 }}>
            <span className="colorbar">
              P = 0
              <span className="bar" style={{ background: "linear-gradient(90deg, var(--seq-lo), var(--seq-mid), var(--seq-hi))" }} />
              P = 1
            </span>
            <span><span className="key c-ink" /> P = 0.5 boundary</span>
            <span><span className="key c-atl" /> ε-walk</span>
            <span>
              <span className="key dot" style={{ background: "#111", boxShadow: "0 0 0 1px #fff" }} /> class 1 ·{" "}
              <span className="key dot" style={{ background: "#fff", boxShadow: "0 0 0 1px #111" }} /> class 0
            </span>
          </div>
        </div>
        <dl className="readout">
          <dt>x₀</dt>
          <dd>({fmtNum(x0[0], 3)}, {fmtNum(x0[1], 3)})</dd>
          <dt>f(x₀) = P(class 1)</dt>
          <dd>{fmtNum(f0, 4)}</dd>
          <dt>walk up ({plural(at.walkUp ? at.walkUp.points.length - 1 : 0, "step")})</dt>
          <dd>→ {fmtNum(at.rawHi, 4)}</dd>
          <dt>walk down ({plural(at.walkDown ? at.walkDown.points.length - 1 : 0, "step")})</dt>
          <dd>→ {fmtNum(at.rawLo, 4)}</dd>
          <dt>LIME attribution</dt>
          <dd>({fmtNum(run.lime.attribution[0], 3)}, {fmtNum(run.lime.attribution[1], 3)})</dd>
          <dt>ArcTanh attribution</dt>
          <dd>({fmtNum(at.attribution[0], 3)}, {fmtNum(at.attribution[1], 3)})</dd>
          <dt>anchors (axes layout)</dt>
          <dd>{run.arctanhDelta.anchors.length}</dd>
          <dt>queries LIME · ArcTanh · δ</dt>
          <dd>{run.queries.lime} · {run.queries.arctanh} · {run.queries.arctanhDelta.toLocaleString("en-US")}</dd>
        </dl>
      </div>

      <div ref={mapsRef} style={{ marginTop: 22 }}>
        <p className="fig-title">Where each surrogate is wrong: |g(x) − f(x)|</p>
        <p className="fig-sub">
          Same colour scale in all three maps (bar below; saturates at {fmtNum(run.cap, 2)}). Dashed ellipses: 1σ
          and 2σ around x₀ in standardised units. Click a map to move x₀.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: mapsWidth >= 600 ? "repeat(3, minmax(0, 1fr))" : "1fr", gap: 12 }}>
          {MAPS.map((m) => (
            <div key={m.key}>
              <div className="legend" style={{ margin: "0 0 4px" }}>
                <span><span className={m.keyClass} /> {m.label}</span>
              </div>
              <Panel width={mapWidth} canvasRef={mapCanvases[m.key]} ticks={false} onPick={setX0}
                label={`Error map of ${m.label}`}>
                <Ellipse x0={x0} k={1} />
                <Ellipse x0={x0} k={2} />
                <Instance x0={x0} anchors={m.key === "arctanhDelta" ? run.arctanhDelta.anchors : undefined} />
              </Panel>
              <p className="small" style={{ margin: "6px 0 0", color: "var(--ink-2)" }}>
                MSE within 1σ <b>{fmtNum(run.stats[m.key].mse1, 3)}</b> · 2σ <b>{fmtNum(run.stats[m.key].mse2, 3)}</b>
                {" "}· {run.queries[m.key].toLocaleString("en-US")} queries
              </p>
            </div>
          ))}
        </div>
        <div className="colorbar" style={{ marginTop: 10 }}>
          0
          <span className="bar" style={{ background: "linear-gradient(90deg, var(--err-lo), var(--err-hi))", border: "1px solid var(--border)" }} />
          ≥ {fmtNum(run.cap, 2)}
        </div>
      </div>
    </figure>
  );
}
