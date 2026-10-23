"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { datasetFromTable, loadPresets, parseCsv, prepareTable } from "@/lib/lab/data";
import type { LabDataset } from "@/lib/lab/data";
import { boundaryRow, buildScene, explain, localFidelity, principalComponents } from "@/lib/lab/scenes";
import type { AxisMode, BuiltScene, Explanation, ViewMode } from "@/lib/lab/scenes";
import { Viz } from "@/lib/lab/viz";
import { fmtNum } from "@/lib/plot";

const MODES: { key: ViewMode; label: string }[] = [
  { key: "surface", label: "Risk surface (3-D)" },
  { key: "space3d", label: "Feature space (3-D)" },
  { key: "space2d", label: "Feature space (2-D)" },
  { key: "tour", label: "Grand tour (n-D)" },
];

const AXES: { key: AxisMode; label: string }[] = [
  { key: "explanation", label: "explanation direction" },
  { key: "pca", label: "principal components" },
  { key: "features", label: "chosen features" },
];

// manim's smooth rate function (for the live 4th-dimension readout)
const smooth = (t: number) => {
  const s = (x: number) => 1 / (1 + Math.exp(-x));
  const e = s(-5);
  return Math.min(1, Math.max(0, (s(10 * (t - 0.5)) - e) / (1 - 2 * e)));
};

export default function Lab3D() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelRefs = useRef<(HTMLDivElement | null)[]>([]);
  const captionRef = useRef<HTMLDivElement>(null);
  const sweepRef = useRef<HTMLDivElement>(null);
  const vizRef = useRef<Viz | null>(null);
  const sceneRef = useRef<BuiltScene | null>(null);
  const drag = useRef<{ x: number; y: number; moved: number; id: number } | null>(null);
  const pinch = useRef<Map<number, { x: number; y: number }>>(new Map());

  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [presets, setPresets] = useState<LabDataset[]>([]);
  const [uploaded, setUploaded] = useState<LabDataset | null>(null);
  const [dsKey, setDsKey] = useState("pima");
  const [modelKey, setModelKey] = useState("mlp");
  const [mode, setMode] = useState<ViewMode>("surface");
  const [axisMode, setAxisMode] = useState<AxisMode>("explanation");
  const [features, setFeatures] = useState([0, 1, 2]);
  const [rowIndex, setRowIndex] = useState(0);
  const [showData, setShowData] = useState(false);
  const [showDelta, setShowDelta] = useState(false);
  const [levels, setLevels] = useState(false);
  const [sweep, setSweep] = useState(true);
  const [rotate, setRotate] = useState(true);
  const [scene, setScene] = useState<BuiltScene | null>(null);
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<{ name: string; header: string[]; cells: string[][]; target: number } | null>(null);
  const [uploadMsg, setUploadMsg] = useState("");

  const datasets = useMemo(() => (uploaded ? [...presets, uploaded] : presets), [presets, uploaded]);
  const ds = datasets.find((d) => d.key === dsKey) ?? datasets[0];
  const model = ds?.models.find((m) => m.key === modelKey) ?? ds?.models[0];

  // default instance: the row the model is least sure about (closest to P = 0.5)
  const defaultRow = useRef<string>("");
  useEffect(() => {
    if (!ds || !model) return;
    const key = `${ds.key}:${ds.title}:${model.key}`;
    if (defaultRow.current === key) return;
    defaultRow.current = key;
    setRowIndex(boundaryRow(ds, model));
  }, [ds, model]);

  // --- engine + presets -----------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    Promise.all([Viz.create("#lab-canvas"), loadPresets()])
      .then(([viz, list]) => {
        if (cancelled) return;
        vizRef.current = viz;
        if (process.env.NODE_ENV !== "production") (window as unknown as { __labViz?: Viz }).__labViz = viz;
        viz.background(0.055, 0.063, 0.078);
        setPresets(list);
        setStatus("ready");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
    return () => {
      cancelled = true;
      vizRef.current?.setPaused(true);
    };
  }, []);

  // size the drawing buffer to the element (device pixels, capped at 2x)
  useEffect(() => {
    if (status !== "ready") return;
    const el = wrapRef.current;
    const canvas = canvasRef.current;
    if (!el || !canvas) return;
    const fit = () => {
      const r = el.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      vizRef.current?.resize(Math.round(r.width * dpr), Math.round(r.height * dpr), r.width, r.height);
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    const io = new IntersectionObserver(([e]) => vizRef.current?.setPaused(!e.isIntersecting), { threshold: 0.01 });
    io.observe(el);
    return () => {
      ro.disconnect();
      io.disconnect();
    };
  }, [status]);

  useEffect(() => {
    vizRef.current?.autoRotate(rotate && mode !== "space2d" ? 0.05 : 0);
  }, [rotate, mode, scene]);

  // --- explanation -------------------------------------------------------------------------------
  // explanations are tagged with what they explain; a stale one (e.g. right after switching
  // datasets, before the recomputation has run) is never paired with the new dataset
  const withDelta = showDelta && mode === "surface";
  const exKey = ds && model ? `${ds.key}|${ds.title}|${model.key}|${rowIndex}|${withDelta}` : "";
  const [exState, setExState] = useState<{ key: string; ex: Explanation } | null>(null);
  const ex = exState && exState.key === exKey ? exState.ex : null;
  useEffect(() => {
    if (status !== "ready" || !ds || !model) return;
    setBusy(true);
    const id = window.setTimeout(() => {
      const x0 = ds.rows[Math.min(rowIndex, ds.rows.length - 1)];
      setExState({ key: exKey, ex: explain(ds, model, x0, withDelta) });
      setBusy(false);
    }, 30);
    return () => window.clearTimeout(id);
  }, [status, ds, model, rowIndex, withDelta, exKey]);

  // --- scene --------------------------------------------------------------------------------------
  useEffect(() => {
    const viz = vizRef.current;
    if (!viz || !ex || !ds || !model) return;
    setBusy(true);
    const id = window.setTimeout(() => {
      const built = buildScene(viz, ds, model, ex, {
        mode, axisMode, features, showData: showData || mode !== "surface", showDelta, levels, sweep, positive: ds.positive,
      });
      sceneRef.current = built;
      setScene(built);
      setBusy(false);
    }, 20);
    return () => window.clearTimeout(id);
  }, [ex, mode, axisMode, features, showData, levels, sweep, ds, model, showDelta]);

  // --- labels + captions follow the engine every frame ----------------------------------------
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const viz = vizRef.current;
      const sc = sceneRef.current;
      if (viz && sc) {
        const t = viz.time();
        sc.labels.forEach((l, i) => {
          const el = labelRefs.current[i];
          if (!el) return;
          const [x, y, vis] = viz.project(l.pos);
          const on = vis && (l.appearAt === undefined || t >= l.appearAt);
          el.style.opacity = on ? "1" : "0";
          el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
        });
        sc.tourAxisLabels.forEach((l, k) => {
          const el = labelRefs.current[sc.labels.length + k];
          if (!el) return;
          const tip = viz.tourAxisTip(l.j);
          const [x, y, vis] = viz.project(tip.map((v) => v * 1.12));
          el.style.opacity = vis ? "1" : "0";
          el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
        });
        let caption = "";
        for (const c of sc.captions) if (t >= c.t) caption = c.text;
        if (captionRef.current && captionRef.current.textContent !== caption) captionRef.current.textContent = caption;
        if (sweepRef.current) {
          const s = sc.sweep;
          if (s && t >= s.start) {
            const u = Math.min(1, (t - s.start) / s.dur);
            const top = s.nt - 1;
            const mid = top / 2;
            const v = u < 0.25 ? mid + (top - mid) * smooth(u / 0.25)
              : u < 0.75 ? top - top * smooth((u - 0.25) / 0.5)
                : mid * smooth((u - 0.75) / 0.25);
            const off = -s.range + (2 * s.range * v) / top;
            sweepRef.current.textContent = `4th-dimension offset ${off >= 0 ? "+" : "−"}${Math.abs(off).toFixed(2)}σ`;
            sweepRef.current.style.opacity = "1";
          } else sweepRef.current.style.opacity = "0";
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // --- interaction ------------------------------------------------------------------------------
  const pickAt = useCallback((clientX: number, clientY: number) => {
    const viz = vizRef.current;
    const sc = sceneRef.current;
    const el = wrapRef.current;
    if (!viz || !sc || !el || sc.pickObject < 0) return;
    const r = el.getBoundingClientRect();
    const hit = viz.pick(clientX - r.left, clientY - r.top, 16);
    if (hit && hit.object === sc.pickObject && hit.index < sc.pickRows.length) setRowIndex(sc.pickRows[hit.index]);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    pinch.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    drag.current = { x: e.clientX, y: e.clientY, moved: 0, id: e.pointerId };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const pts = pinch.current;
    if (pts.has(e.pointerId) && pts.size === 2) {
      const [a, b] = [...pts.values()];
      const before = Math.hypot(a.x - b.x, a.y - b.y);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const [c, d] = [...pts.values()];
      const after = Math.hypot(c.x - d.x, c.y - d.y);
      if (before > 0 && after > 0) vizRef.current?.zoom(before / after);
      if (drag.current) drag.current.moved = 99;
      return;
    }
    const dg = drag.current;
    if (!dg || dg.id !== e.pointerId) return;
    const dx = e.clientX - dg.x;
    const dy = e.clientY - dg.y;
    dg.x = e.clientX;
    dg.y = e.clientY;
    dg.moved += Math.abs(dx) + Math.abs(dy);
    if (mode !== "space2d") vizRef.current?.orbit(-dx * 0.008, -dy * 0.008);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    pinch.current.delete(e.pointerId);
    const dg = drag.current;
    if (dg && dg.id === e.pointerId && dg.moved < 5) pickAt(e.clientX, e.clientY);
    drag.current = null;
  };
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return; // plain scrolling keeps scrolling the page
      e.preventDefault();
      vizRef.current?.zoom(Math.exp(e.deltaY * 0.01));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [status]);

  const replay = () => {
    vizRef.current?.setTime(0);
    vizRef.current?.play(true, 1);
  };
  const skip = () => vizRef.current?.setTime((sceneRef.current?.duration ?? 0) + 0.01);
  const randomRow = () => ds && setRowIndex(Math.floor(Math.random() * ds.rows.length));

  // --- upload ---------------------------------------------------------------------------------------
  const onFile = async (file: File) => {
    setUploadMsg("");
    try {
      const text = await file.text();
      const { header, cells } = parseCsv(text);
      setUpload({ name: file.name, header, cells, target: header.length - 1 });
    } catch (err) {
      setUploadMsg(err instanceof Error ? err.message : String(err));
    }
  };
  const trainUpload = () => {
    if (!upload) return;
    setUploadMsg("Training an MLP in your browser…");
    window.setTimeout(() => {
      try {
        const table = prepareTable(upload.header, upload.cells, upload.target);
        const { dataset } = datasetFromTable(upload.name, table);
        setUploaded(dataset);
        setDsKey("upload");
        setModelKey("mlp");
        setRowIndex(0);
        setFeatures([0, Math.min(1, dataset.features.length - 1), Math.min(2, dataset.features.length - 1)]);
        setUploadMsg("");
      } catch (err) {
        setUploadMsg(err instanceof Error ? err.message : String(err));
      }
    }, 30);
  };

  // --- readout ---------------------------------------------------------------------------------------
  const readout = useMemo(() => {
    if (!ex || !ds || !model) return null;
    const near = localFidelity(ds, model, ex, 0.5);
    const mid = localFidelity(ds, model, ex, 1.0);
    const at = ex.at.attribution;
    const lime = ex.lime.attribution;
    const top = at.map((v, j) => [Math.abs(v) + Math.abs(lime[j]), j]).sort((a, b) => b[0] - a[0]).slice(0, 6).map(([, j]) => j);
    return { near, mid, top, at, lime };
  }, [ex, ds, model]);

  const pc = ds ? principalComponents(ds) : null;
  const labelCount = (scene?.labels.length ?? 0) + (scene?.tourAxisLabels.length ?? 0);

  return (
    <figure className="card lab" aria-label="3-D lab">
      <div className="controls">
        <label className="control">
          <span>Dataset</span>
          <select value={ds?.key ?? ""} onChange={(e) => { setDsKey(e.target.value); setRowIndex(0); setModelKey("mlp"); setFeatures([0, 1, 2]); }}>
            {datasets.map((d) => <option key={d.key} value={d.key}>{d.title}</option>)}
          </select>
        </label>
        <label className="control">
          <span>Black box</span>
          <select value={model?.key ?? ""} onChange={(e) => setModelKey(e.target.value)}>
            {ds?.models.map((m) => <option key={m.key} value={m.key}>{m.title}</option>)}
          </select>
        </label>
        <div className="control" style={{ minWidth: 0 }}>
          <span>View</span>
          <div className="seg" role="group" aria-label="View">
            {MODES.map((m) => (
              <button key={m.key} type="button" aria-pressed={mode === m.key} onClick={() => setMode(m.key)}>{m.label}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="controls">
        {mode !== "tour" && (
          <label className="control">
            <span>Axes</span>
            <select value={axisMode} onChange={(e) => setAxisMode(e.target.value as AxisMode)}>
              {AXES.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
            </select>
          </label>
        )}
        {mode !== "tour" && axisMode === "features" && ds &&
          [0, 1, 2].slice(0, mode === "space3d" ? 3 : 2).map((k) => (
            <label className="control" key={k}>
              <span>{["x", "y", "z"][k]} axis</span>
              <select value={features[k]} onChange={(e) => setFeatures((f) => f.map((v, q) => (q === k ? Number(e.target.value) : v)))}>
                {ds.features.map((name, j) => <option key={j} value={j}>{name}</option>)}
              </select>
            </label>
          ))}
        <div className="chips" style={{ alignSelf: "flex-end" }}>
          {mode === "surface" && (
            <>
              <button type="button" className="chip" aria-pressed={showData} onClick={() => setShowData((v) => !v)}>data rows</button>
              <button type="button" className="chip" aria-pressed={showDelta} onClick={() => setShowDelta((v) => !v)}>±δ ensemble</button>
            </>
          )}
          {(mode === "space3d" || mode === "space2d") && (
            <button type="button" className="chip" aria-pressed={levels} onClick={() => setLevels((v) => !v)}>0.25 / 0.75 levels</button>
          )}
          {mode === "space3d" && (
            <button type="button" className="chip" aria-pressed={sweep} onClick={() => setSweep((v) => !v)}>sweep a 4th dimension</button>
          )}
          {mode !== "space2d" && (
            <button type="button" className="chip" aria-pressed={rotate} onClick={() => setRotate((v) => !v)}>ambient rotation</button>
          )}
        </div>
      </div>

      <div
        ref={wrapRef}
        className="lab-stage"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <canvas id="lab-canvas" ref={canvasRef} aria-label="3-D rendering of the model, LIME and ArcTanh-LIME" />
        {Array.from({ length: labelCount }, (_, i) => {
          const text = i < (scene?.labels.length ?? 0) ? scene!.labels[i].text : scene!.tourAxisLabels[i - scene!.labels.length].text;
          const kind = i < (scene?.labels.length ?? 0) ? scene!.labels[i].kind : "axis";
          return (
            <div key={`${i}-${text}`} ref={(el) => { labelRefs.current[i] = el; }} className={`lab-label ${kind}`}>{text}</div>
          );
        })}
        <div ref={sweepRef} className="lab-sweep" />
        <div ref={captionRef} className="lab-caption" aria-live="polite" />
        {status === "loading" && <div className="lab-status">Loading the C++/WebGL2 engine…</div>}
        {status === "error" && <div className="lab-status">Could not start the 3-D engine: {error}</div>}
        {busy && status === "ready" && <div className="lab-busy">computing…</div>}
        <div className="lab-buttons">
          <button type="button" onClick={replay} title="Replay the animation">↺ Replay</button>
          <button type="button" onClick={skip} title="Jump to the end of the animation">⇥ End</button>
          <button type="button" onClick={() => vizRef.current?.zoom(0.85)} aria-label="Zoom in">＋</button>
          <button type="button" onClick={() => vizRef.current?.zoom(1.18)} aria-label="Zoom out">－</button>
        </div>
      </div>

      <div className="legend" style={{ marginTop: 10 }}>
        <span><span className="key" style={{ color: "#c9c8c2" }} /> black box</span>
        <span><span className="key c-lime" /> LIME</span>
        <span><span className="key c-atl" /> ArcTanh-LIME (walk, asymptotes, surrogate)</span>
        {showDelta && mode === "surface" && <span><span className="key" style={{ color: "#f9bd8c" }} /> ±δ blend</span>}
        <span><span className="key dot" style={{ color: "#e87ba4" }} /> {ds?.positive ?? "positive"}</span>
        <span><span className="key dot" style={{ color: "#1baf7a" }} /> negative</span>
      </div>

      <div className="grid-main-side" style={{ marginTop: 14 }}>
        <div className="small" style={{ color: "var(--ink-2)" }}>
          <p style={{ margin: "0 0 6px" }}>
            <b>Drag</b> to orbit, <b>pinch</b> or <b>ctrl/⌘ + scroll</b> to zoom, <b>click a data point</b> to explain it instead.
            {pc && mode !== "tour" && axisMode === "pca" && ` Principal components explain ${(pc.explained.slice(0, 3).reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of the standardised variance.`}
          </p>
          <p className="muted" style={{ margin: 0 }}>
            Rendering: C++17 + OpenGL ES 3 (WebGL2) compiled to WebAssembly; LIME, the ε-walk and the tanh⁻¹ fit run in
            the page (the TypeScript port checked against the Python reference). {ds?.attribution}
          </p>
          {ds?.notes.map((n) => <p key={n} className="muted" style={{ margin: "4px 0 0" }}>{n}</p>)}
          <div className="lab-upload">
            <label className="btn">
              Explore your own CSV…
              <input type="file" accept=".csv,.tsv,.txt,text/csv" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
            </label>
            {upload && (
              <>
                <label className="control" style={{ minWidth: 160 }}>
                  <span>Target column (binary risk)</span>
                  <select value={upload.target} onChange={(e) => setUpload({ ...upload, target: Number(e.target.value) })}>
                    {upload.header.map((h, j) => <option key={j} value={j}>{h}</option>)}
                  </select>
                </label>
                <button type="button" className="btn primary" onClick={trainUpload}>Train &amp; explore {upload.name}</button>
              </>
            )}
            {uploadMsg && <span className="small">{uploadMsg}</span>}
          </div>
        </div>
        {readout && ex && ds && (
          <div>
            <dl className="readout">
              <dt>instance</dt>
              <dd>row {rowIndex + 1} of {ds.rows.length} · label {ds.labels[rowIndex] > 0.5 ? ds.positive : "negative"}</dd>
              <dt>model P(x₀)</dt>
              <dd>{fmtNum(ex.f0, 3)}</dd>
              <dt>walk (up / down)</dt>
              <dd>{(ex.at.walkUp?.points.length ?? 1) - 1} / {(ex.at.walkDown?.points.length ?? 1) - 1} steps → [{fmtNum(ex.at.rawLo, 3)}, {fmtNum(ex.at.rawHi, 3)}]</dd>
              <dt>MSE at 0.5σ: LIME · ArcTanh</dt>
              <dd>{fmtNum(readout.near.lime, 3)} · {fmtNum(readout.near.arctanh, 3)}</dd>
              <dt>MSE at 1σ: LIME · ArcTanh</dt>
              <dd>{fmtNum(readout.mid.lime, 3)} · {fmtNum(readout.mid.arctanh, 3)}</dd>
            </dl>
            <div className="table-scroll" style={{ marginTop: 8 }}>
              <table className="data">
                <thead><tr><th>Feature (top 6)</th><th>LIME</th><th>ArcTanh</th></tr></thead>
                <tbody>
                  {readout.top.map((j) => (
                    <tr key={j}><td>{ds.features[j]}</td><td>{fmtNum(readout.lime[j], 3)}</td><td>{fmtNum(readout.at[j], 3)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button type="button" className="btn" style={{ marginTop: 8 }} onClick={randomRow}>Random instance</button>
          </div>
        )}
      </div>
    </figure>
  );
}
