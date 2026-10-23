"use client";

import { useState } from "react";
import { useElementWidth } from "@/lib/hooks";
import { fmtNum, linePath, logTicks, scaleLinear, scaleLog, svgPoint, fmtTick } from "@/lib/plot";
import type { FidelityPanel } from "@/lib/results";

const STYLE: Record<string, { color: string; dashed: boolean }> = {
  LIME: { color: "var(--lime)", dashed: false },
  "ArcTanh-LIME": { color: "var(--atl)", dashed: false },
  "LIME-δ": { color: "var(--lime)", dashed: true },
  "ArcTanh-LIME-δ": { color: "var(--atl)", dashed: true },
};

const M = { l: 44, r: 10, t: 8, b: 26 };

function Panel({ panel }: { panel: FidelityPanel }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(260);
  const [hover, setHover] = useState<number | null>(null);
  const W = width;
  const H = 168;
  const all = panel.series.flatMap((s) => s.values).filter((v) => v > 0);
  const lo = Math.min(...all) * 0.7;
  const hi = Math.max(...all) * 1.4;
  const x = scaleLinear([0, Math.max(...panel.radii)], [M.l, W - M.r]);
  const y = scaleLog([lo, hi], [H - M.b, M.t]);
  const ticks = logTicks(lo, hi);
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const p = svgPoint(e.currentTarget, e.clientX, e.clientY);
    const r = x.invert(p.x);
    let best = 0;
    panel.radii.forEach((v, i) => Math.abs(v - r) < Math.abs(panel.radii[best] - r) && (best = i));
    setHover(best);
  };
  return (
    <div className="multiple" ref={ref}>
      <h4>{panel.title}</h4>
      <p className="sub">{panel.sub} · {panel.n} instances</p>
      <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`${panel.title}: mean squared error against distance from the explained instance`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 5} y={y(t) + 3.5} textAnchor="end">{fmtTick(t)}</text>
          </g>
        ))}
        <line className="axisline" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        {[0, 1, 2, 3, 4].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 15} textAnchor="middle">{t}σ</text>
        ))}
        {hover !== null && (
          <line x1={x(panel.radii[hover])} x2={x(panel.radii[hover])} y1={M.t} y2={H - M.b} stroke="var(--ink)"
            strokeWidth={1} opacity={0.35} />
        )}
        {panel.series.map((s) => (
          <g key={s.label}>
            <path d={linePath(panel.radii.map(x), s.values.map(y))} fill="none" stroke={STYLE[s.label].color}
              strokeWidth={2} strokeDasharray={STYLE[s.label].dashed ? "5 4" : undefined} strokeLinejoin="round"
              strokeLinecap="round" />
            {hover !== null && (
              <circle cx={x(panel.radii[hover])} cy={y(s.values[hover])} r={4} fill={STYLE[s.label].color}
                stroke="var(--surface)" strokeWidth={2} />
            )}
          </g>
        ))}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: Math.min(Math.max(x(panel.radii[hover]) - 80, 0), W - 170), top: 40 }}>
          <div className="tip-title">r = {panel.radii[hover]}σ · mean MSE</div>
          {[...panel.series].sort((a, b) => a.values[hover] - b.values[hover]).map((s) => (
            <div className="row" key={s.label}>
              <span className={STYLE[s.label].dashed ? "key dashed" : "key"} style={{ color: STYLE[s.label].color }} />
              <span>{s.label}</span>
              <b>{fmtNum(s.values[hover], 3)}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FidelityMultiples({ panels }: { panels: FidelityPanel[] }) {
  const [table, setTable] = useState(false);
  const labels = panels[0].series.map((s) => s.label);
  return (
    <figure className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <p className="fig-title">Fidelity against distance from the explained instance</p>
          <p className="fig-sub">Mean squared error on fresh black-box queries at exactly r standard deviations from x₀ (log scale; lower is better).</p>
        </div>
        <button type="button" className="btn" onClick={() => setTable((t) => !t)} aria-pressed={table}>
          {table ? "Show charts" : "Show table"}
        </button>
      </div>
      <div className="legend">
        {labels.map((l) => (
          <span key={l}>
            <span className={STYLE[l].dashed ? "key dashed" : "key"} style={{ color: STYLE[l].color }} /> {l}
          </span>
        ))}
      </div>
      {table ? (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Setting</th>
                <th>Method</th>
                {panels[0].radii.map((r) => <th key={r}>r = {r}σ</th>)}
              </tr>
            </thead>
            <tbody>
              {panels.flatMap((p) =>
                p.series.map((s, i) => (
                  <tr key={`${p.key}-${s.label}`}>
                    <td>{i === 0 ? p.title : ""}</td>
                    <td>{s.label}</td>
                    {s.values.map((v, j) => <td key={j}>{fmtNum(v, 3)}</td>)}
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="multiples">
          {panels.map((p) => <Panel key={p.key} panel={p} />)}
        </div>
      )}
      <figcaption>
        Each point is the mean over the explained instances (60 per 1-D/2-D setting, 40 per 8-D/10-D setting) of the
        MSE on 100 random points on the sphere of radius r (clipped to the data domain; in 1-D the two points x₀ ± r).
        Hover a panel for values.
      </figcaption>
    </figure>
  );
}
