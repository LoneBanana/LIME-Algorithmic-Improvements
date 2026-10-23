"use client";

import { useState } from "react";
import { useElementWidth } from "@/lib/hooks";
import { fmtNum, fmtRatio, meaningful, scaleLog } from "@/lib/plot";
import type { RatioComparison } from "@/lib/results";

interface Props {
  comparisons: RatioComparison[];
  metrics: { key: string; label: string }[];
}

const ROW = 30;
const LABEL_W = 178;

export default function RatioPlot({ comparisons, metrics }: Props) {
  const [ci, setCi] = useState(0);
  const [metric, setMetric] = useState("r1");
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);
  const [ref, width] = useElementWidth<HTMLDivElement>(720);
  const comp = comparisons[ci];
  const narrow = width < 560;
  const labelW = narrow ? 128 : LABEL_W;
  const W = width;
  const top = 26;
  const H = top + comp.rows.length * ROW + 34;
  const x = scaleLog([0.01, 5], [labelW + 8, W - 14]);
  const ticks = [0.01, 0.03, 0.1, 0.3, 1, 3];
  const color = comp.a.startsWith("ArcTanh") ? "var(--atl)" : "var(--lime)";

  return (
    <figure className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <p className="fig-title">Paired comparison: mean-MSE ratio {comp.a} / {comp.b}</p>
          <p className="fig-sub">Left of 1 = {comp.a} more faithful. Filled dots: significant (Wilcoxon p &lt; 0.05, ≥ 5% apart, rank test agrees on the direction); hollow: not.</p>
        </div>
        <button type="button" className="btn" onClick={() => setTable((t) => !t)} aria-pressed={table}>
          {table ? "Show chart" : "Show table"}
        </button>
      </div>
      <div className="controls" style={{ marginBottom: 6 }}>
        <div className="control" style={{ minWidth: 0 }}>
          <span>Comparison</span>
          <div className="seg" role="group" aria-label="Comparison">
            {comparisons.map((c, i) => (
              <button key={c.key} type="button" aria-pressed={i === ci} onClick={() => setCi(i)}>
                {c.a} vs {c.b}
              </button>
            ))}
          </div>
        </div>
        <div className="control" style={{ minWidth: 0 }}>
          <span>Evaluated at</span>
          <div className="seg" role="group" aria-label="Distance">
            {metrics.map((m) => (
              <button key={m.key} type="button" aria-pressed={m.key === metric} onClick={() => setMetric(m.key)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {table ? (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Setting</th>
                {metrics.map((m) => <th key={m.key}>{m.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {comp.rows.map((r) => (
                <tr key={r.key}>
                  <td>{r.title}</td>
                  {metrics.map((m) => {
                    const c = r.cells[m.key];
                    const sig = meaningful(c.ratio, c.p, c.z);
                    return (
                      <td key={m.key} className={sig ? (c.ratio < 1 ? "better" : "worse") : undefined}>
                        {fmtRatio(c.ratio)} <span className="muted">({Math.round(c.win * 100)}%)</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">Ratio of mean MSE (win rate of {comp.a} in parentheses). Bold green/red: significant by the same rule, in favour of / against {comp.a}.</p>
        </div>
      ) : (
        <div ref={ref} style={{ position: "relative" }}>
          <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
            aria-label={`Dot plot of MSE ratios, ${comp.a} versus ${comp.b}`}>
            {ticks.map((t) => (
              <g key={t}>
                <line className="gridline" x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 30} />
                <text x={x(t)} y={H - 14} textAnchor="middle">{t}×</text>
              </g>
            ))}
            <line x1={x(1)} x2={x(1)} y1={top - 10} y2={H - 30} stroke="var(--ink)" strokeWidth={1.4} />
            <text x={x(1) - 6} y={top - 12} textAnchor="end">{narrow ? "← better" : `← ${comp.a} better`}</text>
            <text x={x(1) + 6} y={top - 12}>{narrow ? "worse →" : `${comp.b} better →`}</text>
            {comp.rows.map((r, i) => {
              const c = r.cells[metric];
              const cy = top + i * ROW + ROW / 2;
              const sig = meaningful(c.ratio, c.p, c.z);
              const cx = x(Math.min(Math.max(c.ratio, 0.0105), 4.8));
              return (
                <g key={r.key} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                  <rect x={0} y={cy - ROW / 2} width={W} height={ROW} fill={hover === i ? "var(--surface-2)" : "transparent"} />
                  <text x={0} y={cy + 4} style={{ fill: "var(--ink-2)", fontSize: narrow ? 10.5 : 12 }}>{r.title}</text>
                  <line x1={x(1)} x2={cx} y1={cy} y2={cy} stroke={color} strokeWidth={1.5} opacity={0.4} />
                  <circle cx={cx} cy={cy} r={5.5} fill={sig ? color : "var(--surface)"} stroke={sig ? "var(--surface)" : color}
                    strokeWidth={2} />
                  {!narrow && (
                    <text x={W - 14} y={cy + 4} textAnchor="end">{fmtRatio(c.ratio)}</text>
                  )}
                </g>
              );
            })}
          </svg>
          {hover !== null && (
            <div className="tip" style={{ left: Math.min(x(Math.min(Math.max(comp.rows[hover].cells[metric].ratio, 0.0105), 4.8)) + 12, W - 190), top: top + hover * ROW + ROW }}>
              <div className="tip-title">{comp.rows[hover].title}</div>
              <div className="row"><span>MSE ratio</span><b>{fmtRatio(comp.rows[hover].cells[metric].ratio)}</b></div>
              <div className="row"><span>{comp.a} wins</span><b>{Math.round(comp.rows[hover].cells[metric].win * 100)}%</b></div>
              <div className="row"><span>Wilcoxon p</span><b>{comp.rows[hover].cells[metric].p < 0.001 ? "< 0.001" : fmtNum(comp.rows[hover].cells[metric].p, 2)}</b></div>
            </div>
          )}
        </div>
      )}
    </figure>
  );
}
