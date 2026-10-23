"use client";

import { ONE_D_DOMAIN } from "@/lib/core/index.ts";
import { METHOD_META, predictor } from "@/lib/explain1d";
import type { MethodKey, OneDRun } from "@/lib/explain1d";
import { useElementWidth } from "@/lib/hooks";
import { clamp, fmtTick, linePath, linspace, niceTicks, scaleLinear } from "@/lib/plot";

const M = { l: 42, r: 16, t: 12, b: 30 };
const GRID = linspace(ONE_D_DOMAIN[0], ONE_D_DOMAIN[1], 401);

/** |g(x) - f(x)| across the domain: how far from x0 each surrogate stays faithful. */
export default function ErrorChart({ run, methods }: { run: OneDRun; methods: MethodKey[] }) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(720);
  const W = width;
  const H = Math.round(clamp(W * 0.26, 150, 210));
  const x = scaleLinear(ONE_D_DOMAIN, [M.l, W - M.r]);
  const x0 = run.settings.x0;
  const errs = methods.map((k) => {
    const p = predictor(run, k);
    return GRID.map((v) => Math.abs(p(v) - run.g(v)));
  });
  // y-range: what happens within +/- 3 standard deviations of x0, capped so lines stay readable
  let cap = 0;
  errs.forEach((e) => e.forEach((v, i) => Math.abs(GRID[i] - x0) <= 3 && (cap = Math.max(cap, v))));
  let fLo = Infinity;
  let fHi = -Infinity;
  for (const v of GRID) {
    fLo = Math.min(fLo, run.g(v));
    fHi = Math.max(fHi, run.g(v));
  }
  cap = Math.min(Math.max(cap, 1e-3), 0.8 * (fHi - fLo));
  const y = scaleLinear([0, cap * 1.08], [H - M.b, M.t]);
  const yTicks = niceTicks(0, cap, 3);
  const xTicks = niceTicks(ONE_D_DOMAIN[0], ONE_D_DOMAIN[1], 6);
  const clipId = `clip-err-${run.settings.fn}`;

  return (
    <div ref={wrapRef}>
      <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="Absolute error of each surrogate across x">
        <defs>
          <clipPath id={clipId}>
            <rect x={M.l} y={M.t - 2} width={W - M.l - M.r} height={H - M.t - M.b + 2} />
          </clipPath>
        </defs>
        <rect x={x(Math.max(ONE_D_DOMAIN[0], x0 - 1))} y={M.t}
          width={x(Math.min(ONE_D_DOMAIN[1], x0 + 1)) - x(Math.max(ONE_D_DOMAIN[0], x0 - 1))}
          height={H - M.t - M.b} fill="var(--sample)" opacity={0.16} />
        {yTicks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 3.5} textAnchor="end">{fmtTick(t)}</text>
          </g>
        ))}
        <line className="axisline" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 16} textAnchor="middle">{fmtTick(t)}</text>
        ))}
        <g clipPath={`url(#${clipId})`}>
          {methods.map((k, i) => (
            <path key={k} d={linePath(GRID.map(x), errs[i].map(y))} fill="none" stroke={METHOD_META[k].color}
              strokeWidth={2} strokeDasharray={METHOD_META[k].dashed ? "6 4" : undefined} />
          ))}
        </g>
        <line x1={x(x0)} x2={x(x0)} y1={M.t} y2={H - M.b} stroke="var(--ink)" opacity={0.35} />
        <text x={M.l + 6} y={M.t + 11} className="label-strong halo">|g(x) − f(x)|</text>
        <text x={x(x0) + 5} y={H - M.b - 6} className="halo">±1σ around x₀</text>
      </svg>
    </div>
  );
}
