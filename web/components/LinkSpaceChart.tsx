"use client";

import { ONE_D_DOMAIN, TanhSurrogate } from "@/lib/core/index.ts";
import type { OneDRun } from "@/lib/explain1d";
import { useElementWidth } from "@/lib/hooks";
import { clamp, fmtTick, niceTicks, scaleLinear } from "@/lib/plot";

const M = { l: 42, r: 16, t: 12, b: 30 };

/** The data seen by the tanh^-1 regressor: artanh((y - c) / a) is (close to) linear in x. */
export default function LinkSpaceChart({ run, maxSamples = 220 }: { run: OneDRun; maxSamples?: number }) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(520);
  const W = width;
  const H = Math.round(clamp(W * 0.55, 220, 320));
  const { arctanh, settings } = run;
  const s = arctanh.surrogate;
  if (!(s instanceof TanhSurrogate)) {
    return (
      <div ref={wrapRef} className="small muted" style={{ padding: 24 }}>
        The walk found no variation (constant region), so there is no link space to show.
      </div>
    );
  }
  const c = s.center;
  const a = s.halfRange;
  const lim = 1 - 1e-3;
  const tMax = Math.atanh(lim);
  const x = scaleLinear(ONE_D_DOMAIN, [M.l, W - M.r]);
  const y = scaleLinear([-tMax - 0.6, tMax + 0.6], [H - M.b, M.t]);
  const { Z, y: ys } = arctanh.fitData;
  const nNb = arctanh.neighborhood.y.length;
  const toT = (v: number) => Math.atanh(clamp((v - c) / a, -lim, lim));
  const eta = (xv: number) => s.beta0 + s.beta[0] * (xv - settings.x0);
  const xs = [ONE_D_DOMAIN[0], ONE_D_DOMAIN[1]];
  const yTicks = niceTicks(-tMax, tMax, 4);
  const xTicks = niceTicks(ONE_D_DOMAIN[0], ONE_D_DOMAIN[1], 6);
  const clipId = `clip-link-${settings.fn}`;

  return (
    <div ref={wrapRef}>
      <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="Link space: the arctanh-transformed black-box outputs against x, with the fitted straight line">
        <defs>
          <clipPath id={clipId}>
            <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} />
          </clipPath>
        </defs>
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
        {[tMax, -tMax].map((t) => (
          <line key={t} x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--atl)" strokeDasharray="3 4"
            strokeWidth={1} />
        ))}
        <text x={W - M.r - 4} y={y(tMax) - 5} textAnchor="end" className="halo">asymptote ↦ +∞ (clipped)</text>
        <text x={W - M.r - 4} y={y(-tMax) + 13} textAnchor="end" className="halo">asymptote ↦ −∞ (clipped)</text>
        <g clipPath={`url(#${clipId})`}>
          {Z.slice(0, Math.min(maxSamples, nNb)).map((z, i) => (
            <circle key={i} cx={x(settings.x0 + z[0])} cy={y(toT(ys[i]))} r={2} fill="var(--sample)" />
          ))}
          {Z.slice(nNb).map((z, i) => (
            <circle key={`w${i}`} cx={x(settings.x0 + z[0])} cy={y(toT(ys[nNb + i]))} r={2.6} fill="var(--atl)" />
          ))}
          <path d={`M${x(xs[0])},${y(eta(xs[0]))}L${x(xs[1])},${y(eta(xs[1]))}`} stroke="var(--atl)"
            strokeWidth={2.2} fill="none" />
        </g>
        <text x={M.l + 6} y={M.t + 12} className="label-strong halo">artanh((f − c) / a)</text>
      </svg>
    </div>
  );
}
