"use client";

import { useMemo, useRef } from "react";
import { ONE_D_DOMAIN, TanhSurrogate } from "@/lib/core/index.ts";
import type { OneDRun } from "@/lib/explain1d";
import { useElementWidth } from "@/lib/hooks";
import { clamp, fmtNum, fmtTick, linePath, linspace, niceTicks, scaleLinear, svgPoint } from "@/lib/plot";

export interface Layers {
  density: boolean;
  samples: boolean;
  lime: boolean;
  walk: boolean;
  asymptotes: boolean;
  arctanh: boolean;
  anchors: boolean;
  limeDelta: boolean;
  arctanhDelta: boolean;
}

export const ALL_LAYERS: Layers = {
  density: true, samples: true, lime: true, walk: true, asymptotes: true, arctanh: true,
  anchors: true, limeDelta: true, arctanhDelta: true,
};

interface Props {
  run: OneDRun;
  layers: Layers;
  onX0?: (x: number) => void;
  /** fraction of each walk to reveal (animation), default 1 */
  walkReveal?: number;
  maxSamples?: number;
  ariaLabel: string;
}

const M = { l: 42, r: 16, t: 14, b: 30 };
const GRID = linspace(ONE_D_DOMAIN[0], ONE_D_DOMAIN[1], 481);

export function yDomainFor(g: (x: number) => number): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const x of GRID) {
    const v = g(x);
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const pad = 0.2 * (hi - lo || 1);
  return [lo - pad, hi + pad];
}

export default function OneDChart({ run, layers, onX0, walkReveal = 1, maxSamples = 220, ariaLabel }: Props) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(720);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const W = width;
  const H = Math.round(clamp(W * 0.5, 250, 390));
  const { g, settings, arctanh, lime, limeDelta, arctanhDelta } = run;
  const yDom = useMemo(() => yDomainFor(g), [g]);
  const x = scaleLinear(ONE_D_DOMAIN, [M.l, W - M.r]);
  const y = scaleLinear(yDom, [H - M.b, M.t]);
  const clipId = `clip-1d-${settings.fn}`;
  const px = GRID.map(x);

  const curve = (pred: (v: number) => number) => linePath(px, GRID.map((v) => y(pred(v))));
  const fPath = useMemo(() => linePath(px, GRID.map((v) => y(g(v)))), [g, W, H]);

  const nb = arctanh.neighborhood;
  const nShow = Math.min(maxSamples, nb.points.length);
  const reveal = (n: number) => Math.max(1, Math.ceil(n * walkReveal));
  const up = arctanh.walkUp;
  const down = arctanh.walkDown;
  const yTicks = niceTicks(yDom[0], yDom[1], 5);
  const xTicks = niceTicks(ONE_D_DOMAIN[0], ONE_D_DOMAIN[1], 6);
  const x0 = settings.x0;
  const f0 = g(x0);
  const tanh = arctanh.surrogate instanceof TanhSurrogate ? arctanh.surrogate : null;

  const setFromEvent = (e: React.PointerEvent) => {
    if (!onX0 || !svgRef.current) return;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    onX0(+clamp(x.invert(p.x), ONE_D_DOMAIN[0] + 0.05, ONE_D_DOMAIN[1] - 0.05).toFixed(2));
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (!onX0) return;
    const step = e.shiftKey ? 0.5 : 0.1;
    let next: number | null = null;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = x0 - step;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") next = x0 + step;
    if (e.key === "Home") next = ONE_D_DOMAIN[0] + 0.05;
    if (e.key === "End") next = ONE_D_DOMAIN[1] - 0.05;
    if (next !== null) {
      e.preventDefault();
      onX0(+clamp(next, ONE_D_DOMAIN[0] + 0.05, ONE_D_DOMAIN[1] - 0.05).toFixed(2));
    }
  };

  const density = useMemo(() => {
    const sd = 1; // LIME samples N(x0, scale^2) with scale = 1
    const xs = linspace(x0 - 3.5 * sd, x0 + 3.5 * sd, 120).filter((v) => v >= ONE_D_DOMAIN[0] && v <= ONE_D_DOMAIN[1]);
    const base = H - M.b;
    const top = xs.map((v) => base - 26 * Math.exp(-0.5 * ((v - x0) / sd) ** 2));
    if (!xs.length) return "";
    return `${linePath(xs.map(x), top)}L${x(xs[xs.length - 1]).toFixed(2)},${base}L${x(xs[0]).toFixed(2)},${base}Z`;
  }, [x0, W, H]);

  const walkDots = (walk: typeof up, kind: "up" | "down") => {
    if (!walk) return null;
    const n = reveal(walk.points.length);
    const pts = walk.points.slice(0, n);
    const vals = walk.values.slice(0, n);
    const done = n >= walk.points.length;
    const endX = x(pts[pts.length - 1][0]);
    const endY = y(vals[vals.length - 1]);
    const labelAbove = kind === "up";
    return (
      <g>
        {pts.slice(1).map((p, i) => (
          <circle key={i} cx={x(p[0])} cy={y(vals[i + 1])} r={2.6} fill="var(--atl)" />
        ))}
        {done && walk.points.length > 1 && (
          <g>
            <circle cx={endX} cy={endY} r={5.5} fill="var(--atl)" stroke="var(--surface)" strokeWidth={2} />
            <text x={endX} y={labelAbove ? endY - 10 : endY + 18} textAnchor="middle" className="label-strong halo">
              {kind === "up" ? "local max" : "local min"}
            </text>
          </g>
        )}
      </g>
    );
  };

  const anchorMembers = arctanhDelta.members.slice(1);

  return (
    <div ref={wrapRef}>
      <svg
        ref={svgRef}
        className="chart"
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} />
          </clipPath>
        </defs>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="gridline" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 3.5} textAnchor="end">{fmtTick(t)}</text>
          </g>
        ))}
        <line className="axisline" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        {xTicks.map((t) => (
          <text key={`x${t}`} x={x(t)} y={H - M.b + 16} textAnchor="middle">{fmtTick(t)}</text>
        ))}

        <g clipPath={`url(#${clipId})`}>
          {layers.density && <path d={density} fill="var(--sample)" opacity={0.28} />}
          {layers.samples && (
            <path
              d={nb.points.slice(0, nShow).map((p) => `M${x(p[0]).toFixed(1)},${H - M.b}v-7`).join("")}
              stroke="var(--ink-2)" strokeWidth={1} opacity={0.45}
            />
          )}
          {layers.anchors &&
            anchorMembers.map((m, i) => (
              <path key={i} d={curve((v) => m.surrogate.predict([v]))} fill="none" stroke="var(--atl)"
                strokeWidth={1} opacity={0.32} />
            ))}
          <path d={fPath} fill="none" stroke="var(--ink)" strokeWidth={2.2} strokeLinejoin="round" />
          {layers.lime && (
            <path d={curve((v) => lime.surrogate.predict([v]))} fill="none" stroke="var(--lime)" strokeWidth={2} />
          )}
          {layers.limeDelta && (
            <path d={curve((v) => limeDelta.surrogate.predict([v]))} fill="none" stroke="var(--lime)"
              strokeWidth={2} strokeDasharray="6 4" />
          )}
          {layers.asymptotes && tanh && (
            <g>
              {[arctanh.hi, arctanh.lo].map((v, i) => (
                <line key={i} x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--atl)" strokeWidth={1}
                  strokeDasharray="3 4" opacity={0.9} />
              ))}
            </g>
          )}
          {layers.arctanh && (
            <path d={curve((v) => arctanh.surrogate.predict([v]))} fill="none" stroke="var(--atl)" strokeWidth={2.4} />
          )}
          {layers.arctanhDelta && (
            <path d={curve((v) => arctanhDelta.surrogate.predict([v]))} fill="none" stroke="var(--atl)"
              strokeWidth={2} strokeDasharray="6 4" />
          )}
          {layers.walk && walkDots(up, "up")}
          {layers.walk && walkDots(down, "down")}
        </g>

        {layers.asymptotes && tanh && W >= 520 && (
          <g>
            <text x={W - M.r - 4} y={y(arctanh.hi) - 5} textAnchor="end" className="halo">asymptote {fmtNum(arctanh.hi, 3)}</text>
            <text x={W - M.r - 4} y={y(arctanh.lo) + 13} textAnchor="end" className="halo">asymptote {fmtNum(arctanh.lo, 3)}</text>
          </g>
        )}
        {layers.anchors &&
          arctanhDelta.anchors.slice(1).map((a, i) => (
            <g key={i}>
              <line x1={x(a[0])} x2={x(a[0])} y1={H - M.b} y2={H - M.b - 7} stroke="var(--ink-2)" strokeWidth={1.5} />
              <path d={`M${x(a[0])},${y(g(a[0])) - 6}l5,6l-5,6l-5,-6z`} fill="var(--surface)" stroke="var(--ink)"
                strokeWidth={1.5} />
              <text x={x(a[0])} y={H - M.b - 10} textAnchor="middle" className="halo">
                {a[0] < x0 ? "x₀−δ" : "x₀+δ"}
              </text>
            </g>
          ))}

        {/* x0 handle */}
        <line x1={x(x0)} x2={x(x0)} y1={M.t} y2={H - M.b} stroke="var(--ink)" strokeWidth={1} opacity={0.35} />
        <g
          role={onX0 ? "slider" : undefined}
          tabIndex={onX0 ? 0 : undefined}
          aria-label={onX0 ? "Instance x0 (drag or use arrow keys)" : undefined}
          aria-valuemin={onX0 ? ONE_D_DOMAIN[0] : undefined}
          aria-valuemax={onX0 ? ONE_D_DOMAIN[1] : undefined}
          aria-valuenow={onX0 ? x0 : undefined}
          onKeyDown={onKey}
        >
          <circle cx={x(x0)} cy={y(f0)} r={7.5} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2.5} />
          <text x={x(x0) + 11} y={y(f0) - 9} className="label-strong halo">x₀</text>
        </g>

        {onX0 && (
          <rect
            className="drag-surface"
            x={M.l}
            y={M.t}
            width={W - M.l - M.r}
            height={H - M.t - M.b}
            fill="transparent"
            onPointerDown={(e) => {
              dragging.current = true;
              (e.target as Element).setPointerCapture(e.pointerId);
              setFromEvent(e);
            }}
            onPointerMove={(e) => dragging.current && setFromEvent(e)}
            onPointerUp={() => (dragging.current = false)}
            onPointerCancel={() => (dragging.current = false)}
          />
        )}
      </svg>
    </div>
  );
}
