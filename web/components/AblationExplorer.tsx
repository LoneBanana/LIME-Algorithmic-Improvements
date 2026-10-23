"use client";

import { useState } from "react";
import { fmtNum, fmtRatio, meaningful } from "@/lib/plot";
import type { AblationTable } from "@/lib/results";

const COLS = [
  { key: "r0.5", label: "0.5σ" },
  { key: "r1", label: "1σ" },
  { key: "r2", label: "2σ" },
  { key: "r4", label: "4σ" },
  { key: "neighborhood_wmse", label: "nbhd" },
  { key: "global_mse", label: "global" },
];

export default function AblationExplorer({ tables }: { tables: AblationTable[] }) {
  const [group, setGroup] = useState<"single" | "ensemble">("single");
  const options = tables.filter((t) => t.group === group);
  const [setting, setSetting] = useState(options[0].setting);
  const table = options.find((t) => t.setting === setting) ?? options[0];

  return (
    <figure className="card">
      <p className="fig-title">Ablations: what each design choice contributes</p>
      <p className="fig-sub">
        Mean MSE ratio of each variant against the default ({table.reference}); ▼ green = better, ▲ red = worse,
        marked only when Wilcoxon p &lt; 0.05, the means differ by at least 5% and the rank test agrees on the
        direction. Raw mean MSE underneath.
      </p>
      <div className="controls">
        <div className="control" style={{ minWidth: 0 }}>
          <span>Study</span>
          <div className="seg" role="group" aria-label="Ablation study">
            <button type="button" aria-pressed={group === "single"} onClick={() => { setGroup("single"); setSetting(tables.find((t) => t.group === "single")!.setting); }}>
              single-anchor design
            </button>
            <button type="button" aria-pressed={group === "ensemble"} onClick={() => { setGroup("ensemble"); setSetting(tables.find((t) => t.group === "ensemble")!.setting); }}>
              ±δ ensemble
            </button>
          </div>
        </div>
        <div className="control" style={{ minWidth: 0 }}>
          <span>Setting</span>
          <div className="seg" role="group" aria-label="Setting">
            {options.map((t) => (
              <button key={t.setting} type="button" aria-pressed={t.setting === table.setting} onClick={() => setSetting(t.setting)}>
                {t.title}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Variant ({table.n} instances)</th>
              <th>Queries</th>
              {COLS.map((c) => <th key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r) => (
              <tr key={r.label}>
                <td style={{ fontWeight: r.label === table.reference ? 650 : 400 }}>{r.label}</td>
                <td>{r.queries.toLocaleString("en-US")}</td>
                {COLS.map((c) => {
                  const ratio = r.ratio[c.key];
                  const p = r.p[c.key];
                  const sig = meaningful(ratio, p, r.z[c.key]);
                  const better = sig && ratio! < 1;
                  const worse = sig && ratio! > 1;
                  return (
                    <td key={c.key} className={better ? "better" : worse ? "worse" : undefined}>
                      {ratio === null ? <span className="muted">ref.</span> : <>{better ? "▼ " : worse ? "▲ " : ""}{fmtRatio(ratio)}</>}
                      <div className="muted" style={{ fontSize: "0.72rem", fontWeight: 400 }}>{fmtNum(r.values[c.key], 3)}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
