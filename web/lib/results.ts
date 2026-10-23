/**
 * Server-side shaping of data/results.json (written by `python3 -m experiments.export_web`).
 * Only the slim, derived structures below are sent to the browser.
 */
import raw from "@/data/results.json";

interface Desc { mean: number; median: number; q25: number; q75: number; n: number }
interface Paired { mean_diff: number; ci95: [number, number]; win_rate: number; wilcoxon_p: number; wilcoxon_z: number; median_ratio: number | null; mean_ratio: number | null }
interface Summary {
  n_instances: number;
  title?: string;
  d?: number;
  quality?: Record<string, number> | null;
  methods: Record<string, Record<string, Desc>>;
  comparisons: Record<string, Record<string, Paired>>;
  walk?: { up_steps: Desc; down_steps: Desc; stop_reasons: Record<string, number>; fallbacks: Record<string, number>; clipped_fraction: Desc };
}
interface Results {
  benchmarks: { radii: number[]; num_samples: number; sphere_points: number; neighborhood_points: number; global_points: number; settings: Record<string, Summary> };
  ablations: { single: Record<string, Summary>; ensemble: Record<string, Summary> };
  stability: { n_instances: number; n_seeds: number; settings: Record<string, { title: string; methods: Record<string, Record<string, { mean: number; median: number }>> }> };
  quality: Record<string, Record<string, number>>;
}

const R = raw as unknown as Results;

export const SETTING_ORDER = [
  "1d_logistic", "1d_sine", "1d_bumps", "1d_staircase", "1d_ripple",
  "moons_mlp", "synth_oracle", "synth_mlp", "synth_rf", "friedman_oracle", "friedman_mlp",
];

export const SETTING_NAMES: Record<string, { title: string; sub: string }> = {
  "1d_logistic": { title: "1-D logistic", sub: "σ(2x)" },
  "1d_sine": { title: "1-D sine", sub: "sin x" },
  "1d_bumps": { title: "1-D bumps", sub: "three Gaussian bumps" },
  "1d_staircase": { title: "1-D staircase", sub: "three sigmoid steps" },
  "1d_ripple": { title: "1-D trend + ripple", sub: "0.15x + 0.25 sin 2x" },
  moons_mlp: { title: "Two moons · MLP", sub: "d = 2, probability" },
  synth_oracle: { title: "Synthetic · true p(y|x)", sub: "d = 8, probability" },
  synth_mlp: { title: "Synthetic · MLP", sub: "d = 8, probability" },
  synth_rf: { title: "Synthetic · random forest", sub: "d = 8, piecewise constant" },
  friedman_oracle: { title: "Friedman #1 · true function", sub: "d = 10, regression" },
  friedman_mlp: { title: "Friedman #1 · MLP", sub: "d = 10, regression" },
};

export const MAIN_METHODS = ["LIME", "ArcTanh-LIME", "LIME-δ", "ArcTanh-LIME-δ"] as const;

const bench = R.benchmarks;
export const RADII = bench.radii;
export const META = {
  numSamples: bench.num_samples,
  spherePoints: bench.sphere_points,
  neighborhoodPoints: bench.neighborhood_points,
  globalPoints: bench.global_points,
};

export interface FidelityPanel {
  key: string;
  title: string;
  sub: string;
  n: number;
  radii: number[];
  series: { label: string; values: number[]; q25: number[]; q75: number[] }[];
}

export function fidelityPanels(): FidelityPanel[] {
  return SETTING_ORDER.map((key) => {
    const s = bench.settings[key];
    return {
      key,
      ...SETTING_NAMES[key],
      n: s.n_instances,
      radii: RADII,
      series: MAIN_METHODS.map((label) => ({
        label,
        values: RADII.map((r) => s.methods[label][`r${r}`].mean),
        q25: RADII.map((r) => s.methods[label][`r${r}`].q25),
        q75: RADII.map((r) => s.methods[label][`r${r}`].q75),
      })),
    };
  });
}

export const RATIO_METRICS = [
  { key: "r0.5", label: "0.5σ" },
  { key: "r1", label: "1σ" },
  { key: "r2", label: "2σ" },
  { key: "r4", label: "4σ" },
  { key: "neighborhood_wmse", label: "LIME neighbourhood" },
  { key: "global_mse", label: "whole test set" },
];

export interface RatioCell { ratio: number; win: number; p: number; z: number }
export interface RatioComparison {
  key: string;
  a: string;
  b: string;
  rows: { key: string; title: string; sub: string; cells: Record<string, RatioCell> }[];
}

export function ratioComparisons(): RatioComparison[] {
  const pairs: [string, string][] = [
    ["ArcTanh-LIME", "LIME (matched budget)"],
    ["ArcTanh-LIME", "LIME (same data)"],
    ["ArcTanh-LIME-δ", "LIME-δ"],
    ["ArcTanh-LIME-δ", "ArcTanh-LIME"],
    ["LIME-δ", "LIME"],
  ];
  return pairs.map(([a, b]) => ({
    key: `${a} vs ${b}`,
    a,
    b,
    rows: SETTING_ORDER.map((k) => {
      const comp = bench.settings[k].comparisons[`${a} vs ${b}`];
      const cells: Record<string, RatioCell> = {};
      for (const m of RATIO_METRICS) {
        const c = comp[m.key];
        cells[m.key] = { ratio: c.mean_ratio ?? NaN, win: c.win_rate, p: c.wilcoxon_p, z: c.wilcoxon_z };
      }
      return { key: k, ...SETTING_NAMES[k], cells };
    }),
  }));
}

function cmp(a: string, b: string, setting: string, metric: string): Paired {
  return bench.settings[setting].comparisons[`${a} vs ${b}`][metric];
}

/** Same rule as the tables: p < 0.05, means at least 5% apart, rank test agrees on the direction. */
function significant(c: Paired, better: boolean): boolean {
  const r = c.mean_ratio ?? 1;
  const ok = c.wilcoxon_p < 0.05 && Math.abs(r - 1) >= 0.05 && c.wilcoxon_z < 0 === r < 1;
  return ok && (better ? r < 1 : r > 1);
}

/** Headline numbers for ArcTanh-LIME against a LIME baseline, all derived from the benchmark summaries. */
export function headline(baseline = "LIME (matched budget)") {
  const A = "ArcTanh-LIME";
  const count = (metric: string, better: boolean) =>
    SETTING_ORDER.filter((s) => significant(cmp(A, baseline, s, metric), better));
  const ratios = (metric: string) =>
    SETTING_ORDER.map((s) => cmp(A, baseline, s, metric).mean_ratio ?? NaN).sort((x, y) => x - y);
  const median = (v: number[]) => v[Math.floor((v.length - 1) / 2)] * 0.5 + v[Math.ceil((v.length - 1) / 2)] * 0.5;
  const q = (s: string, m: string) => bench.settings[s].methods[m].queries.median;
  const overhead = SETTING_ORDER.map((s) => q(s, A) / q(s, "LIME") - 1);
  const deltaCost = SETTING_ORDER.map((s) => q(s, "ArcTanh-LIME-δ") / q(s, "LIME"));
  return {
    total: SETTING_ORDER.length,
    betterAt05: count("r0.5", true),
    betterAt1: count("r1", true),
    worseAt2: count("r2", false),
    worseAt4: count("r4", false),
    worseNeighborhood: count("neighborhood_wmse", false),
    medianRatio05: median(ratios("r0.5")),
    medianRatio1: median(ratios("r1")),
    rangeRatio1: [ratios("r1")[0], ratios("r1")[ratios("r1").length - 1]],
    overheadRange: [Math.min(...overhead), Math.max(...overhead)],
    deltaCostRange: [Math.min(...deltaCost), Math.max(...deltaCost)],
    ensembleBetterAt4: SETTING_ORDER.filter((s) => significant(cmp("ArcTanh-LIME-δ", "LIME", s, "r4"), true)),
  };
}

export function walkDiagnostics(setting: string) {
  return bench.settings[setting].walk!;
}

export function benchMethod(setting: string, method: string) {
  return bench.settings[setting].methods[method];
}

export function benchComparison(setting: string, a: string, b: string, metric: string) {
  return cmp(a, b, setting, metric);
}

export interface CostRow { key: string; title: string; queries: Record<string, number>; seconds: Record<string, number> }

export function costRows(): CostRow[] {
  return SETTING_ORDER.map((k) => {
    const m = bench.settings[k].methods;
    const labels = ["LIME", "ArcTanh-LIME", "LIME-δ", "ArcTanh-LIME-δ"];
    return {
      key: k,
      title: SETTING_NAMES[k].title,
      queries: Object.fromEntries(labels.map((l) => [l, m[l].queries.median])),
      seconds: Object.fromEntries(labels.map((l) => [l, m[l].seconds.median])),
    };
  });
}

export interface GradRow { key: string; title: string; lime: number; arctanh: number; p?: number }

export function gradientRows(): GradRow[] {
  return SETTING_ORDER.filter((k) => bench.settings[k].methods["LIME"].grad_cosine?.n).map((k) => ({
    key: k,
    title: SETTING_NAMES[k].title,
    lime: bench.settings[k].methods["LIME"].grad_cosine.mean,
    arctanh: bench.settings[k].methods["ArcTanh-LIME"].grad_cosine.mean,
  }));
}

export interface AblationTable {
  group: "single" | "ensemble";
  setting: string;
  title: string;
  n: number;
  reference: string;
  rows: {
    label: string;
    queries: number;
    values: Record<string, number>;
    ratio: Record<string, number | null>;
    p: Record<string, number | null>;
    z: Record<string, number | null>;
  }[];
}

export const ABLATION_METRICS = ["r0.5", "r1", "r2", "r4", "neighborhood_wmse", "global_mse"];

export function ablationTables(): AblationTable[] {
  const out: AblationTable[] = [];
  for (const group of ["single", "ensemble"] as const) {
    const reference = group === "single" ? "ArcTanh-LIME (default)" : "ArcTanh-LIME-δ (δ = 1, default)";
    for (const [setting, s] of Object.entries(R.ablations[group])) {
      out.push({
        group,
        setting,
        title: SETTING_NAMES[setting]?.title ?? setting,
        n: s.n_instances,
        reference,
        rows: Object.entries(s.methods).map(([label, m]) => {
          const comp = s.comparisons[`${label} vs ${reference}`];
          return {
            label,
            queries: m.queries.median,
            values: Object.fromEntries(ABLATION_METRICS.map((k) => [k, m[k].mean])),
            ratio: Object.fromEntries(ABLATION_METRICS.map((k) => [k, comp ? comp[k].mean_ratio : null])),
            p: Object.fromEntries(ABLATION_METRICS.map((k) => [k, comp ? comp[k].wilcoxon_p : null])),
            z: Object.fromEntries(ABLATION_METRICS.map((k) => [k, comp ? comp[k].wilcoxon_z : null])),
          };
        }),
      });
    }
  }
  return out;
}

export function ablationValue(setting: string, label: string, metric: string, group: "single" | "ensemble" = "single") {
  return R.ablations[group][setting].methods[label][metric].mean;
}

export function ablationComparison(setting: string, a: string, b: string, metric: string): Paired {
  return R.ablations.single[setting].comparisons[`${a} vs ${b}`][metric];
}

/** Settings of the single-anchor ablation where variant A is significantly (p < 0.05) better than B on a metric. */
export function ablationWins(a: string, b: string, metric: string): { wins: string[]; total: number } {
  const settings = Object.keys(R.ablations.single);
  const wins = settings.filter((s) => {
    const c = R.ablations.single[s].comparisons[`${a} vs ${b}`]?.[metric];
    return c !== undefined && c.wilcoxon_p < 0.05 && (c.mean_ratio ?? 1) < 1 && c.wilcoxon_z < 0;
  });
  return { wins, total: settings.length };
}

export interface StabilityRow { key: string; title: string; values: Record<string, { cosine: number; spread: number; predSd: number }> }

export function stabilityRows(): StabilityRow[] {
  return Object.entries(R.stability.settings).map(([key, s]) => ({
    key,
    title: SETTING_NAMES[key]?.title ?? s.title,
    values: Object.fromEntries(Object.entries(s.methods).map(([m, v]) => [m, {
      cosine: v.pairwise_cosine.mean, spread: v.relative_spread.mean, predSd: v.prediction_sd.mean,
    }])),
  }));
}

export const STABILITY_META = { instances: R.stability.n_instances, seeds: R.stability.n_seeds };
export const QUALITY = R.quality;
