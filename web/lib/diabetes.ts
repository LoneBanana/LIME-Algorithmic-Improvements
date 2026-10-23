/**
 * Server-side shaping of data/diabetes.json (written by `python3 -m experiments.export_web` from
 * results/diabetes/benchmarks.json). Same statistics and significance rule as lib/results.ts.
 */
import raw from "@/data/diabetes.json";
import type { RatioCell, RatioComparison } from "@/lib/results";

interface Desc { mean: number; median: number; q25: number; q75: number; n: number }
interface Paired { mean_ratio: number | null; win_rate: number; wilcoxon_p: number; wilcoxon_z: number }
interface Setting {
  n_instances: number;
  title: string;
  dataset: string;
  model: string;
  d: number;
  quality: { test_accuracy: number; test_auc: number; base_rate: number };
  methods: Record<string, Record<string, Desc>>;
  comparisons: Record<string, Record<string, Paired>>;
  walk: { up_steps: Desc; down_steps: Desc; clipped_fraction: Desc };
}
interface Diabetes {
  radii: number[];
  num_samples: number;
  models: Record<string, string>;
  datasets: Record<string, { title: string; features: string[]; n_train: number; n_test: number; positive_rate: number; attribution: string }>;
  settings: Record<string, Setting>;
}

const D = raw as unknown as Diabetes;

export const DIA_DATASETS = ["pima", "early", "brfss", "efron"] as const;
export const DIA_MODELS = ["lr", "mlp", "rf", "gbm"] as const;
const SHORT: Record<string, string> = { lr: "LR", mlp: "MLP", rf: "RF", gbm: "GBM" };
const TREES = new Set(["rf", "gbm"]);

const key = (ds: string, m: string) => `dia_${ds}_${m}`;
const SETTINGS = DIA_DATASETS.flatMap((ds) => DIA_MODELS.map((m) => key(ds, m)));
const short = (s: string) => `${D.datasets[D.settings[s].dataset].title} · ${SHORT[D.settings[s].model]}`;

export const DIA_META = { numSamples: D.num_samples, instances: D.settings[SETTINGS[0]].n_instances, total: SETTINGS.length };

export function diabetesDatasets() {
  return DIA_DATASETS.map((ds) => {
    const info = D.datasets[ds];
    return {
      key: ds,
      title: info.title,
      d: info.features.length,
      nTrain: info.n_train,
      nTest: info.n_test,
      positiveRate: info.positive_rate,
      attribution: info.attribution,
      auc: Object.fromEntries(DIA_MODELS.map((m) => [m, D.settings[key(ds, m)].quality.test_auc])) as Record<string, number>,
    };
  });
}

export const DIA_MODEL_NAMES = D.models;

/** Ratio-plot input: one row per dataset × model, in dataset order. */
export function diabetesComparisons(): RatioComparison[] {
  const pairs: { a: string; b: string; label?: string }[] = [
    { a: "ArcTanh-LIME", b: "LIME (matched budget)" },
    { a: "ArcTanh-LIME, [0,1] asymptotes (no walk)", b: "LIME", label: "ArcTanh-LIME, [0, 1] asymptotes" },
    { a: "ArcTanh-LIME-δ", b: "LIME" },
    { a: "LIME-δ", b: "LIME" },
  ];
  const metrics = ["r0.5", "r1", "r2", "r4", "neighborhood_wmse", "global_mse"];
  return pairs.map(({ a, b, label }) => ({
    key: `${a} vs ${b}`,
    a: label ?? a,
    b,
    rows: SETTINGS.map((s) => {
      const comp = D.settings[s].comparisons[`${a} vs ${b}`];
      const cells: Record<string, RatioCell> = {};
      for (const m of metrics) {
        const c = comp[m];
        cells[m] = { ratio: c.mean_ratio ?? NaN, win: c.win_rate, p: c.wilcoxon_p, z: c.wilcoxon_z };
      }
      return { key: s, title: short(s), sub: D.settings[s].title, cells };
    }),
  }));
}

/** Same rule as everywhere else: p < 0.05, means at least 5% apart, rank test agrees on the direction. */
function significant(c: Paired, better: boolean): boolean {
  const r = c.mean_ratio ?? 1;
  const ok = c.wilcoxon_p < 0.05 && Math.abs(r - 1) >= 0.05 && c.wilcoxon_z < 0 === r < 1;
  return ok && (better ? r < 1 : r > 1);
}

function count(a: string, b: string, metric: string, better: boolean, filter: (s: string) => boolean = () => true) {
  return SETTINGS.filter(filter).filter((s) => significant(D.settings[s].comparisons[`${a} vs ${b}`][metric], better));
}

/** Every distance and aggregate in the benchmark, with the labels used in prose. */
const ALL_METRICS: [string, string][] = [
  ["r0.25", "0.25σ"], ["r0.5", "0.5σ"], ["r1", "1σ"], ["r1.5", "1.5σ"], ["r2", "2σ"], ["r3", "3σ"], ["r4", "4σ"],
  ["neighborhood_wmse", "the LIME neighbourhood"], ["global_mse", "the whole test set"],
];

/** "Setting at distance" for every significant loss of a against b. */
function losses(a: string, b: string, filter: (s: string) => boolean = () => true) {
  return SETTINGS.filter(filter).flatMap((s) =>
    ALL_METRICS.filter(([m]) => significant(D.settings[s].comparisons[`${a} vs ${b}`][m], false)).map(([, label]) => `${short(s)} at ${label}`),
  );
}

function range(values: number[]): [number, number] {
  return [Math.min(...values), Math.max(...values)];
}

export function diabetesHeadline() {
  const A = "ArcTanh-LIME";
  const L = "LIME (matched budget)";
  const ratios = (filter: (s: string) => boolean, metric: string, a = A, b = L) =>
    SETTINGS.filter(filter).map((s) => D.settings[s].comparisons[`${a} vs ${b}`][metric].mean_ratio ?? NaN);
  const isTree = (s: string) => TREES.has(D.settings[s].model);
  const isModel = (m: string) => (s: string) => D.settings[s].model === m;
  const trees = SETTINGS.filter(isTree);
  const NOWALK = "ArcTanh-LIME, [0,1] asymptotes (no walk)";
  const q = (s: string, m: string) => D.settings[s].methods[m].queries.median;
  return {
    total: SETTINGS.length,
    trees: trees.length,
    betterAt05: count(A, L, "r0.5", true),
    betterAt1: count(A, L, "r1", true),
    worseAt2: count(A, L, "r2", false),
    worseAt4: count(A, L, "r4", false),
    worseGlobal: count(A, L, "global_mse", false),
    treesWorseAt3: count(A, L, "r3", false, isTree),
    treesWorseGlobal: count(A, L, "global_mse", false, isTree),
    smoothLosses: losses(A, L, (s) => !isTree(s)),
    lr05: range(ratios(isModel("lr"), "r0.5")),
    mlp05: range(ratios(isModel("mlp"), "r0.5")),
    mlp2: range(ratios(isModel("mlp"), "r2")),
    tree05: range(ratios(isTree, "r0.5")),
    treeGlobal: range(ratios(isTree, "global_mse")),
    treeSteps: range(trees.map((s) => D.settings[s].walk.up_steps.median)),
    treeClipped: range(trees.map((s) => D.settings[s].walk.clipped_fraction.mean)),
    noWalkLosses: losses(NOWALK, "LIME"),
    noWalkBetterAt05: count(NOWALK, "LIME", "r0.5", true),
    noWalkTree05: range(ratios(isTree, "r0.5", NOWALK, "LIME")),
    ensembleTreesWorseAt4: count("ArcTanh-LIME-δ", "LIME", "r4", false, isTree),
    overhead: range(SETTINGS.map((s) => q(s, A) / q(s, "LIME") - 1)),
    ensembleCost: range(SETTINGS.map((s) => q(s, "ArcTanh-LIME-δ") / q(s, "LIME"))),
  };
}
