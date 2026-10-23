/**
 * Typed wrapper around the C++/WebGL2 renderer (viz/src/engine.cpp, compiled to
 * public/viz/viz.{js,wasm} by viz/build.sh). Arrays are copied into the WASM heap
 * for each call; the engine keeps its own copies.
 */

export const PROP = { ALPHA: 0, REVEAL: 1, MORPH: 2, SCALE: 3, EMISSIVE: 4, TIME: 5, OFFSET_Z: 6, VISIBLE: 7 } as const;
export const CAM = { THETA: 10, PHI: 11, DIST: 12, TARGET_Z: 13, ORTHO_SIZE: 14 } as const;
export const EASE = { LINEAR: 0, SMOOTH: 1, CUBIC: 2, THERE_AND_BACK: 3 } as const;
export const CAMERA = -1;

export type RGBA = [number, number, number, number];

interface EmModule {
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  HEAPF32: Float32Array;
  cwrap(name: string, ret: string | null, args: string[]): (...a: unknown[]) => number;
}

declare global {
  interface Window {
    createVizModule?: (opts: Record<string, unknown>) => Promise<EmModule>;
  }
}

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

let loading: Promise<EmModule> | null = null;

function loadModule(): Promise<EmModule> {
  if (loading) return loading;
  loading = new Promise<EmModule>((resolve, reject) => {
    const start = () => {
      window.createVizModule!({
        locateFile: (p: string) => `${BASE}/viz/${p}`,
        print: () => undefined,
        printErr: (s: string) => console.warn("[viz]", s),
      }).then(resolve, reject);
    };
    if (window.createVizModule) return start();
    const script = document.createElement("script");
    script.src = `${BASE}/viz/viz.js`;
    script.async = true;
    script.onload = start;
    script.onerror = () => reject(new Error("could not load the WebAssembly renderer"));
    document.head.appendChild(script);
  });
  return loading;
}

const f = "number";

export class Viz {
  private m: EmModule;
  private fn: Record<string, (...a: unknown[]) => number> = {};

  private constructor(m: EmModule) {
    this.m = m;
    const sig: Record<string, [string | null, string[]]> = {
      viz_init: [f, ["string"]],
      viz_resize: [null, [f, f, f, f]],
      viz_clear: [null, []],
      viz_remove: [null, [f]],
      viz_set_paused: [null, [f]],
      viz_set_background: [null, [f, f, f]],
      viz_set_clip: [null, [f, f, f, f, f, f, f]],
      viz_add_surface: [f, [f, f, f, f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_isosurface: [f, [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_isosurface_4d: [f, [f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_isolines: [f, [f, f, f, f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_points: [f, [f, f, f, f, f]],
      viz_points_target: [null, [f, f]],
      viz_add_tube: [f, [f, f, f, f, f, f, f]],
      viz_add_arrow: [f, [f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_plane_z: [f, [f, f, f, f, f, f, f, f, f]],
      viz_add_box: [f, [f, f, f, f, f, f, f, f, f, f, f]],
      viz_add_floor_grid: [f, [f, f, f, f, f, f, f, f, f, f, f]],
      viz_set: [null, [f, f, f]],
      viz_set_unlit: [null, [f, f]],
      viz_set_scale_origin: [null, [f, f, f, f]],
      viz_set_clip_object: [null, [f, f]],
      viz_tween: [null, [f, f, f, f, f, f, f]],
      viz_clear_tweens: [null, []],
      viz_set_time: [null, [f]],
      viz_time: [f, []],
      viz_play: [null, [f, f]],
      viz_camera: [null, [f, f, f, f, f, f]],
      viz_camera_ortho: [null, [f, f]],
      viz_auto_rotate: [null, [f]],
      viz_orbit: [null, [f, f]],
      viz_zoom: [null, [f]],
      viz_project: [null, [f, f, f, f]],
      viz_pick: [f, [f, f, f]],
      viz_tour_begin: [f, [f, f, f, f, f, f, f, f]],
      viz_tour_walk: [f, [f, f, f, f, f, f]],
      viz_tour_arrows: [f, [f, f, f, f]],
      viz_tour_axes: [f, [f]],
      viz_tour_speed: [null, [f]],
      viz_tour_time: [null, [f]],
      viz_tour_axis_tip: [null, [f, f]],
      viz_frame_cost_ms: [f, []],
    };
    for (const [name, [ret, args]] of Object.entries(sig)) this.fn[name] = m.cwrap(name, ret, args);
  }

  static async create(canvasSelector: string): Promise<Viz> {
    const viz = new Viz(await loadModule());
    if (!viz.fn.viz_init(canvasSelector)) throw new Error("WebGL2 is not available in this browser");
    return viz;
  }

  /** Copy numbers into the WASM heap; the caller frees the pointer. */
  private alloc(values: ArrayLike<number>): number {
    const ptr = this.m._malloc(Math.max(values.length, 1) * 4);
    this.m.HEAPF32.set(values instanceof Float32Array ? values : Float32Array.from(values as ArrayLike<number>), ptr >> 2);
    return ptr;
  }

  private withArrays<T>(arrays: (ArrayLike<number> | null)[], run: (ptrs: number[]) => T): T {
    const ptrs = arrays.map((a) => (a ? this.alloc(a) : 0));
    try {
      return run(ptrs);
    } finally {
      for (const p of ptrs) if (p) this.m._free(p);
    }
  }

  resize(pxW: number, pxH: number, cssW: number, cssH: number) {
    this.fn.viz_resize(pxW, pxH, cssW, cssH);
  }
  clear() {
    this.fn.viz_clear();
  }
  remove(id: number) {
    this.fn.viz_remove(id);
  }
  setPaused(p: boolean) {
    this.fn.viz_set_paused(p ? 1 : 0);
  }
  background(r: number, g: number, b: number) {
    this.fn.viz_set_background(r, g, b);
  }
  clip(on: boolean, x0 = 0, x1 = 0, y0 = 0, y1 = 0, z0 = 0, z1 = 0) {
    this.fn.viz_set_clip(on ? 1 : 0, x0, x1, y0, y1, z0, z1);
  }

  surface(hA: Float32Array, hB: Float32Array | null, nu: number, nv: number, box: [number, number, number, number],
    zscale: number, c: RGBA, checker = false): number {
    return this.withArrays([hA, hB], ([a, b]) =>
      this.fn.viz_add_surface(a, b, nu, nv, box[0], box[1], box[2], box[3], zscale, ...c, checker ? 1 : 0));
  }

  isosurface(field: Float32Array, n: [number, number, number], iso: number, lo: number[], hi: number[], c: RGBA): number {
    return this.withArrays([field], ([p]) =>
      this.fn.viz_add_isosurface(p, n[0], n[1], n[2], iso, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ...c));
  }

  isosurface4d(fields: Float32Array, nt: number, n: [number, number, number], iso: number, lo: number[], hi: number[],
    c: RGBA): number {
    return this.withArrays([fields], ([p]) =>
      this.fn.viz_add_isosurface_4d(p, nt, n[0], n[1], n[2], iso, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ...c));
  }

  isolines(field: Float32Array, nx: number, ny: number, iso: number, box: [number, number, number, number], z: number,
    radius: number, c: RGBA): number {
    return this.withArrays([field], ([p]) =>
      this.fn.viz_add_isolines(p, nx, ny, iso, box[0], box[1], box[2], box[3], z, radius, ...c));
  }

  points(xyz: Float32Array, rgba: Float32Array, radius: number, pickable = false): number {
    return this.withArrays([xyz, rgba], ([p, c]) => this.fn.viz_add_points(p, c, xyz.length / 3, radius, pickable ? 1 : 0));
  }

  pointsTarget(id: number, xyz: Float32Array) {
    this.withArrays([xyz], ([p]) => this.fn.viz_points_target(id, p));
  }

  tube(xyz: number[], radius: number, c: RGBA): number {
    return this.withArrays([xyz], ([p]) => this.fn.viz_add_tube(p, xyz.length / 3, radius, ...c));
  }

  arrow(from: number[], to: number[], radius: number, c: RGBA): number {
    return this.fn.viz_add_arrow(from[0], from[1], from[2], to[0], to[1], to[2], radius, ...c);
  }

  planeZ(z: number, box: [number, number, number, number], c: RGBA): number {
    return this.fn.viz_add_plane_z(z, box[0], box[1], box[2], box[3], ...c);
  }

  box(lo: number[], hi: number[], radius: number, c: RGBA): number {
    return this.fn.viz_add_box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], radius, ...c);
  }

  floorGrid(box: [number, number, number, number], z: number, step: number, radius: number, c: RGBA): number {
    return this.fn.viz_add_floor_grid(box[0], box[1], box[2], box[3], z, step, radius, ...c);
  }

  set(id: number, prop: number, value: number) {
    this.fn.viz_set(id, prop, value);
  }
  unlit(id: number, on = true) {
    this.fn.viz_set_unlit(id, on ? 1 : 0);
  }
  scaleOrigin(id: number, p: number[]) {
    this.fn.viz_set_scale_origin(id, p[0], p[1], p[2]);
  }
  clipObject(id: number, on: boolean) {
    this.fn.viz_set_clip_object(id, on ? 1 : 0);
  }
  tween(target: number, prop: number, from: number, to: number, start: number, dur: number, easing: number = EASE.SMOOTH) {
    this.fn.viz_tween(target, prop, from, to, start, dur, easing);
  }
  clearTweens() {
    this.fn.viz_clear_tweens();
  }
  setTime(t: number) {
    this.fn.viz_set_time(t);
  }
  time(): number {
    return this.fn.viz_time();
  }
  play(on: boolean, speed = 1) {
    this.fn.viz_play(on ? 1 : 0, speed);
  }
  camera(theta: number, phi: number, dist: number, target: number[]) {
    this.fn.viz_camera(theta, phi, dist, target[0], target[1], target[2]);
  }
  ortho(on: boolean, size = 4) {
    this.fn.viz_camera_ortho(on ? 1 : 0, size);
  }
  autoRotate(speed: number) {
    this.fn.viz_auto_rotate(speed);
  }
  orbit(dTheta: number, dPhi: number) {
    this.fn.viz_orbit(dTheta, dPhi);
  }
  zoom(factor: number) {
    this.fn.viz_zoom(factor);
  }

  /** Moving average of the engine's CPU time per frame, in ms (diagnostics). */
  frameCostMs(): number {
    return this.fn.viz_frame_cost_ms();
  }

  /** CSS-pixel screen position of a world point: [x, y, visible]. */
  project(p: number[]): [number, number, boolean] {
    const ptr = this.m._malloc(12);
    try {
      this.fn.viz_project(p[0], p[1], p[2], ptr);
      const h = this.m.HEAPF32;
      return [h[ptr >> 2], h[(ptr >> 2) + 1], h[(ptr >> 2) + 2] > 0.5];
    } finally {
      this.m._free(ptr);
    }
  }

  /** Nearest pickable point: { object, index } or null. */
  pick(x: number, y: number, maxPx = 14): { object: number; index: number } | null {
    const r = this.fn.viz_pick(x, y, maxPx);
    if (r < 0) return null;
    return { object: r >> 20, index: r & 0xfffff };
  }

  tourBegin(X: Float32Array, n: number, d: number, rgba: Float32Array, radius: number, keyframes: Float32Array,
    nk: number, scale: number): number {
    return this.withArrays([X, rgba, keyframes], ([x, c, k]) => this.fn.viz_tour_begin(x, n, d, c, radius, k, nk, scale));
  }
  tourWalk(rows: Float32Array, m: number, c: RGBA): number {
    return this.withArrays([rows], ([p]) => this.fn.viz_tour_walk(p, m, ...c));
  }
  tourArrows(origin: Float32Array, vecs: Float32Array, k: number, colors: Float32Array): number {
    return this.withArrays([origin, vecs, colors], ([o, v, c]) => this.fn.viz_tour_arrows(o, v, k, c));
  }
  tourAxes(len: number): number {
    return this.fn.viz_tour_axes(len);
  }
  tourSpeed(s: number) {
    this.fn.viz_tour_speed(s);
  }
  tourAxisTip(j: number): number[] {
    const ptr = this.m._malloc(12);
    try {
      this.fn.viz_tour_axis_tip(j, ptr);
      const h = this.m.HEAPF32;
      return [h[ptr >> 2], h[(ptr >> 2) + 1], h[(ptr >> 2) + 2]];
    } finally {
      this.m._free(ptr);
    }
  }
}
