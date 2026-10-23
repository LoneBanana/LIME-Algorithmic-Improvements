/**
 * TypeScript port of the `atlime` Python package (only what the website needs).
 * Every function mirrors its Python counterpart operation-for-operation; the
 * parity test (`npm run parity`) checks the two against shared fixtures.
 */
export { Rng } from "./rng.ts";
export { fsum, sigmoid, sumprod } from "./numeric.ts";
export { solveSpd, weightedR2, weightedRidge } from "./linalg.ts";
export { BlendedSurrogate, LinearSurrogate, standardize, TanhSurrogate } from "./surrogates.ts";
export type { Surrogate } from "./surrogates.ts";
export { DEFAULT_LIME, defaultKernelWidth, LimeTabular, limeKernel, sampleNeighborhood } from "./lime.ts";
export type { BlackBox, LimeConfig, LimeResult, Neighborhood } from "./lime.ts";
export { epsilonWalk, finiteDifferenceGradient } from "./walk.ts";
export type { Bounds, StopReason, WalkResult } from "./walk.ts";
export { ArcTanhLime, DEFAULT_ARCTANH, fitTanhOutput } from "./arctanh.ts";
export type { ArcTanhConfig, ArcTanhResult } from "./arctanh.ts";
export { anchorPoints, DEFAULT_DELTA, DeltaAnchorEnsemble } from "./ensemble.ts";
export type { DeltaConfig, EnsembleResult } from "./ensemble.ts";
export { gbmPredictor, mlpPredictor, ONE_D_DOMAIN, ONE_D_FUNCTIONS } from "./blackboxes.ts";
export type { GbmWeights, MlpWeights, OneDName, TreeWeights } from "./blackboxes.ts";
