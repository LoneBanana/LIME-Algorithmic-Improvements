# ArcTanh-LIME — LIME rebuilt from scratch, then bent

This repository is a small research project in three parts:

1. **LIME, implemented from scratch** in pure Python (standard library only — no NumPy, no scikit-learn, no `lime`).
2. **A proposed improvement, ArcTanh-LIME**: instead of fitting a straight line around the explained instance, walk up
   and down the black box's response surface in steps of width ε until the nearest local maximum and minimum, use
   those two values as the asymptotes of tanh⁻¹, and fit the explanation as a tanh⁻¹ regressor. Repeating the procedure
   at points ±δ away and blending the results extends locality toward something *partially global*.
3. **An honest evaluation** (11 benchmark settings, paired statistics, ablations, stability) and a **Next.js website**
   (`web/`) with interactive demos that run a TypeScript port of the exact same algorithm.

Everything below describes what was actually built and measured. Where the idea had to be interpreted, the choice is
stated; where something was changed during development, that is recorded in [§2.4](#24-what-changed-during-development);
where the method does not work, that is in [§7](#7-limitations-and-failure-modes).

---

## TL;DR

- **Locally, it works, and it is the surrogate's shape that helps.** ArcTanh-LIME's surrogate has significantly lower
  error than LIME at 0.5σ and at 1σ from the explained instance in **10 of 11** settings (Wilcoxon signed-rank
  p < 0.05). That holds against LIME given the same query budget (median ratio at 1σ 0.63×, range 0.09–0.81×), and
  against LIME fitted on *exactly the same points* including the walk path (median 0.70×).
- **Farther out, it is mixed.** At 2σ it is significantly *worse* than LIME in 2 of 11 settings (two-moons MLP 1.32×,
  random forest 1.79×). On the whole held-out test set it is better on four of the five 1-D curves and equal or worse
  on all six multi-dimensional settings.
- **On Friedman #1, much of the gain comes from the data box, not the walk.** Confining the walk to the training-data
  range matters there: with an unbounded walk, ArcTanh-LIME is no longer significantly better than LIME on the true
  function (0.91× at 0.5σ, p = 0.2) and keeps a smaller gain on the MLP (0.79×). Elsewhere the box changes nothing.
- **The ±δ ensemble does extend locality.** ArcTanh-LIME-δ beats LIME at 4σ in 10 of 11 settings, but at 3–56× LIME's
  query cost. Much of that extension comes from the anchors themselves: the same ensemble built from *linear* LIME
  members (LIME-δ) extends locality too.
- **For classifiers, the walk is mostly not what helps.** With probability outputs, fixing the asymptotes to [0, 1]
  without any walk is about as faithful (8-D MLP at 1σ: 0.0245 vs 0.0225 MSE) and cheaper. The walk earns its queries
  on multimodal or non-saturating black boxes: replacing it by the global output range is 4.2× worse on the
  trend + ripple curve and 1.6× worse on the bumps curve at 0.5σ.
- **It fails on piecewise-constant models.** On a random forest the finite-difference walk stalls on plateaus after a
  median of 2 steps. The asymptotes are then far too tight (52% of the regression targets fall outside them), fidelity
  beyond 2σ is worse than LIME, and attributions become unstable across random seeds (cosine 0.59 vs 0.99).
- **Attributions are not better gradient estimates.** The cosine similarity between the attribution and the true
  gradient at x₀ is essentially unchanged (e.g. 0.914 vs 0.914; 0.833 vs 0.826).

---

## Contents

1. [Background: LIME and its local-linearity problem](#1-background-lime-and-its-local-linearity-problem)
2. [The proposed method](#2-the-proposed-method)
3. [Implementation (from scratch)](#3-implementation-from-scratch)
4. [Experimental setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Ablations: what drives the gains](#6-ablations-what-drives-the-gains)
7. [Limitations and failure modes](#7-limitations-and-failure-modes)
8. [Implications and next steps](#8-implications-and-next-steps)
9. [Reproducing everything](#9-reproducing-everything)
10. [The website](#10-the-website)
11. [Repository layout](#11-repository-layout)
12. [References](#12-references)

---

## 1. Background: LIME and its local-linearity problem

LIME (Ribeiro, Singh & Guestrin, 2016) explains one prediction f(x₀) of a black-box model f as follows:

1. draw N perturbations z ~ N(x₀, diag(σ²)), with σ the training standard deviation of each feature (the first
   "perturbation" is x₀ itself);
2. query the black box on every perturbation;
3. weight each perturbation by the kernel π(z) = √exp(−d²/w²), where d is the Euclidean distance to x₀ in standardised
   units and w = 0.75·√(number of features);
4. fit a weighted ridge regression (α = 1) of f on the standardised features.

The regression coefficients are the explanation. The surrogate is a **straight line (hyperplane)**. Wherever the model
curves inside the sampled neighbourhood, the line is a compromise. It is too steep in one place and too flat in
another, and it is unbounded at the edges even when the model's output (a probability, say) is bounded.

## 2. The proposed method

### 2.1 The idea, as proposed

> LIME suffers from local errors largely because a linear approximator is used. To solve this, use a tanh⁻¹(x)
> regressor. The sample point to be analysed is mapped to the corresponding curve in feature space and serves as the
> "origin" of the tanh⁻¹(x) regressor. Take steps of width ε up that decision curve until the nearest local maximum is
> encountered, and likewise down the lower tail until the first local minimum. A tanh⁻¹(x) regressor is then trained
> with the local minimum/maximum values treated as the asymptotes of tanh⁻¹(x), filling in its β parameters
> accordingly. With a number δ, perform these steps for points ±δ away from the sample and build tanh⁻¹(x) regressors
> many times like this, to extend the locality of LIME's predictions to be partially global.

(Paraphrased from the project brief.)

### 2.2 How it was implemented

All distances are in **standardised units**: z = (x − x₀)/σ, so "1σ" means one training standard deviation per
feature, the scale LIME samples at.

**Step 1 — the origin.** x₀ is the origin of the regressor's coordinate system (z = 0). The usual LIME neighbourhood is
sampled around it (same distribution, kernel and N as LIME).

**Step 2 — the ε-walk** (`atlime/walk.py`). From x₀, repeatedly step a fixed distance ε along the normalised direction
of steepest ascent:

    x_{k+1} = x_k + ε · σ ⊙ ĝ_k / ‖ĝ_k‖,   ĝ_k = central finite-difference gradient (2d queries)

A step is **accepted** only if it raises f by more than `tol_rel · |f(x_k) − f(x₀)|` (default `tol_rel` = 10⁻³). The
first step therefore needs only a strict improvement, while later steps stop once the curve has levelled off relative
to the climb so far. The walk ends at the last accepted point, for one of four reasons:

- `extremum`: the step did not climb, i.e. we reached or crossed the nearest local maximum, or a saturated tail;
- `flat`: zero gradient;
- `bounds`: the training-data box blocks every move;
- `budget`: 40 steps were taken.

The same walk is run downhill to the nearest local minimum; the two walks share the gradient at x₀. The black box is
only *queried* — the method stays model-agnostic.

**Step 3 — asymptotes and the tanh⁻¹ regressor** (`atlime/arctanh.py`). Let lo = f(x_min), hi = f(x_max). They are
widened by a 5% margin, lo ← lo − 0.05(hi − lo) and hi ← hi + 0.05(hi − lo), so that the extreme values themselves
map to finite targets. With c = (hi + lo)/2 and a = (hi − lo)/2, every observed output is mapped to u = (f − c)/a,
clipped to ±(1 − 10⁻³) (outputs beyond the asymptotes are the "clipped targets" reported below), and transformed by
tanh⁻¹:

    t = artanh((f(z) − c) / a)       ≈   β₀ + βᵀz        (kernel-weighted ridge, α = 1)

In this link space the local minimum and maximum *values* are exactly the vertical asymptotes of artanh, so this is
literally a tanh⁻¹ regressor whose asymptotes come from the walk. Mapped back to output space it is a bounded S-curve
with horizontal asymptotes at lo and hi:

    g(x) = c + a · tanh(β₀ + βᵀz)

The regression data are the LIME sample plus the points visited by the two walks (with their kernel weights).

**Step 3b — output-space refinement (default on).** Least squares in link space over-weights points near the
asymptotes, where small output differences become huge link differences. β₀ and β are therefore refined by weighted
least squares on the original outputs, using Levenberg–Marquardt initialised at the link-space solution. The ridge
penalty is α·a²‖β‖², the analogue of LIME's penalty on output-space slopes. The pure link-space fit (`fit="link"`) is
evaluated as an ablation.

**The explanation** reported at x₀ is the local slope of g:

    attribution = a · (1 − tanh²β₀) · β

It has the same units as LIME's coefficients. β itself describes the whole S-curve, and the walk endpoints and the
asymptotes come with it.

**Step 4 — the ±δ ensemble** (`atlime/ensemble.py`). The same procedure runs at x₀ and at the anchors
x₀ ± k·δ·σ_j·e_j, for every feature j and k = 1..rings. That gives 2d + 1 anchors for one ring, with δ = 1σ by
default; anchors are clipped to the data box. The anchor surrogates are blended by a Gaussian partition of unity:

    G(x) = Σ_a p_a(x) g_a(x),    p_a ∝ exp(−‖z − z_a‖² / 2h²),    h = δ/2

The ensemble's attribution at x₀ is the x₀ anchor's own. The analytic gradient of the blend also contains
∂p_a/∂x terms, which are blending artefacts.

**Degenerate cases.** If both walks fail to move (zero variation), the asymptotes fall back to the observed sample
range (`fallback="sample-range"`). If the black box is constant there, the surrogate is constant. Across all benchmark
runs the fallback never triggered (see the walk diagnostics in §5.6).

### 2.3 Interpretation decisions

The proposal leaves room for interpretation in a few places. These are the readings implemented:

| Phrase in the proposal | Implemented as | Alternatives not chosen |
|---|---|---|
| "mapped to the corresponding curve … the origin" | x₀ is the coordinate origin (z = 0); the fitted curve need not pass exactly through f(x₀). Forcing g(x₀) = f(x₀) is the `pin_origin` ablation. | putting x₀ at the inflection point (would force f(x₀) = (hi + lo)/2) |
| "steps of width ε up that decision curve" (d dimensions) | fixed-length steps along the steepest-ascent direction, estimated by central finite differences | walking each feature axis separately; walking along LIME's coefficient vector |
| "until the nearest local maximum is encountered" | until a step stops climbing (relative tolerance), capped at 40 steps and at the data range | strict improvement only (ablation `tol_rel = 0`) |
| "local min/max values treated as asymptotes of tanh⁻¹(x)" | the *output values* at the two extrema are the asymptotes (± a 5% margin) | asymptotes at the extrema *locations* (would make the surrogate diverge at the extrema, the opposite shape) |
| "filling in β parameters of tanh⁻¹(x)" | kernel-weighted ridge on artanh-transformed outputs, then output-space Levenberg–Marquardt | link-space fit only (ablation) |
| "points ±δ away" | ±δ along every feature axis (2d + 1 anchors), Gaussian blending with bandwidth δ/2 | random directions (implemented: `layout="random"`, not benchmarked); nearest-anchor (hard) blending |

### 2.4 What changed during development

For transparency, these changes were made after looking at intermediate results:

1. **Walk stopping rule.** The first version stopped when a step gained less than 10⁻³ × std(f over the LIME sample).
   On the two-moons MLP most instances sit in saturated regions (f ≈ 10⁻⁶), so *both* walks stopped at step 0. The
   method then silently fell back to the sample range. This rule was replaced by the relative-to-climb rule above,
   which lets saturated starts walk while still ending in flat tails. The strict rule (`tol_rel = 0`) gives the same
   fidelity at up to 35% more queries (ablation).
2. **Ensemble attribution.** The first version reported the gradient of the blended surrogate at x₀. In 1-D it flipped
   sign on saturated tails because of the ∂(weights)/∂x terms. It was replaced by the x₀ anchor's own attribution.
3. **Extra controls added after the first ablation round.** Two observations prompted them. First, fixed [0, 1]
   asymptotes matched the walk on the classifiers. Second, pinning the origin helped dramatically at 0.5σ. Two control
   variants were therefore added: "asymptotes = global output range (no walk)" for every setting, and
   "LIME with its intercept pinned to f(x₀)".
4. **Margin.** The 5% asymptote margin was fixed before any experiment and kept for all reported results. The ablation
   shows that the margin-free version — closest to the original description — is slightly *better* (see §6). The
   default was deliberately not re-tuned after seeing test results.
5. **After an independent code review**, which re-derived and re-ran the core maths and found no correctness bugs:
   - the two walks now share their first gradient at x₀, saving 2d queries per explanation; the surrogate is
     unchanged;
   - the test-set metric now excludes the explained instances;
   - the "same data" LIME control was added, because the matched-budget baseline gives LIME random rather than
     walk-placed extra points;
   - an "unbounded walk" ablation was added, because confining the walk to the data box turned out to drive much of
     the Friedman #1 gain;
   - significance markers now also require the rank test to agree with the direction of the mean ratio;
   - the MLP's backpropagation got a proper finite-difference test.

   All results in this README come from the re-run after these changes.

All hyperparameters (ε = 0.1, 40 steps, `tol_rel` = 10⁻³, margin 5%, δ = 1, h = δ/2, N = 1000) were set once,
before the main benchmark, and never tuned per setting.

## 3. Implementation (from scratch)

**No external libraries on the research side.** The Python package and experiments import only the standard library
(`math`, `json`, `dataclasses`, `typing`, `multiprocessing`, `pathlib`, `time`, `os`, `sys`, `unittest`). All linear
algebra, random number generation, models, optimisation and statistics are hand-written. The website necessarily uses
Next.js and React; its algorithm port (`web/lib/core/`) has no dependencies.

| Module | What it contains |
|---|---|
| `atlime/rng.py` | Mulberry32 PRNG + Box–Muller normals (bit-identical to the JS port) |
| `atlime/linalg.py` | Cholesky solver (with graceful regularisation), weighted ridge with unpenalised intercept (= scikit-learn's `Ridge(alpha).fit(X, y, sample_weight=w)`), weighted R² |
| `atlime/lime.py` | **LIME**: Gaussian neighbourhood, LIME kernel, feature selection (`none`, `auto`, `forward_selection`, `highest_weights`), weighted ridge surrogate, score and local prediction |
| `atlime/walk.py` | the **ε-walk** and finite-difference gradients (one-sided at bounds) |
| `atlime/arctanh.py` | **ArcTanh-LIME**: asymptotes, tanh⁻¹ link regression, Levenberg–Marquardt refinement, fallbacks |
| `atlime/ensemble.py` | **±δ anchor ensemble** (axes or random layout), works with any base explainer |
| `atlime/surrogates.py` | linear, tanh and blended surrogates with analytic gradients |
| `atlime/models.py` | black boxes: MLP (tanh layers, Adam, cross-entropy or MSE), CART regression trees and random forest |
| `atlime/datasets.py` | synthetic data: five 1-D curves, two moons, an 8-D classification problem, Friedman #1 |
| `atlime/metrics.py` | fidelity sets, MSE, gradients, cosine, Wilcoxon signed-rank test, bootstrap CI, win rate |
| `tests/test_core.py` | 29 unit tests (all pass: `python3 -m unittest discover -s tests -t .`) |

**Fidelity to the reference LIME.** The implementation follows `lime.lime_tabular.LimeTabularExplainer` in continuous
mode (`discretize_continuous=False`, `sample_around_instance=True`). It uses the same kernel, kernel width, ridge
penalty and first-sample-is-the-instance convention. Known differences:

- the default quartile-discretised mode, categorical features, and the text/image explainers are **not** implemented;
- `highest_weights` feature selection ranks features by |coefficient|. LIME ranks by |coefficient × standardised
  instance value|, which for continuous features favours features far from the training mean;
- the surrogate's features are centred on x₀ rather than on the training mean. The fitted function is identical; only
  the intercept's meaning changes (it equals the local prediction);
- the benchmarks use all features (no feature selection).

**Determinism and the TypeScript port.** Both implementations share the Mulberry32 stream. `web/lib/core/numeric.ts`
reimplements Python's `math.fsum` (Shewchuk) and `math.sumprod` (doubled-precision dot product). `npm run parity`
re-runs 2,550 checks against fixtures written by the Python code: RNG streams; LIME, ArcTanh-LIME and its variants on
every 1-D curve; both ensembles; and the two-moons MLP. The largest absolute deviation is **4.4 × 10⁻¹⁶**, and a
deliberate 10⁻⁶ corruption of one fixture value is caught.

## 4. Experimental setup

**Settings** (`experiments/common.py`). All data are synthetic and generated in code.

| Setting | d | Black box | Output | Held-out quality | Explained instances |
|---|---|---|---|---|---|
| 1-D logistic σ(2x) | 1 | analytic | curve | — | 60 (grid over [−5, 5]) |
| 1-D sine | 1 | analytic | curve | — | 60 |
| 1-D bumps (three Gaussians) | 1 | analytic | curve | — | 60 |
| 1-D staircase (three sigmoid steps) | 1 | analytic | curve | — | 60 |
| 1-D trend + ripple 0.15x + 0.25 sin 2x | 1 | analytic | curve | — | 60 |
| Two moons (noise 0.15) | 2 | MLP 2-24-24-1 | probability | accuracy 0.970 | 60 test points |
| Synthetic classification | 8 | true p(y\|x) | probability | accuracy 0.819 (Bayes) | 40 test points |
| Synthetic classification | 8 | MLP 8-32-1 | probability | accuracy 0.775 | 40 |
| Synthetic classification | 8 | random forest (40 trees, depth 10) | probability | accuracy 0.745 | 40 |
| Friedman #1 (noise 1) | 10 | true function | regression | R² 0.957 | 40 |
| Friedman #1 | 10 | MLP 10-32-1 | regression | R² 0.945 | 40 |

The 8-D classification problem has p(y=1|x) = σ(2 sin 1.5x₁ + 1.5x₂x₃ + 2 tanh 2x₄ − 0.8x₅² + 0.8), with x ~ N(0, I₈);
features x₆–x₈ are irrelevant. Friedman #1 is 10 sin(πx₁x₂) + 20(x₃ − ½)² + 10x₄ + 5x₅ on [0, 1]¹⁰, with 5 irrelevant
features. The 1-D curves treat x as already standardised (σ = 1) on the domain [−6, 6].

**Methods.**

| Method | Description |
|---|---|
| LIME | the baseline (N = 1000) |
| LIME (matched budget) | LIME with N = the number of queries ArcTanh-LIME used for the same instance — "what a LIME user gets for the same cost" |
| LIME (same data) | LIME's weighted ridge fitted on *exactly* ArcTanh-LIME's regression data (its LIME sample plus the walk path, same kernel weights) — only the surrogate's form differs |
| ArcTanh-LIME | proposed method, defaults of §2.4 |
| LIME-δ | the ±δ ensemble with linear LIME members (control for "more anchors") |
| ArcTanh-LIME-δ | the ±δ ensemble with ArcTanh-LIME members |

LIME and ArcTanh-LIME draw from the **same random stream**, so they see the same perturbation sample. ArcTanh-LIME
only adds the walks. The two baselines answer different questions. The matched budget gives LIME extra *random*
draws, which carry little kernel weight in high dimensions. The same-data control gives LIME the walk points
themselves, and isolates the effect of the tanh⁻¹ form.

**Metrics.** Fidelity is measured on black-box queries drawn independently of the explainers:

- *MSE at radius r*: 100 random points at exactly r ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4} standardised units from x₀, clipped
  to the data box (in 1-D the "sphere" is just the two points x₀ ± r). This is the "how far does the explanation stay
  faithful" curve;
- *LIME neighbourhood wMSE*: 500 newly drawn points from LIME's own sampling distribution, kernel-weighted;
- *test-set MSE*: the surrogate evaluated on 400 held-out data points that exclude the explained instances (240 for
  two moons; 241 grid points for 1-D) — the "global" view;
- |g(x₀) − f(x₀)|, cosine(attribution, true gradient) (finite differences on the black box), queries, wall time.

In 1-D the evaluation points are not perfectly disjoint from the explainers' queries. Walk points x₀ ± kε coincide
exactly with sphere points x₀ ± r, and the ensemble anchors x₀ ± 1 with the r = 1 points. An independent check found
that removing these coincidences moves the sphere MSE by at most 1.4%, with no consistent sign.

**Statistics.** Comparisons are paired over the explained instances. Each reports the ratio of mean MSE, the win
rate, a two-sided Wilcoxon signed-rank test (normal approximation) and a bootstrap 95% CI of the mean difference
(in the JSON). In the tables below a ratio is **bold ▼** when p < 0.05, the means differ by ≥ 5% *and* the rank
test agrees on the direction; ▲ marks a significantly worse result by the same rule; anything else is plain. For the
1-D curves the explained instances are a fixed grid, so their p-values describe that grid rather than a random
sample.

## 5. Results

Generated tables live in `results/` (`benchmarks.md`, `ablations.md`, `stability.md`, `summary.md`). The tables below
are taken verbatim from `results/summary.md` (`python3 -m experiments.readme_tables`). Ratios are A / B; lower favours A.

### 5.1 ArcTanh-LIME vs LIME

**Same query budget** — LIME draws as many perturbations as ArcTanh-LIME made queries:

| Setting | 0.5σ | 1σ | 2σ | 4σ | LIME nbhd | test set |
|---|---|---|---|---|---|---|
| 1-D logistic | **0.31×** ▼ | **0.086×** ▼ | **0.13×** ▼ | **0.31×** ▼ | **0.094×** ▼ | **0.55×** ▼ |
| 1-D sine | **0.40×** ▼ | **0.51×** ▼ | **0.37×** ▼ | **0.30×** ▼ | **0.38×** ▼ | **0.18×** ▼ |
| 1-D bumps | **0.19×** ▼ | **0.81×** ▼ | **0.60×** ▼ | **0.27×** ▼ | **0.56×** ▼ | **0.12×** ▼ |
| 1-D staircase | 0.99× | 0.80× | **0.68×** ▼ | **0.21×** ▼ | **0.77×** ▼ | 0.69× |
| 1-D trend + ripple | **0.24×** ▼ | **0.74×** ▼ | **0.60×** ▼ | **0.49×** ▼ | **0.53×** ▼ | **0.51×** ▼ |
| Two moons · MLP | **0.70×** ▼ | **0.75×** ▼ | 1.32× ▲ | 1.03× | **0.81×** ▼ | 1.13× |
| Synthetic 8-D · true p(y\|x) | **0.51×** ▼ | **0.53×** ▼ | **0.69×** ▼ | **0.91×** ▼ | **0.86×** ▼ | 1.00× |
| Synthetic 8-D · MLP | **0.61×** ▼ | **0.61×** ▼ | **0.76×** ▼ | **0.90×** ▼ | **0.88×** ▼ | 1.04× |
| Synthetic 8-D · random forest | **0.40×** ▼ | **0.55×** ▼ | 1.79× ▲ | 1.64× ▲ | 1.98× ▲ | 1.45× ▲ |
| Friedman #1 · true function | **0.65×** ▼ | **0.70×** ▼ | 0.96× | 0.97× | 1.05× ▲ | 0.99× |
| Friedman #1 · MLP | **0.64×** ▼ | **0.63×** ▼ | **0.78×** ▼ | 1.06× ▲ | 1.00× | 1.10× ▲ |

**Same data** — LIME's ridge fitted on exactly ArcTanh-LIME's regression points (its sample plus the walk path), so
only the surrogate differs:

| Setting | 0.5σ | 1σ | 2σ | 4σ | LIME nbhd | test set |
|---|---|---|---|---|---|---|
| 1-D logistic | **0.30×** ▼ | **0.087×** ▼ | **0.13×** ▼ | **0.31×** ▼ | **0.093×** ▼ | **0.55×** ▼ |
| 1-D sine | **0.41×** ▼ | **0.51×** ▼ | **0.37×** ▼ | **0.30×** ▼ | **0.38×** ▼ | **0.18×** ▼ |
| 1-D bumps | **0.19×** ▼ | **0.81×** ▼ | **0.60×** ▼ | **0.27×** ▼ | **0.56×** ▼ | **0.12×** ▼ |
| 1-D staircase | 0.95× | 0.81× | **0.69×** ▼ | **0.22×** ▼ | **0.77×** ▼ | 0.71× |
| 1-D trend + ripple | **0.25×** ▼ | **0.74×** ▼ | **0.59×** ▼ | **0.48×** ▼ | **0.53×** ▼ | **0.51×** ▼ |
| Two moons · MLP | **0.71×** ▼ | **0.75×** ▼ | 1.31× ▲ | 1.03× | **0.81×** ▼ | 1.13× |
| Synthetic 8-D · true p(y\|x) | **0.53×** ▼ | **0.55×** ▼ | **0.71×** ▼ | **0.87×** ▼ | **0.85×** ▼ | 0.96× |
| Synthetic 8-D · MLP | **0.63×** ▼ | **0.63×** ▼ | **0.78×** ▼ | **0.86×** ▼ | **0.87×** ▼ | 1.00× |
| Synthetic 8-D · random forest | **0.40×** ▼ | **0.56×** ▼ | 1.80× ▲ | 1.62× ▲ | 1.97× ▲ | 1.44× ▲ |
| Friedman #1 · true function | **0.67×** ▼ | **0.72×** ▼ | 0.96× | **0.95×** ▼ | 1.04× | 0.97× |
| Friedman #1 · MLP | **0.69×** ▼ | **0.70×** ▼ | **0.87×** ▼ | 1.00× | 0.96× | 1.04× |

The two baselines tell the same story. LIME's error here is bias, not variance: giving it more random samples changes
almost nothing, and giving it the informative walk points helps it only a little (mostly on Friedman #1). The local
gain therefore comes from the surrogate's form and its asymptotes, not from the extra queries. On Friedman #1 those
asymptotes owe much to the data-box confinement of the walk (§6, point 1).

### 5.2 The ±δ ensemble

**ArcTanh-LIME-δ vs LIME:**

| Setting | 0.5σ | 1σ | 2σ | 4σ | LIME nbhd | test set |
|---|---|---|---|---|---|---|
| 1-D logistic | **0.078×** ▼ | **0.0067×** ▼ | **0.0042×** ▼ | **0.11×** ▼ | **0.013×** ▼ | **0.31×** ▼ |
| 1-D sine | **0.17×** ▼ | **0.024×** ▼ | **0.019×** ▼ | **0.16×** ▼ | **0.043×** ▼ | **0.15×** ▼ |
| 1-D bumps | **0.082×** ▼ | **0.071×** ▼ | **0.083×** ▼ | **0.31×** ▼ | **0.085×** ▼ | **0.086×** ▼ |
| 1-D staircase | **0.76×** ▼ | **0.21×** ▼ | **0.07×** ▼ | **0.35×** ▼ | **0.30×** ▼ | **0.35×** ▼ |
| 1-D trend + ripple | **0.11×** ▼ | **0.075×** ▼ | **0.061×** ▼ | **0.46×** ▼ | **0.095×** ▼ | **0.26×** ▼ |
| Two moons · MLP | **0.63×** ▼ | **0.57×** ▼ | **0.39×** ▼ | **0.58×** ▼ | **0.52×** ▼ | **0.56×** ▼ |
| Synthetic 8-D · true p(y\|x) | **0.48×** ▼ | **0.51×** ▼ | **0.51×** ▼ | **0.51×** ▼ | **0.50×** ▼ | **0.70×** ▼ |
| Synthetic 8-D · MLP | **0.60×** ▼ | **0.60×** ▼ | **0.57×** ▼ | **0.56×** ▼ | **0.56×** ▼ | **0.77×** ▼ |
| Synthetic 8-D · random forest | **0.38×** ▼ | **0.44×** ▼ | **0.92×** ▼ | 1.07× | 1.10× | 1.13× |
| Friedman #1 · true function | **0.57×** ▼ | **0.59×** ▼ | **0.45×** ▼ | **0.44×** ▼ | **0.60×** ▼ | **0.56×** ▼ |
| Friedman #1 · MLP | **0.58×** ▼ | **0.56×** ▼ | **0.50×** ▼ | **0.55×** ▼ | **0.56×** ▼ | **0.70×** ▼ |

**But the anchors alone do much of this.** The same ensemble with linear LIME members (LIME-δ vs LIME) is also
significantly better than LIME at 4σ in 9 of 11 settings and on the test set in 10 of 11 (full table in
`results/summary.md`). Like for like (ArcTanh-LIME-δ vs LIME-δ), the tanh⁻¹ members help most within 2σ. On the
multi-dimensional settings the gain shrinks at 4σ and is gone on the test set — except on the random forest, where the
tanh⁻¹ members are markedly worse:

| Setting | 0.5σ | 1σ | 2σ | 4σ | LIME nbhd | test set |
|---|---|---|---|---|---|---|
| Two moons · MLP | **0.63×** ▼ | **0.67×** ▼ | **0.85×** ▼ | **0.75×** ▼ | **0.66×** ▼ | **0.92×** ▼ |
| Synthetic 8-D · true p(y\|x) | **0.51×** ▼ | **0.52×** ▼ | **0.58×** ▼ | **0.89×** ▼ | **0.70×** ▼ | 1.04× |
| Synthetic 8-D · MLP | **0.62×** ▼ | **0.61×** ▼ | **0.65×** ▼ | **0.92×** ▼ | **0.76×** ▼ | 1.04× |
| Synthetic 8-D · random forest | **0.39×** ▼ | **0.44×** ▼ | 0.94× | 2.02× ▲ | 1.44× ▲ | 1.62× ▲ |
| Friedman #1 · true function | **0.60×** ▼ | **0.63×** ▼ | **0.73×** ▼ | 0.98× | 1.13× ▲ | 1.04× |
| Friedman #1 · MLP | **0.61×** ▼ | **0.59×** ▼ | **0.62×** ▼ | 0.95× | **0.90×** ▼ | 1.07× ▲ |

In the 1-D rows (see `results/summary.md`) ArcTanh-LIME-δ beats LIME-δ in every cell except three non-significant
ones (bumps and staircase at 2σ, staircase on the test set).

### 5.3 Error at 1σ and cost

| Setting | LIME | ArcTanh-LIME | LIME-δ | ArcTanh-LIME-δ | median queries: LIME / ArcTanh / LIME-δ / ArcTanh-δ |
|---|---|---|---|---|---|
| 1-D logistic | 0.001812 | 0.0001545 | 0.0004182 | 1.205e-05 | 1,000 / 1,196 / 3,000 / 3,579 |
| 1-D sine | 0.04667 | 0.02363 | 0.008013 | 0.001108 | 1,000 / 1,097 / 3,000 / 3,291 |
| 1-D bumps | 0.04262 | 0.03454 | 0.01304 | 0.003009 | 1,000 / 1,058 / 3,000 / 3,174 |
| 1-D staircase | 0.001232 | 0.0009849 | 0.0003651 | 0.0002589 | 1,000 / 1,194 / 3,000 / 3,597 |
| 1-D trend + ripple | 0.02463 | 0.01813 | 0.007932 | 0.001837 | 1,000 / 1,058 / 3,000 / 3,162 |
| Two moons · MLP | 0.1007 | 0.07556 | 0.08672 | 0.05768 | 1,000 / 1,071 / 5,000 / 5,410 |
| Synthetic 8-D · true p(y\|x) | 0.03266 | 0.01757 | 0.03219 | 0.01669 | 1,000 / 1,724 / 17,000 / 29,436 |
| Synthetic 8-D · MLP | 0.03643 | 0.02249 | 0.03601 | 0.0219 | 1,000 / 1,732 / 17,000 / 29,606 |
| Synthetic 8-D · random forest | 0.0214 | 0.01176 | 0.02148 | 0.009421 | 1,000 / 1,120 / 17,000 / 19,108 |
| Friedman #1 · true function | 1.323 | 0.9177 | 1.243 | 0.7859 | 1,000 / 2,660 / 21,000 / 55,692 |
| Friedman #1 · MLP | 1.705 | 1.043 | 1.618 | 0.9508 | 1,000 / 2,660 / 21,000 / 55,860 |

Each walk step costs 2d + 1 queries, and the two walks share their first gradient. The single-anchor overhead is
therefore +6% (1-D) to +166% (10-D), and the ensemble multiplies everything by 2d + 1 anchors. In pure Python a single
ArcTanh-LIME explanation takes tens of milliseconds and a 10-D ensemble about a second (`seconds` in
`results/benchmarks.json`).

### 5.4 Attribution accuracy

Mean cosine between the attribution and the true gradient at x₀ (smooth black boxes, d ≥ 2; in 1-D the cosine is just a
sign, and a random forest has no useful gradient):

| Setting | LIME | ArcTanh-LIME |
|---|---|---|
| Two moons · MLP | 0.656 | 0.664 |
| Synthetic 8-D · true p(y\|x) | 0.914 | 0.914 |
| Synthetic 8-D · MLP | 0.826 | 0.833 |
| Friedman #1 · true function | 0.975 | 0.982 |
| Friedman #1 · MLP | 0.963 | 0.968 |

Essentially unchanged: the better *surrogate* does not translate into a better *gradient estimate*.

### 5.5 Seed stability

15 instances × 5 seeds (`python3 -m experiments.run_stability`). *Pairwise cosine*: mean cosine similarity between the
attributions from different seeds (1 = identical). *Relative spread*: mean distance to the across-seed mean
attribution, relative to its norm. *Prediction SD*: across-seed standard deviation of the surrogate at 50 fixed points
1σ away.

| Setting | pairwise cosine LIME | ArcTanh-LIME | relative spread LIME | ArcTanh-LIME | prediction SD LIME | ArcTanh-LIME |
|---|---|---|---|---|---|---|
| 1-D bumps | 0.920 | 1.000 | 0.083 | 0.042 | 0.0084 | 0.0036 |
| Two moons · MLP | 0.999 | 0.931 | 0.054 | 0.062 | 0.0119 | 0.0132 |
| Synthetic 8-D · MLP | 0.990 | 0.992 | 0.087 | 0.081 | 0.0092 | 0.0120 |
| Synthetic 8-D · random forest | 0.993 | 0.590 | 0.075 | 0.341 | 0.0056 | 0.0022 |
| Friedman #1 · MLP | 0.994 | 0.996 | 0.067 | 0.057 | 0.1045 | 0.1067 |

Comparable on the smooth models, more stable on the 1-D curve, and clearly less stable on the random forest. The walk
itself is deterministic given f and x₀, so only the perturbation sample varies between seeds.

### 5.6 What the ε-walk did

| Setting | median steps up / down | stop reasons (both walks) | fallbacks | mean share of targets outside (lo, hi) |
|---|---|---|---|---|
| 1-D logistic | 30 / 30 | extremum 62, budget 46, bounds 12 | none 60 | 0.000 |
| 1-D sine | 14.5 / 14.5 | extremum 118, bounds 2 | none 60 | 0.013 |
| 1-D bumps | 8.5 / 11 | extremum 119, bounds 1 | none 60 | 0.032 |
| 1-D staircase | 40 / 40 | budget 72, extremum 42, bounds 6 | none 60 | 0.000 |
| 1-D trend + ripple | 8 / 8 | extremum 120 | none 60 | 0.076 |
| Two moons · MLP | 8 / 6 | extremum 113, budget 6, bounds 1 | none 60 | 0.025 |
| Synthetic 8-D · true p(y\|x) | 22 / 20 | extremum 79, budget 1 | none 40 | 0.000 |
| Synthetic 8-D · MLP | 20.5 / 21 | extremum 77, budget 3 | none 40 | 0.000 |
| Synthetic 8-D · random forest | 2 / 2.5 | extremum 80 | none 40 | 0.521 |
| Friedman #1 · true function | 40 / 40 | budget 72, extremum 8 | none 40 | 0.062 |
| Friedman #1 · MLP | 40 / 40 | budget 76, extremum 4 | none 40 | 0.008 |

On Friedman #1 the function keeps increasing inside [0, 1]¹⁰ along the gradient path, so most walks hit the 40-step
budget or the box. On the random forest they stall almost immediately.

## 6. Ablations: what drives the gains

`python3 -m experiments.run_ablations` varies one design choice at a time on eight settings (and the ensemble on four),
using the same instances and seeds. Ratios are relative to default ArcTanh-LIME; the 0.5σ and 2σ versions and the
ensemble ablations are in `results/summary.md`, and everything else is in `results/ablations.md`.

**Single-anchor ablations at 1σ** (ratio to default ArcTanh-LIME):

| Variant | 1-D sine | 1-D bumps | 1-D trend + ripple | Two moons · MLP | Synthetic 8-D · MLP | Synthetic 8-D · random forest | Friedman #1 · true function | Friedman #1 · MLP |
|---|---|---|---|---|---|---|---|---|
| LIME | 1.98× ▲ | 1.23× ▲ | 1.36× ▲ | 1.33× ▲ | 1.62× ▲ | 1.82× ▲ | 1.44× ▲ | 1.64× ▲ |
| LIME, intercept pinned to f(x0) | 4.78× ▲ | 3.04× ▲ | 3.30× ▲ | 2.77× ▲ | 1.15× | 1.15× | **0.38×** ▼ | **0.63×** ▼ |
| fit = link space only (pure tanh⁻¹ regression) | 1.52× ▲ | 1.08× | 1.16× | 1.34× ▲ | 1.09× ▲ | 1.15× | 2.96× ▲ | 1.09× |
| asymptotes = global output range (no walk) | 1.01× | 1.13× ▲ | 1.28× | 1.00× | 1.09× ▲ | 1.43× ▲ | 1.12× | 1.19× ▲ |
| asymptotes fixed to [0, 1] (no walk) | n/a | n/a | n/a | 1.00× | 1.09× ▲ | 1.47× ▲ | n/a | n/a |
| asymptotes = walk ∪ sample range | 1.02× | 1.12× ▲ | 1.42× ▲ | 1.00× | 1.00× | 1.41× ▲ | 1.27× ▲ | 1.16× ▲ |
| walk not confined to the data box | 1.02× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.35× ▲ | 1.23× ▲ |
| pin origin: g(x0) = f(x0) | 1.14× ▲ | 1.11× ▲ | 1.06× ▲ | 1.34× | **0.59×** ▼ | **0.86×** ▼ | **0.31×** ▼ | **0.38×** ▼ |
| walk points not used in fit | 0.99× | 1.00× | 1.00× | 1.00× | 1.09× ▲ | 1.02× | 1.07× ▲ | 1.26× ▲ |
| ε = 0.05 (80 steps) | 1.01× | 1.01× | 1.00× | 1.01× | **0.93×** ▼ | 1.14× | **0.94×** ▼ | **0.85×** ▼ |
| ε = 0.4 (10 steps) | 0.97× | 0.97× | 0.97× | 1.00× | 1.07× ▲ | 1.32× ▲ | 1.05× ▲ | 1.16× ▲ |
| tol_rel = 0 (strict improvement) | 1.00× | 1.00× | 1.00× | 0.99× | 0.99× | 1.00× | 1.00× | 1.00× |
| margin = 0 | **0.73×** ▼ | **0.90×** ▼ | **0.83×** ▼ | 0.97× | **0.95×** ▼ | 0.99× | 0.98× | **0.95×** ▼ |
| margin = 0.2 | 1.41× ▲ | 1.20× ▲ | 1.23× ▲ | 1.13× ▲ | 1.19× ▲ | 1.04× | 1.08× ▲ | 1.13× ▲ |

What this shows:

1. **Confining the walk to the data box carries much of the Friedman #1 gain.** Without the bounds the walk leaves
   [0, 1]¹⁰, where the polynomial terms keep growing, and the asymptotes become far too wide. The variant is then
   1.35× (true function) and 1.23× (MLP) worse at 1σ than default ArcTanh-LIME. Against LIME, the unbounded variant is
   no longer significantly better on the true function (0.91× at 0.5σ, p = 0.2; 0.94× at 1σ), and keeps a smaller gain
   on the MLP (0.79× and 0.75×). On every other setting the bounds change nothing. (Found by the independent review.)
2. **The literal tanh⁻¹ regressor mostly helps already; the output-space refinement helps more.** With β fitted only in
   link space, error at 0.5σ is 0.49–0.83× LIME's in 7 of 8 settings (all significant). The exception is Friedman's true
   function, where the link-only fit is 1.68× *worse* than LIME. The refinement lowers error further at ≤ 1σ in most
   settings (the link-only fit has up to 3.3× the refined fit's error at 0.5σ). The link-only fit is slightly better at
   2σ on three settings (bumps, trend + ripple, two moons).
3. **The walk matters only when the local output range differs from the global one.** For the probability outputs,
   fixed [0, 1] asymptotes (a "local logistic" LIME) are as good on the two-moons MLP and only 9% worse on the 8-D MLP,
   with fewer queries (1,000 vs 1,732 on the 8-D MLP). On the random forest they are worse within 1σ but about twice
   as good at 2σ and 4σ. On multimodal or non-saturating functions the walk pays off: using the global output range
   instead raises the error 1.13–4.2× (bumps and trend + ripple at 0.5σ, bumps and Friedman MLP at 1σ). Conversely,
   even *without* a walk, a tanh⁻¹ regressor with the global output range as asymptotes beats LIME at 0.5σ in 7 of 8
   settings (0.30–0.78×; trend + ripple is the exception) — the bounded link alone is worth having.
4. **Pinning the origin trades range for locality.** Forcing g(x₀) = f(x₀) cuts error at 0.5σ by up to 17× in 8–10
   dimensions but is worse at 2σ in the multi-dimensional settings (1.05–1.39×). LIME's intercept is biased in high
   dimensions too, and pinning LIME gives a similar local boost — yet like for like the pinned tanh⁻¹ regressor beats
   pinned LIME at 0.5σ (0.17–0.76×) and 1σ (0.24–0.80×), significantly in all eight settings.
5. **Tolerance, ε and walk points barely matter on smooth models.** `tol_rel = 0` gives identical fidelity at more
   queries. Smaller ε helps slightly in 8–10-D at more queries. A larger ε (0.4) helps the random forest beyond 2σ
   (0.44×) by stepping over plateaus.
6. **A 0% margin is slightly better than the default 5%** (closest to the original description; better at 1σ in 5 of 8
   settings, by 5–27%), and 20% is worse.
7. **Ensemble geometry** (`results/summary.md`): a second ring of anchors improves 4σ fidelity in all four ensemble
   settings (0.30–0.85×) at 1.7× the queries. δ = 2 does so in three of four at no extra cost (Friedman: 1.07× worse).
   Both improve test-set fidelity (0.67–0.83×; δ = 2 is not significant on the 1-D sine). A softer blend (h = δ) is
   worse, and a sharper one (h = δ/4) is about the same.

## 7. Limitations and failure modes

- **Piecewise-constant models.** On the random forest, finite differences see plateaus; the walk stops after about
  2 steps, and the asymptotes enclose only a sliver of the model's local range (52% of targets clipped). At 2σ and 4σ
  the error is 1.79× and 1.64× LIME's (1.98× on LIME's own neighbourhood), and attributions are unstable across seeds.
  A larger ε, or asymptotes from the sample range or from [0, 1], fix the fidelity loss (§6). The method as specified
  should not be used on tree ensembles without such a change.
- **Part of the gain comes from the data box, not the walk.** On Friedman #1 the improvement depends on confining the
  walk to the training-data range (§6, point 1). This is legitimate information — LIME also uses training statistics —
  but it is not the "nearest local extremum" mechanism of the proposal.
- **Confident extrapolation.** A bounded S-curve commits to its asymptotes. Where the model turns back (the curved
  boundary of the two moons), that confidence becomes error: 1.32× LIME's MSE at 2σ.
- **Single-direction surrogate.** g is a ridge function: in d dimensions its level sets are hyperplanes, exactly like
  LIME's. It bends along one direction only, so curved decision boundaries are captured only by the ±δ blend.
- **Walk domain vs sampling domain.** Walks are confined to the training-data box, while LIME's Gaussian samples are
  not. On Friedman #1 (inputs in [0, 1]¹⁰) ArcTanh-LIME is therefore no better than LIME on LIME's own neighbourhood
  metric (1.05× and 1.00×).
- **The nearest extremum may be irrelevant.** On trend + ripple, the walk stops at the nearest ripple crest, so far from
  x₀ the S-curve flattens while the model keeps rising. On average it still beats LIME there, but about one instance
  in five loses to LIME at 4σ.
- **Cost.** (2d + 1) queries per walk step; the ensemble repeats everything at 2d + 1 anchors. At d = 10 an
  ArcTanh-LIME-δ explanation used ~56,000 queries.
- **Global fidelity is not achieved.** Even the ensemble is "partially global" at best. On the multi-dimensional
  settings the single-anchor method does not improve test-set fidelity at all.
- **Scope of the evidence.** All black boxes and data are synthetic; there is no real-world dataset (none could be
  bundled without external downloads) and no user study. The results concern surrogate fidelity, not human
  understanding. Hyperparameters were fixed once, not tuned per setting; other choices (kernel width, N, ε) may change
  the picture. The 1-D evaluation points partly coincide with walk points (§4; ≤ 1.4% effect).
- **Implementation scope.** Continuous tabular features only; the discretised LIME mode, categorical features and
  text/image explainers are not implemented.

## 8. Implications and next steps

- **Bounded outputs deserve bounded surrogates.** For classifiers, most of the local gain comes from fitting LIME on
  the tanh⁻¹ scale (with [0, 1] asymptotes this is half the logit). That is a near-free change to existing pipelines.
- **The walk is informative in its own right.** Its endpoints are the nearest points, along steepest ascent and
  descent, where the prediction peaks or bottoms out. That is counterfactual-style information which comes for free
  with the asymptotes. It is not evaluated here.
- **Locality can be widened deliberately.** ±δ anchors trade queries for range in a controlled way.
- **Worth trying next:**
  - walking on a smoothed black box, so the walk crosses tree-ensemble plateaus;
  - anchors placed along the walk path instead of the feature axes, which would scale better with d;
  - choosing between walk asymptotes and [0, 1] automatically;
  - multi-direction (additive) tanh surrogates for curved boundaries;
  - real datasets and user studies.

## 9. Reproducing everything

Requirements: Python ≥ 3.12 (uses `math.sumprod`; developed on 3.14) and, for the website, Node ≥ 22.18 (runs `.ts`
files natively; developed on Node 26).

```bash
python3 -m unittest discover -s tests -t .   # 29 unit tests
python3 -m experiments.run_benchmarks        # main benchmark, ~3 min on 10 cores (trains/caches the models first)
python3 -m experiments.run_ablations         # ablations, ~11 min
python3 -m experiments.run_stability         # seed stability, ~10 s
python3 -m experiments.readme_tables         # results/summary.md (the tables in this README)
python3 -m experiments.export_web            # web/data/*.json (results, MLP weights, parity fixtures)
```

Everything is deterministic: seeds are fixed, and the trained models are cached in `results/models/`. Re-running
reproduces the numbers above exactly.

Using the library directly:

```python
from atlime import ArcTanhLime, ArcTanhConfig, LimeTabular, LimeConfig, DeltaAnchorEnsemble, DeltaConfig, Rng
from atlime import training_scale, training_bounds

scale, bounds = training_scale(X_train), training_bounds(X_train)
lime = LimeTabular(scale, LimeConfig(num_samples=1000)).explain(model, x0, Rng(0))
atl = ArcTanhLime(scale, ArcTanhConfig(), bounds).explain(model, x0, Rng(0))
ens = DeltaAnchorEnsemble(ArcTanhLime(scale, ArcTanhConfig(), bounds), DeltaConfig(delta=1.0), bounds).explain(model, x0, Rng(0))

atl.attribution              # local slope at x0 (same units as lime.attribution)
atl.lo, atl.hi               # asymptotes (walk extrema, 5% margin)
atl.walk_up.end_point        # where the nearest local maximum is
ens.surrogate.predict(x)     # partially-global surrogate
```

`model` is any callable `list[float] -> float`, e.g. a class probability.

## 10. The website

`web/` is a Next.js 16 / React 19 site (static export) that demonstrates the method:

- **How it works** — a four-step interactive walkthrough (sample → walk → asymptotes and link space → ±δ blend);
- **Playground** — five 1-D black boxes, draggable x₀, sliders for ε, δ, kernel width and N, all four explainers,
  the link-space view and live error curves;
- **Two dimensions** — the two-moons MLP (the exact weights trained by the Python code) with the 2-D walk, anchors and
  error maps;
- **Results, ablations, limits** — charts and tables read from the experiment outputs at build time.

The charts are hand-written SVG and canvas, with no chart library. The algorithm runs in the browser via
`web/lib/core/`, the TypeScript port verified by `npm run parity`.

```bash
cd web
npm install
npm run parity   # TS port vs Python reference (2,550 checks)
npm run dev      # http://localhost:3000
npm run build    # static site in web/out/
```

## 11. Repository layout

```
atlime/                 the stdlib-only Python package (LIME, ε-walk, ArcTanh-LIME, ensemble, models, metrics)
tests/                  unit tests
experiments/            benchmark, ablation and stability runners; README tables; website export
results/                generated: benchmarks/ablations/stability JSON + Markdown, raw per-instance records, models
web/                    Next.js site; web/lib/core is the TypeScript port, web/scripts/parity.ts its test
```

## 12. References

- M. T. Ribeiro, S. Singh, C. Guestrin. "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
  KDD 2016. (LIME; the reference implementation is the `lime` Python package.)
- J. H. Friedman. Multivariate Adaptive Regression Splines. *Annals of Statistics* 19(1), 1991. (Benchmark function #1.)
- F. Wilcoxon. Individual Comparisons by Ranking Methods. *Biometrics Bulletin* 1(6), 1945.
- D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. ICLR 2015. (Optimiser of the MLP black boxes.)
- L. Breiman. Random Forests. *Machine Learning* 45, 2001. (The random-forest black box.)
