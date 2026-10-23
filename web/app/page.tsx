import AblationExplorer from "@/components/AblationExplorer";
import ClientOnly from "@/components/ClientOnly";
import FidelityMultiples from "@/components/FidelityMultiples";
import Lab3D from "@/components/Lab3D";
import MethodSteps from "@/components/MethodSteps";
import MoonsExplorer from "@/components/MoonsExplorer";
import OneDExplorer from "@/components/OneDExplorer";
import RatioPlot from "@/components/RatioPlot";
import ThemeToggle from "@/components/ThemeToggle";
import {
  ablationComparison, ablationTables, ablationValue, ablationWins, benchComparison, benchMethod, costRows, fidelityPanels, gradientRows, headline,
  META, QUALITY, RATIO_METRICS, ratioComparisons, SETTING_NAMES, STABILITY_META, stabilityRows, walkDiagnostics,
} from "@/lib/results";
import { DIA_META, DIA_MODEL_NAMES, DIA_MODELS, diabetesComparisons, diabetesDatasets, diabetesHeadline } from "@/lib/diabetes";
import { fmtRatio } from "@/lib/plot";

const fx = (v: number) => fmtRatio(v);
const pct = (v: number) => `${Math.round(v * 100)}%`;
const names = (keys: string[]) => (keys.length ? keys.map((k) => SETTING_NAMES[k].title).join(", ") : "none");
const ratio = (s: string, a: string, b: string, m: string) => benchComparison(s, a, b, m).mean_ratio ?? NaN;

export default function Page() {
  const h = headline();
  const hs = headline("LIME (same data)");
  const UNB = "walk not confined to the data box";
  const unbO = ablationComparison("friedman_oracle", UNB, "LIME", "r0.5");
  const unbM = ablationComparison("friedman_mlp", UNB, "LIME", "r0.5");
  const rfWalk = walkDiagnostics("synth_rf");
  const stab = Object.fromEntries(stabilityRows().map((r) => [r.key, r.values]));
  const at = "ArcTanh-LIME";
  const lm = "LIME (matched budget)";
  const abl = (s: string, label: string, m: string) => ablationValue(s, label, m);
  const D = "ArcTanh-LIME (default)";
  const quality = QUALITY;
  const linkWins = ablationWins("fit = link space only (pure tanh⁻¹ regression)", "LIME", "r0.5");
  const globalWins = ablationWins("asymptotes = global output range (no walk)", "LIME", "r0.5");
  const pinWins = ablationWins("pin origin: g(x0) = f(x0)", "LIME, intercept pinned to f(x0)", "r1");
  const dh = diabetesHeadline();
  const span = ([lo, hi]: [number, number]) => `${fx(lo)}–${fx(hi)}`;

  return (
    <>
      <header className="topbar">
        <div className="wrap">
          <a href="#top" className="brand" style={{ textDecoration: "none" }}>ArcTanh<span>-</span>LIME</a>
          <nav className="nav" aria-label="Sections">
            <a href="#idea">Idea</a>
            <a href="#method">Method</a>
            <a href="#playground">Playground</a>
            <a href="#two-d">2-D</a>
            <a href="#lab">3-D lab</a>
            <a href="#results">Results</a>
            <a href="#diabetes">Diabetes</a>
            <a href="#ablations">Ablations</a>
            <a href="#limits">Limits</a>
            <a href="#reproduce">Reproduce</a>
          </nav>
          <ThemeToggle />
        </div>
      </header>

      <main className="wrap" id="top">
        <section className="hero">
          <p className="kicker">Research note · LIME rebuilt from scratch</p>
          <h1>
            Bending LIME with a <em>tanh⁻¹</em> regressor
          </h1>
          <p className="lede">
            LIME explains a prediction by fitting a straight line around it, so it is wrong wherever the model curves.
            This project rebuilds LIME from scratch and tests a different surrogate: walk up and down the model&apos;s
            response surface to its nearest local maximum and minimum, use those values as the asymptotes of tanh⁻¹,
            fit the explanation in that bounded space — and repeat at ±δ to stretch locality toward the global.
          </p>
          <div className="byline">
            <span>Python standard library only — no NumPy, no scikit-learn, no lime</span>
            <span>11 synthetic + {dh.total} diabetes-risk settings · paired statistics · full ablations</span>
            <span>Interactive figures run a TypeScript port that matches the Python to 4×10⁻¹⁶</span>
          </div>
          <div className="stats">
            <div className="card stat">
              <div className="label">More faithful than LIME at 1σ</div>
              <div className="value">{h.betterAt1.length} / {h.total}</div>
              <div className="note">
                settings vs equal-budget LIME ({hs.betterAt1.length}/{hs.total} vs LIME fitted on the same points),
                Wilcoxon p &lt; 0.05
              </div>
            </div>
            <div className="card stat">
              <div className="label">Median MSE ratio at 0.5σ</div>
              <div className="value">{fx(h.medianRatio05)}</div>
              <div className="note">of LIME&apos;s error; better in {h.betterAt05.length} of {h.total} settings</div>
            </div>
            <div className="card stat">
              <div className="label">Less faithful than LIME at 2σ</div>
              <div className="value">{h.worseAt2.length} / {h.total}</div>
              <div className="note">{names(h.worseAt2)}</div>
            </div>
            <div className="card stat">
              <div className="label">Extra black-box queries</div>
              <div className="value">+{Math.round(h.overheadRange[0] * 100)}–{Math.round(h.overheadRange[1] * 100)}%</div>
              <div className="note">single anchor; the ±δ ensemble costs {Math.round(h.deltaCostRange[0])}–{Math.round(h.deltaCostRange[1])}× LIME</div>
            </div>
          </div>
        </section>

        <section id="idea" className="prose">
          <h2>The idea</h2>
          <p>
            LIME (Ribeiro, Singh &amp; Guestrin, 2016) explains a single prediction f(x₀) of a black-box model by
            perturbing x₀, querying the model on the perturbations, and fitting a kernel-weighted linear regression to
            the answers. The regression coefficients are the explanation. The weak spot is the word <i>linear</i>:
            whenever the model curves inside the sampled neighbourhood, the line is a compromise — too steep in one
            place, too flat in another, and unbounded at the edges even when the model&apos;s output (a probability, for
            instance) is not.
          </p>
          <p>The modification studied here replaces the line with a tanh⁻¹ regressor built in four moves:</p>
          <ol>
            <li>map the instance onto the model&apos;s response curve; it becomes the <b>origin</b> of the regressor;</li>
            <li>take steps of width <b>ε</b> up the curve until the nearest local maximum, and down it until the
              nearest local minimum;</li>
            <li>treat those two extreme values as the <b>asymptotes</b> of tanh⁻¹ and fit its β parameters;</li>
            <li>repeat the whole procedure at points <b>±δ</b> away and combine the regressors, extending LIME&apos;s
              locality toward something <i>partially global</i>.</li>
          </ol>
          <p>
            Every piece — LIME itself, the walk, both regressors, the black-box models (an MLP and a random forest),
            the data generators and the statistics — is implemented from scratch in the Python standard library. The
            figures on this page run a line-by-line TypeScript port of that code, checked against the Python outputs
            by an automated parity test.
          </p>
        </section>

        <section id="method">
          <div className="prose">
            <h2>How it works</h2>
            <p>Step through the procedure on a curve with several bumps. Every number in the text is live.</p>
          </div>
          <ClientOnly minHeight={640}>
            <MethodSteps />
          </ClientOnly>
          <div className="prose">
            <h3>The whole method in five lines</h3>
          </div>
          <div className="equation" role="math" aria-label="Method equations">
            walk&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x<sub>k+1</sub> = x<sub>k</sub> ± ε · σ ⊙ ĝ<sub>k</sub> / ‖ĝ<sub>k</sub>‖,
            &nbsp; ĝ = finite-difference gradient (standardised units); stop when a step stops climbing
            <br />
            asymptotes&nbsp;&nbsp;lo = f(x<sub>min</sub>), hi = f(x<sub>max</sub>), &nbsp;c = (hi + lo)/2, &nbsp;a = (hi − lo)/2
            <br />
            regressor&nbsp;&nbsp;&nbsp;artanh((f(x) − c)/a) ≈ β₀ + βᵀz, &nbsp; z = (x − x₀)/σ &nbsp;(kernel-weighted ridge)
            <br />
            surrogate&nbsp;&nbsp;&nbsp;g(x) = c + a · tanh(β₀ + βᵀz), &nbsp; attribution = a·(1 − tanh²β₀)·β
            <br />
            ensemble&nbsp;&nbsp;&nbsp;&nbsp;G(x) = Σ<sub>k</sub> w<sub>k</sub>(x) g<sub>k</sub>(x), &nbsp;w<sub>k</sub> ∝ exp(−‖z − z<sub>k</sub>‖² / 2h²), &nbsp;anchors x₀ ± δσ<sub>j</sub>e<sub>j</sub>
          </div>
          <div className="prose">
            <div className="callout neutral">
              <b>Interpretation choices.</b> The idea leaves some room, so these are the readings implemented (all
              documented in the README):
              <ul style={{ margin: "8px 0 0", paddingLeft: 20 }}>
                <li>&ldquo;Up the curve&rdquo; in d dimensions is steepest ascent with central finite differences
                  (2d + 1 queries per step); ε is measured in training standard deviations.</li>
                <li>A step counts as climbing if it gains more than 0.1% of the climb so far, so a saturating
                  sigmoid&apos;s flat tail ends the walk; walks are capped at 40 steps (4σ) and stay inside the data range.</li>
                <li>The asymptotes are the <i>output values</i> at the two extrema — vertical asymptotes of artanh in
                  link space, horizontal asymptotes of tanh in output space — widened by a 5% margin so the extreme
                  values themselves stay finite (outputs beyond them are clipped).</li>
                <li>β is first fitted literally as a tanh⁻¹ regressor (weighted ridge on artanh-transformed outputs),
                  then refined by weighted least squares in output space (Levenberg–Marquardt). Both are evaluated.</li>
                <li>&ldquo;±δ&rdquo; in d dimensions means ±δ along every feature axis (2d + 1 anchors), blended with
                  Gaussian weights of bandwidth δ/2.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="playground">
          <div className="prose">
            <h2>Playground</h2>
            <p>
              Five 1-D black boxes, all four explainers, recomputed as you drag. Things worth trying:{" "}
              on <b>logistic</b>, the S-curve is the right family and the tanh surrogate is essentially exact while
              the line overshoots both plateaus; on <b>trend + ripple</b>, the walk stops at the nearest ripple crest,
              so the asymptotes are tighter than the trend — far from x₀ the orange curve flattens while the model
              keeps rising; on <b>staircase</b>, every plateau still climbs a little, so the walk crosses all of them.
            </p>
          </div>
          <ClientOnly minHeight={1150}>
            <OneDExplorer />
          </ClientOnly>
        </section>

        <section id="two-d">
          <div className="prose">
            <h2>In two dimensions</h2>
            <p>
              The same procedure on an MLP trained (from scratch) on the two-moons data
              ({pct(quality.moons_mlp.test_accuracy)} test accuracy). In 2-D the ε-walk is a genuine curve: it follows
              the probability gradient into the interior of one moon and down into the other. Note that both single
              surrogates are <i>ridge functions</i> — their level sets are straight lines — so neither can follow the
              curved decision boundary; the ±δ blend can, approximately.
            </p>
          </div>
          <ClientOnly minHeight={980}>
            <MoonsExplorer />
          </ClientOnly>
        </section>

        <section id="lab">
          <div className="prose">
            <h2>3-D lab: any dataset, in 2, 3 or n dimensions</h2>
            <p>
              A manim-style renderer written in C++ against OpenGL ES 3 and compiled to WebAssembly (WebGL2) draws
              the black box, the explained instance, LIME and ArcTanh-LIME for real diabetes-risk data — or for any
              CSV you drop in, which is parsed, modelled and explained without leaving your browser. Four views:
            </p>
            <ul>
              <li><b>Risk surface</b> — two axes through x₀ and the model&apos;s probability as height; watch the LIME
                plane transform into the ArcTanh-LIME S-surface between the walk&apos;s asymptotes.</li>
              <li><b>Feature space (3-D)</b> — the model&apos;s P = 0.5 boundary as an iso-surface against the two
                surrogates&apos; decision planes; optionally swept along a fourth direction, so a 4-D slice plays out
                over time.</li>
              <li><b>Feature space (2-D)</b> — the same as contours, with optional 0.25/0.75 levels.</li>
              <li><b>Grand tour</b> — every feature at once: the projection rotates smoothly through n-dimensional
                space, carrying the ε-walk and both attribution vectors with it.</li>
            </ul>
            <p className="small muted">
              Axes default to the explanation direction (the surrogate&apos;s β) plus principal components orthogonal to
              it; slices hold every other feature at x₀&apos;s value. Data points are projections, so in 3-D they do not
              lie exactly on the slice.
            </p>
          </div>
          <ClientOnly minHeight={760}>
            <Lab3D />
          </ClientOnly>
        </section>

        <section id="results">
          <div className="prose">
            <h2>Results</h2>
            <p>
              Eleven settings: five analytic 1-D curves; the two-moons MLP; an 8-D synthetic classification problem
              explained through its true probability function, an MLP ({pct(quality.synth_mlp.test_accuracy)} accuracy;
              Bayes-optimal {pct(quality.synth_oracle.test_accuracy)}) and a random forest
              ({pct(quality.synth_rf.test_accuracy)}); and the 10-D Friedman #1 regression function, both exact and
              as an MLP (test R² {quality.friedman_mlp.test_r2.toFixed(2)}). Every explainer uses N = {META.numSamples.toLocaleString("en-US")}{" "}
              perturbations, LIME and ArcTanh-LIME see the <i>same</i> perturbation sample, and fidelity is measured on
              black-box queries drawn independently of the explainers. Because the walk costs extra queries, there are
              two fair baselines. &ldquo;LIME (matched budget)&rdquo; gets exactly as many queries as ArcTanh-LIME
              used. &ldquo;LIME (same data)&rdquo; is fitted on exactly ArcTanh-LIME&apos;s points, walk included, so
              that only the shape of the surrogate differs.
            </p>
          </div>
          <ClientOnly minHeight={900}>
            <FidelityMultiples panels={fidelityPanels()} />
          </ClientOnly>
          <ClientOnly minHeight={520}>
            <RatioPlot comparisons={ratioComparisons()} metrics={RATIO_METRICS} />
          </ClientOnly>
          <div className="prose">
            <h3>What the numbers say</h3>
            <ul>
              <li>
                <b>Locally it works.</b> Against equal-budget LIME, ArcTanh-LIME has significantly lower error at 0.5σ
                in {h.betterAt05.length} of {h.total} settings and at 1σ in {h.betterAt1.length} of {h.total} (median
                MSE ratio {fx(h.medianRatio1)} at 1σ, range {fx(h.rangeRatio1[0])}–{fx(h.rangeRatio1[1])}). Giving LIME
                the extra queries changes nothing: its error is bias, not variance.
              </li>
              <li>
                <b>It is the shape, not the extra points.</b> Fitting LIME on exactly ArcTanh-LIME&apos;s points, walk
                included, barely helps LIME: ArcTanh-LIME still wins at 1σ in {hs.betterAt1.length} of {hs.total}{" "}
                settings (median ratio {fx(hs.medianRatio1)}).
              </li>
              <li>
                <b>On Friedman #1 the data box does much of the work.</b> Confining the walk to the training range
                matters there. With an unbounded walk, ArcTanh-LIME&apos;s ratio to LIME at 0.5σ rises to{" "}
                {fx(unbO.mean_ratio ?? NaN)} on the true function (p = {unbO.wilcoxon_p.toFixed(2)}, not significant)
                and {fx(unbM.mean_ratio ?? NaN)} on the MLP.
              </li>
              <li>
                <b>Farther out it is mixed.</b> At 2σ it is significantly worse in {h.worseAt2.length} settings
                ({names(h.worseAt2)}) and at 4σ in {h.worseAt4.length} ({names(h.worseAt4)}). A bounded S-curve commits
                to its asymptotes; where the model turns, it is confidently wrong.
              </li>
              <li>
                <b>The ±δ anchors do extend locality</b> — ArcTanh-LIME-δ beats LIME at 4σ in{" "}
                {h.ensembleBetterAt4.length} of {h.total} settings — but much of that comes from having more anchors,
                not from tanh⁻¹: the linear LIME-δ ablation extends locality too. Like for like, ArcTanh-LIME-δ beats
                LIME-δ mainly within 2σ (switch the comparison above).
              </li>
              <li>
                <b>Attributions are not more accurate gradients.</b> The cosine between each method&apos;s attribution
                and the true gradient at x₀ is essentially unchanged (table below).
              </li>
            </ul>
          </div>

          <div className="grid-2 wide-only">
            <figure className="card" style={{ margin: 0 }}>
              <p className="fig-title">Cost: median black-box queries per explanation</p>
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr><th>Setting</th><th>LIME</th><th>ArcTanh</th><th>LIME-δ</th><th>ArcTanh-δ</th></tr>
                  </thead>
                  <tbody>
                    {costRows().map((r) => (
                      <tr key={r.key}>
                        <td>{r.title}</td>
                        {["LIME", "ArcTanh-LIME", "LIME-δ", "ArcTanh-LIME-δ"].map((m) => (
                          <td key={m}>{Math.round(r.queries[m]).toLocaleString("en-US")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <figcaption>Each ε-walk step costs 2d + 1 queries; the ensemble repeats everything at 2d + 1 anchors.</figcaption>
            </figure>
            <div className="stack">
              <figure className="card" style={{ margin: 0 }}>
                <p className="fig-title">Attribution vs true gradient (mean cosine)</p>
                <div className="table-scroll">
                  <table className="data">
                    <thead><tr><th>Setting</th><th>LIME</th><th>ArcTanh-LIME</th></tr></thead>
                    <tbody>
                      {gradientRows().filter((r) => !r.key.startsWith("1d_")).map((r) => (
                        <tr key={r.key}><td>{r.title}</td><td>{r.lime.toFixed(3)}</td><td>{r.arctanh.toFixed(3)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <figcaption>1-D settings omitted (a 1-D cosine is just the sign). The random forest has no useful gradient.</figcaption>
              </figure>
              <figure className="card" style={{ margin: 0 }}>
                <p className="fig-title">Seed stability ({STABILITY_META.instances} instances × {STABILITY_META.seeds} seeds)</p>
                <div className="table-scroll">
                  <table className="data">
                    <thead><tr><th>Setting</th><th>LIME cos</th><th>ArcTanh cos</th><th>LIME spread</th><th>ArcTanh spread</th></tr></thead>
                    <tbody>
                      {stabilityRows().map((r) => (
                        <tr key={r.key}>
                          <td>{r.title}</td>
                          <td>{r.values.LIME.cosine.toFixed(3)}</td>
                          <td>{r.values["ArcTanh-LIME"].cosine.toFixed(3)}</td>
                          <td>{r.values.LIME.spread.toFixed(3)}</td>
                          <td>{r.values["ArcTanh-LIME"].spread.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <figcaption>cos: mean pairwise cosine between attributions from different seeds (1 = identical). Spread: mean distance to the across-seed mean, relative to its norm.</figcaption>
              </figure>
            </div>
          </div>
        </section>

        <section id="diabetes">
          <div className="prose">
            <h2>Real data: diabetes risk</h2>
            <p>
              The same benchmark on four public diabetes datasets, each explained through four black boxes — logistic
              regression, an MLP, a random forest and gradient-boosted trees, all trained from scratch in the standard
              library. That makes {dh.total} settings with {DIA_META.instances} test instances each,
              N = {DIA_META.numSamples.toLocaleString("en-US")} perturbations, and the same fidelity measures, baselines
              and significance rule as above. Distances are in training standard deviations; yes/no indicators are
              perturbed as continuous values, as in LIME&apos;s continuous mode.
            </p>
          </div>
          <figure className="card">
            <p className="fig-title">Datasets and black boxes</p>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Dataset</th><th>d</th><th>train / test</th><th>positive</th>
                    {DIA_MODELS.map((m) => <th key={m}>AUC {DIA_MODEL_NAMES[m]}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {diabetesDatasets().map((r) => (
                    <tr key={r.key}>
                      <td>{r.title}</td>
                      <td>{r.d}</td>
                      <td>{r.nTrain.toLocaleString("en-US")} / {r.nTest.toLocaleString("en-US")}</td>
                      <td>{pct(r.positiveRate)}</td>
                      {DIA_MODELS.map((m) => <td key={m}>{r.auc[m].toFixed(2)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <figcaption>
              Pima Indians (NIDDK, via OpenML 37): impossible zeros in glucose, blood pressure, skin fold, insulin and
              BMI replaced by training medians. Early-stage risk (Sylhet; UCI 529, CC BY 4.0): yes/no symptoms as 0/1.
              CDC BRFSS 2015 health indicators (UCI 891, CC BY 4.0): a seeded 6,000-row subsample of 253,680, since the
              pure-Python models cannot train on all of it. Efron et al. (2004) progression data: the one-year
              progression score split at its median as &ldquo;high progression&rdquo;. The raw files are downloaded by a
              script and not committed; the 3-D lab above loads up to 500 preprocessed rows of each, with attribution.
            </figcaption>
          </figure>
          <ClientOnly minHeight={640}>
            <RatioPlot comparisons={diabetesComparisons()} metrics={RATIO_METRICS} />
          </ClientOnly>
          <div className="prose">
            <h3>What the diabetes numbers say</h3>
            <ul>
              <li>
                <b>Locally it holds up on real data.</b> Against equal-budget LIME, ArcTanh-LIME is significantly more
                faithful at 0.5σ in {dh.betterAt05.length} of {dh.total} settings and at 1σ in {dh.betterAt1.length} of{" "}
                {dh.total}.
              </li>
              <li>
                <b>Logistic regression is a sanity check, not a finding.</b> Because tanh(z) = 2σ(2z) − 1, a logistic
                model already lies in ArcTanh-LIME&apos;s surrogate family, so the error nearly vanishes
                ({span(dh.lr05)} LIME&apos;s at 0.5σ). LIME, being linear in probability space, cannot do this.
              </li>
              <li>
                <b>The MLPs are the informative case.</b> {span(dh.mlp05)} LIME&apos;s error at 0.5σ and {span(dh.mlp2)}{" "}
                at 2σ. Significant losses to LIME among the smooth models, over all nine distances and aggregates:{" "}
                {dh.smoothLosses.length ? dh.smoothLosses.join("; ") : "none"}.
              </li>
              <li>
                <b>Trees break it farther out — the same failure as the synthetic forest.</b> On the forests and
                boosted trees the walk hits a plateau after a median of {dh.treeSteps[0]}–{dh.treeSteps[1]} steps, and{" "}
                {pct(dh.treeClipped[0])}–{pct(dh.treeClipped[1])} of the regression targets fall outside the
                asymptotes. The local gain survives ({span(dh.tree05)} at 0.5σ), but the surrogate is significantly
                worse than LIME at 3σ in {dh.treesWorseAt3.length} of {dh.trees} tree settings and on the whole test
                set in {dh.treesWorseGlobal.length} of {dh.trees} (ratios {span(dh.treeGlobal)} across all {dh.trees}).
              </li>
              <li>
                <b>The cheap variant is the safe one.</b> Fixing the asymptotes at [0, 1] — no walk, no extra queries —{" "}
                {dh.noWalkLosses.length
                  ? <>is significantly worse than LIME only for {dh.noWalkLosses.join("; ")}</>
                  : <>is never significantly worse than LIME, in any setting, at any distance or on either aggregate</>}, and better at 0.5σ
                in {dh.noWalkBetterAt05.length} of {dh.total}. On trees its gain is small ({span(dh.noWalkTree05)}),
                but it does not fail far out.
              </li>
              <li>
                <b>±δ anchors do not rescue the trees.</b> ArcTanh-LIME-δ is still significantly worse than LIME at 4σ
                in {dh.ensembleTreesWorseAt4.length} of {dh.trees} tree settings, at {Math.round(dh.ensembleCost[0])}–
                {Math.round(dh.ensembleCost[1])}× LIME&apos;s query cost (the single-anchor walk costs +
                {Math.round(dh.overhead[0] * 100)}% to +{Math.round(dh.overhead[1] * 100)}%).
              </li>
            </ul>
          </div>
        </section>

        <section id="ablations">
          <div className="prose">
            <h2>Ablations</h2>
            <p>
              Which parts of the idea carry the improvement? Each variant below changes one thing and is compared, on
              the same instances, with the default configuration.
            </p>
          </div>
          <AblationExplorer tables={ablationTables()} />
          <div className="prose">
            <ul>
              <li>
                <b>The literal tanh⁻¹ regressor already helps.</b> Fitting β only in link space (no output-space
                refinement) still beats LIME at 0.5σ in {linkWins.wins.length} of {linkWins.total} ablation settings
                (Wilcoxon p &lt; 0.05); the refinement adds more (e.g. 8-D MLP at 1σ:{" "}
                {abl("synth_mlp", "fit = link space only (pure tanh⁻¹ regression)", "r1").toFixed(4)}
                {" "}→ {abl("synth_mlp", D, "r1").toFixed(4)}).
              </li>
              <li>
                <b>The walk matters only when the local range differs from the global one.</b> For probability
                outputs, fixing the asymptotes to [0, 1] without any walk is about as good (8-D MLP at 1σ:{" "}
                {abl("synth_mlp", "asymptotes fixed to [0, 1] (no walk)", "r1").toFixed(4)} vs{" "}
                {abl("synth_mlp", D, "r1").toFixed(4)}) and costs fewer queries. On multimodal or non-saturating
                curves the walk pays off: using the global output range instead is{" "}
                {fx(abl("1d_ripple", "asymptotes = global output range (no walk)", "r0.5") / abl("1d_ripple", D, "r0.5"))}{" "}
                worse on trend + ripple and{" "}
                {fx(abl("1d_bumps", "asymptotes = global output range (no walk)", "r0.5") / abl("1d_bumps", D, "r0.5"))}{" "}
                worse on bumps at 0.5σ. Conversely, even with no walk at all, the global-range tanh⁻¹ regressor beats
                LIME at 0.5σ in {globalWins.wins.length} of {globalWins.total} settings — the bounded link alone is
                worth having.
              </li>
              <li>
                <b>Pinning the origin</b> (forcing g(x₀) = f(x₀)) slashes error at 0.5σ in 8–10 dimensions but
                is up to 1.4× worse at 2σ in the multi-dimensional settings. LIME&apos;s intercept is biased in high dimensions too — pinning LIME gives a similar
                local boost — yet like for like the pinned tanh⁻¹ regressor still beats pinned LIME at 1σ in {pinWins.wins.length} of{" "}
                {pinWins.total} settings.
              </li>
              <li>
                <b>A 0% margin</b> (the extrema used exactly as asymptotes, as originally described) is slightly
                better than the 5% margin fixed before the experiments; the walk&apos;s tolerance and ε barely matter for
                smooth models, while a larger ε helps the random forest by stepping over plateaus.
              </li>
            </ul>
          </div>
        </section>

        <section id="limits">
          <div className="prose">
            <h2>Where it breaks</h2>
          </div>
          <div className="failure-grid">
            <div className="card failure">
              <h4>Piecewise-constant models</h4>
              <p>
                On the random forest the finite-difference walk stalls on plateaus (a median of{" "}
                {rfWalk.up_steps.median} steps up and {rfWalk.down_steps.median} down), so the asymptotes are far too
                tight: on average {pct(rfWalk.clipped_fraction.mean)} of the regression targets lie outside them.
                Fidelity beyond 2σ is worse than LIME ({fx(ratio("synth_rf", at, lm, "r2"))} at 2σ) and attributions
                become unstable across seeds (cosine {stab.synth_rf["ArcTanh-LIME"].cosine.toFixed(2)} vs{" "}
                {stab.synth_rf.LIME.cosine.toFixed(2)}).
              </p>
            </div>
            <div className="card failure">
              <h4>The data box, not the walk (Friedman #1)</h4>
              <p>
                Much of the Friedman gain depends on keeping the walk inside the training range. An unbounded walk
                overshoots into regions where the polynomial keeps growing, and the asymptotes become too wide:{" "}
                {fx(unbO.mean_ratio ?? NaN)} and {fx(unbM.mean_ratio ?? NaN)} LIME&apos;s error at 0.5σ instead of{" "}
                {fx(ratio("friedman_oracle", at, lm, "r0.5"))} and {fx(ratio("friedman_mlp", at, lm, "r0.5"))}.
              </p>
            </div>
            <div className="card failure">
              <h4>Confident extrapolation</h4>
              <p>
                A bounded S-curve saturates toward one asymptote. Where the model turns back — the curved boundary of
                the two moons — that confidence becomes error: {fx(ratio("moons_mlp", at, lm, "r2"))} LIME&apos;s MSE at
                2σ.
              </p>
            </div>
            <div className="card failure">
              <h4>Walk domain ≠ sampling domain</h4>
              <p>
                Walks stay inside the data range, LIME&apos;s Gaussian samples do not. On Friedman #1 (inputs in
                [0, 1]¹⁰) ArcTanh-LIME is therefore no better than LIME on LIME&apos;s own neighbourhood metric
                ({fx(ratio("friedman_oracle", at, lm, "neighborhood_wmse"))} and{" "}
                {fx(ratio("friedman_mlp", at, lm, "neighborhood_wmse"))}).
              </p>
            </div>
            <div className="card failure">
              <h4>Cost grows with dimension</h4>
              <p>
                Each walk step costs 2d + 1 queries: +{Math.round(h.overheadRange[0] * 100)}% to +
                {Math.round(h.overheadRange[1] * 100)}% over LIME here, and the ±δ ensemble multiplies the bill by
                2d + 1 anchors ({Math.round(benchMethod("friedman_mlp", "ArcTanh-LIME-δ").queries.median).toLocaleString("en-US")}{" "}
                queries per explanation at d = 10).
              </p>
            </div>
            <div className="card failure">
              <h4>Fidelity, not understanding</h4>
              <p>
                The evidence is synthetic functions plus four public diabetes datasets ({DIA_META.instances} test
                instances per setting, BRFSS subsampled). Yes/no features are perturbed as continuous values, so both
                methods are queried at points no patient could have. Nothing here is a user study: the results describe
                surrogate fidelity, not whether people understand the explanations better.
              </p>
            </div>
          </div>
        </section>

        <section id="implications" className="prose">
          <h2>Implications</h2>
          <ul>
            <li>
              <b>Bounded outputs deserve bounded surrogates.</b> For classifiers, most of the local gain comes from
              fitting LIME on the tanh⁻¹ (equivalently, logit) scale — a nearly free change to existing pipelines.
            </li>
            <li>
              <b>The walk is a by-product worth showing.</b> Its endpoints are the nearest points, along steepest
              ascent and descent, where the prediction peaks or bottoms out — counterfactual-style information that
              comes free with the asymptotes (not evaluated as an explanation here).
            </li>
            <li>
              <b>Locality can be widened deliberately.</b> ±δ anchors trade queries for range in a controlled way;
              the surrogate&apos;s faithful region grows with δ and with extra rings.
            </li>
            <li>
              <b>Match the explainer to the model.</b> On smooth models the method is a solid local improvement; on
              tree ensembles the walk needs smoothing (or the asymptote ablations) before it can be trusted. The
              diabetes benchmark shows the same split on real data, with random forests and gradient boosting alike.
            </li>
          </ul>
        </section>

        <section id="reproduce" className="prose">
          <h2>Reproduce</h2>
          <p>Everything on this page regenerates from the repository in a few minutes on a laptop:</p>
          <pre>
            <code>{`python3 -m unittest discover -s tests -t .   # 30 unit tests
python3 -m experiments.run_benchmarks          # main benchmark  (~3 min)
python3 -m experiments.run_ablations           # ablations       (~11 min)
python3 -m experiments.run_stability           # seed stability  (~10 s)
python3 -m experiments.diabetes download       # 4 public datasets (~14 MB; hashes checked against results/diabetes/data_manifest.json)
python3 -m experiments.diabetes models         # train the 16 black boxes
python3 -m experiments.run_diabetes            # diabetes benchmark
python3 -m experiments.export_web              # data for this site
viz/build.sh test && viz/build.sh wasm         # 3-D engine (needs Emscripten)
cd web && npm install && npm run parity && npm run dev`}</code>
          </pre>
        </section>

        <footer>
          <p>
            References: M. T. Ribeiro, S. Singh, C. Guestrin. &ldquo;Why Should I Trust You?&rdquo;: Explaining the
            Predictions of Any Classifier. KDD 2016. · J. H. Friedman. Multivariate Adaptive Regression Splines. Annals
            of Statistics 1991 (benchmark function #1).
          </p>
          <p>
            All code is original and dependency-free on the research side; the site uses Next.js and React only.
            Numbers on this page are read from the experiment outputs at build time.
          </p>
        </footer>
      </main>
    </>
  );
}
