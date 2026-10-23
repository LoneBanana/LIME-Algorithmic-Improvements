"""Unit tests for the stdlib-only core (run: python3 -m unittest discover -s tests)."""

from __future__ import annotations

import math
import unittest

from atlime import (ArcTanhConfig, ArcTanhLime, BlendedSurrogate, DeltaAnchorEnsemble, DeltaConfig, LimeConfig,
                    LimeTabular, LinearSurrogate, Rng, TanhSurrogate, anchor_points, epsilon_walk)
from atlime.datasets import make_moons
from atlime.lime import default_kernel_width, select_features
from atlime.linalg import solve_spd, weighted_r2, weighted_ridge
from atlime.metrics import (bootstrap_mean_diff_ci, cosine, true_gradient, unit_directions, wilcoxon_signed_rank,
                            win_rate)
from atlime.models import MLP, GradientBoosting, RandomForest
from atlime.walk import finite_difference_gradient


class TestRng(unittest.TestCase):
    def test_matches_javascript_mulberry32(self):
        # reference values printed by the canonical JS implementation, seed 42
        r = Rng(42)
        expected = [0.6011037519201636, 0.44829055899754167, 0.8524657934904099, 0.6697340414393693,
                    0.17481389874592423]
        self.assertEqual([r.uniform() for _ in range(5)], expected)

    def test_normal_moments(self):
        r = Rng(1)
        xs = [r.normal() for _ in range(20000)]
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        self.assertAlmostEqual(m, 0.0, delta=0.03)
        self.assertAlmostEqual(v, 1.0, delta=0.04)

    def test_shuffle_is_permutation_and_spawn_deterministic(self):
        a = list(range(50))
        Rng(3).shuffle(a)
        self.assertEqual(sorted(a), list(range(50)))
        self.assertNotEqual(a, list(range(50)))
        self.assertEqual(Rng(9).spawn(4).uniform(), Rng(9).spawn(4).uniform())


class TestLinalg(unittest.TestCase):
    def test_solve_spd(self):
        r = Rng(5)
        n = 6
        B = [[r.normal() for _ in range(n)] for _ in range(n)]
        A = [[sum(B[k][i] * B[k][j] for k in range(n)) + (1.0 if i == j else 0.0) for j in range(n)]
             for i in range(n)]
        b = [r.normal() for _ in range(n)]
        x = solve_spd(A, b)
        for i in range(n):
            self.assertAlmostEqual(sum(A[i][j] * x[j] for j in range(n)), b[i], places=10)

    def test_ridge_recovers_exact_linear_model(self):
        r = Rng(6)
        rows = [[r.normal(), r.normal(), r.normal()] for _ in range(80)]
        y = [1.5 + 2.0 * a - 3.0 * b + 0.5 * c for a, b, c in rows]
        w = [0.1 + r.uniform() for _ in rows]
        beta, b0 = weighted_ridge(rows, y, w, alpha=0.0)
        for got, want in zip(beta, [2.0, -3.0, 0.5]):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(b0, 1.5, places=9)

    def test_ridge_closed_form_one_feature(self):
        xs = [[-1.0], [0.0], [2.0], [3.0]]
        y = [0.0, 1.0, 1.0, 4.0]
        w = [1.0, 2.0, 1.0, 0.5]
        alpha = 0.7
        sw = sum(w)
        xm = sum(wi * x[0] for wi, x in zip(w, xs)) / sw
        ym = sum(wi * v for wi, v in zip(w, y)) / sw
        sxy = sum(wi * (x[0] - xm) * (v - ym) for wi, x, v in zip(w, xs, y))
        sxx = sum(wi * (x[0] - xm) ** 2 for wi, x in zip(w, xs))
        beta, b0 = weighted_ridge(xs, y, w, alpha=alpha)
        self.assertAlmostEqual(beta[0], sxy / (sxx + alpha), places=12)
        self.assertAlmostEqual(b0, ym - beta[0] * xm, places=12)

    def test_weighted_r2(self):
        self.assertEqual(weighted_r2([1, 2, 3], [1, 2, 3], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(weighted_r2([1, 2, 3], [2, 2, 2], [1, 1, 1]), 0.0)


class TestLime(unittest.TestCase):
    def test_linear_black_box_is_recovered(self):
        scale = [2.0, 0.5]
        f = lambda x: 3.0 + 2.0 * x[0] - 1.0 * x[1]
        res = LimeTabular(scale, LimeConfig(num_samples=400, alpha=1e-9)).explain(f, [1.0, -2.0], Rng(1))
        self.assertAlmostEqual(res.surrogate.coef[0], 2.0 * 2.0, places=6)   # per std of feature 0
        self.assertAlmostEqual(res.surrogate.coef[1], -1.0 * 0.5, places=6)
        self.assertAlmostEqual(res.local_pred, f([1.0, -2.0]), places=6)
        self.assertAlmostEqual(res.score, 1.0, places=9)
        self.assertEqual(res.neighborhood.points[0], [1.0, -2.0])  # first sample is the instance

    def test_pinned_intercept_variant(self):
        f = lambda x: math.sin(x[0])
        res = LimeTabular([1.0], LimeConfig(num_samples=200, pin_intercept=True)).explain(f, [0.8], Rng(2))
        self.assertEqual(res.local_pred, f([0.8]))
        self.assertAlmostEqual(res.surrogate.predict([0.8]), f([0.8]), places=12)

    def test_default_kernel_width(self):
        self.assertAlmostEqual(default_kernel_width(16), 3.0)

    def test_feature_selection_finds_informative_features(self):
        r = Rng(2)
        Z = [[r.normal() for _ in range(6)] for _ in range(300)]
        y = [4.0 * z[1] - 3.0 * z[4] + 0.01 * z[0] for z in Z]
        w = [1.0] * len(Z)
        self.assertEqual(select_features(Z, y, w, "forward_selection", 2), [1, 4])
        self.assertEqual(select_features(Z, y, w, "highest_weights", 2), [1, 4])
        res = LimeTabular([1.0] * 6, LimeConfig(num_samples=300, feature_selection="auto", num_features=2))
        out = res.explain(lambda x: 4.0 * x[1] - 3.0 * x[4], [0.0] * 6, Rng(3))
        self.assertEqual(out.used_features, [1, 4])
        self.assertEqual(sum(1 for c in out.surrogate.coef if c != 0.0), 2)


class TestWalk(unittest.TestCase):
    def test_sine_walk_finds_neighbouring_extrema(self):
        f = lambda x: math.sin(x[0])
        up = epsilon_walk(f, [0.3], f([0.3]), [1.0], +1, 0.05, 200, 0.05, 0.0)
        dn = epsilon_walk(f, [0.3], f([0.3]), [1.0], -1, 0.05, 200, 0.05, 0.0)
        self.assertEqual(up.stop_reason, "extremum")
        self.assertEqual(dn.stop_reason, "extremum")
        self.assertLess(abs(up.end_point[0] - math.pi / 2), 0.05)
        self.assertLess(abs(dn.end_point[0] + math.pi / 2), 0.05)
        self.assertTrue(all(b > a for a, b in zip(up.values, up.values[1:])))   # monotone ascent

    def test_bowl_and_steps_have_length_epsilon(self):
        f = lambda x: -(x[0] ** 2 + 4.0 * x[1] ** 2)
        res = epsilon_walk(f, [1.0, 1.0], f([1.0, 1.0]), [1.0, 1.0], +1, 0.1, 100, 1e-3, 0.0)
        self.assertEqual(res.stop_reason, "extremum")
        self.assertLess(math.hypot(*res.end_point), 0.15)
        for p, q in zip(res.points, res.points[1:]):
            self.assertAlmostEqual(math.hypot(q[0] - p[0], q[1] - p[1]), 0.1, places=9)

    def test_bounds_and_flat(self):
        f = lambda x: x[0]
        res = epsilon_walk(f, [0.5], 0.5, [1.0], +1, 0.1, 100, 0.01, 0.0, bounds=[(0.0, 1.0)])
        self.assertEqual(res.stop_reason, "bounds")
        self.assertAlmostEqual(res.end_point[0], 1.0)
        flat = epsilon_walk(lambda x: 2.0, [0.0], 2.0, [1.0], +1, 0.1, 10, 0.1, 0.0)
        self.assertEqual(flat.stop_reason, "flat")
        self.assertEqual(flat.steps, 0)

    def test_finite_difference_gradient_units(self):
        g, q = finite_difference_gradient(lambda x: 3.0 * x[0] + x[1] ** 2, [0.0, 1.0], [2.0, 1.0], 1e-4)
        self.assertAlmostEqual(g[0], 6.0, places=6)   # per std: 3 * 2
        self.assertAlmostEqual(g[1], 2.0, places=6)
        self.assertEqual(q, 4)


class TestArcTanh(unittest.TestCase):
    def test_recovers_tanh_family_with_fixed_asymptotes(self):
        lo, hi = 0.1, 0.9
        true = TanhSurrogate([0.0, 0.0], [1.0, 1.0], lo, hi, 0.4, [1.5, -0.7])
        cfg = ArcTanhConfig(num_samples=300, alpha=1e-9, asymptotes="fixed", fixed_bounds=(lo, hi), margin=0.0)
        res = ArcTanhLime([1.0, 1.0], cfg).explain(true.predict, [0.0, 0.0], Rng(4))
        self.assertAlmostEqual(res.surrogate.beta0, 0.4, places=5)
        self.assertAlmostEqual(res.surrogate.beta[0], 1.5, places=5)
        self.assertAlmostEqual(res.surrogate.beta[1], -0.7, places=5)
        for got, want in zip(res.attribution, true.gradient([0.0, 0.0])):
            self.assertAlmostEqual(got, want, places=5)

    def test_link_fit_is_ridge_on_artanh_targets(self):
        f = lambda x: 1.0 / (1.0 + math.exp(-2.0 * x[0]))
        cfg = ArcTanhConfig(num_samples=200, fit="link", use_walk_points=False)
        res = ArcTanhLime([1.0], cfg).explain(f, [0.2], Rng(5))
        nb = res.neighborhood
        c, a = 0.5 * (res.hi + res.lo), 0.5 * (res.hi - res.lo)
        lim = 1.0 - cfg.clip
        t = [math.atanh(max(-lim, min(lim, (v - c) / a))) for v in nb.y]
        beta, b0 = weighted_ridge(nb.Z, t, nb.w, alpha=cfg.alpha)
        self.assertAlmostEqual(res.link_surrogate.beta[0], beta[0], places=12)
        self.assertAlmostEqual(res.link_surrogate.beta0, b0, places=12)

    def test_walk_asymptotes_on_logistic(self):
        f = lambda x: 1.0 / (1.0 + math.exp(-2.0 * x[0]))
        res = ArcTanhLime([1.0], ArcTanhConfig(num_samples=300, max_steps=200)).explain(f, [0.0], Rng(6))
        self.assertLess(res.raw_lo, 0.01)
        self.assertGreater(res.raw_hi, 0.99)
        lime = LimeTabular([1.0], LimeConfig(num_samples=300)).explain(f, [0.0], Rng(6))
        pts = [[-2.0 + 0.1 * i] for i in range(41)]
        err_at = sum((res.surrogate.predict(p) - f(p)) ** 2 for p in pts)
        err_lime = sum((lime.surrogate.predict(p) - f(p)) ** 2 for p in pts)
        self.assertLess(err_at, 0.1 * err_lime)

    def test_pin_origin_interpolates_instance(self):
        f = lambda x: math.sin(x[0]) + 0.3 * x[1]
        cfg = ArcTanhConfig(num_samples=200, pin_origin=True)
        res = ArcTanhLime([1.0, 1.0], cfg).explain(f, [0.4, -0.2], Rng(7))
        self.assertAlmostEqual(res.surrogate.predict([0.4, -0.2]), f([0.4, -0.2]), places=9)

    def test_constant_black_box_falls_back(self):
        res = ArcTanhLime([1.0], ArcTanhConfig(num_samples=50)).explain(lambda x: 0.25, [0.0], Rng(8))
        self.assertEqual(res.fallback, "constant")
        self.assertEqual(res.surrogate.predict([3.0]), 0.25)


class TestEnsemble(unittest.TestCase):
    def test_anchor_layout(self):
        anchors = anchor_points([0.0, 0.0, 0.0], [1.0, 2.0, 1.0], DeltaConfig(delta=0.5, rings=2))
        self.assertEqual(len(anchors), 1 + 2 * 3 * 2)
        self.assertIn([0.0, 1.0, 0.0], anchors)          # +0.5 std of feature 1 (scale 2)
        self.assertIn([0.0, 0.0, -1.0], anchors)         # -2 * 0.5 std of feature 2
        clipped = anchor_points([0.0], [1.0], DeltaConfig(delta=1.0, rings=2), bounds=[(-1.5, 1.5)])
        self.assertEqual(sorted(a[0] for a in clipped), [-1.5, -1.0, 0.0, 1.0, 1.5])

    def test_blend_weights_and_gradient(self):
        members = [LinearSurrogate([0.0, 0.0], [1.0, 1.0], 0.0, [1.0, 0.5]),
                   TanhSurrogate([1.0, 0.0], [1.0, 1.0], -1.0, 2.0, 0.3, [0.8, -0.4]),
                   LinearSurrogate([0.0, -1.0], [1.0, 1.0], 1.0, [-0.2, 0.1])]
        blend = BlendedSurrogate([0.0, 0.0], [1.0, 1.0], [m.anchor for m in members], members, 0.7)
        x = [0.3, -0.4]
        self.assertAlmostEqual(sum(blend.weights(x)), 1.0, places=12)
        g = blend.gradient(x)
        h = 1e-6
        for j in range(2):
            xp, xm = list(x), list(x)
            xp[j] += h
            xm[j] -= h
            self.assertAlmostEqual(g[j], (blend.predict(xp) - blend.predict(xm)) / (2 * h), places=6)

    def test_ensemble_runs_and_counts_queries(self):
        calls = [0]

        def f(x):
            calls[0] += 1
            return math.sin(x[0])

        base = ArcTanhLime([1.0], ArcTanhConfig(num_samples=100))
        res = DeltaAnchorEnsemble(base, DeltaConfig(delta=1.0)).explain(f, [0.0], Rng(9))
        self.assertEqual(len(res.anchors), 3)
        self.assertEqual(res.n_queries, calls[0])
        self.assertLess(abs(res.surrogate.predict([0.0]) - 0.0), 0.05)


class TestModels(unittest.TestCase):
    def test_mlp_learns_moons_and_roundtrips(self):
        X, y = make_moons(300, 0.1, Rng(10))
        net = MLP([2, 16, 1], "sigmoid", seed=1)
        hist = net.fit(X, y, epochs=60, batch_size=32, lr=0.02, seed=2)
        self.assertLess(hist[-1], hist[0])
        acc = sum((net(x) > 0.5) == (t > 0.5) for x, t in zip(X, y)) / len(X)
        self.assertGreater(acc, 0.95)
        clone = MLP.from_dict(net.to_dict())
        self.assertEqual(clone(X[0]), net(X[0]))

    def test_mlp_backprop_matches_finite_differences(self):
        # the exact routine fit() uses, checked parameter by parameter, for both output types
        for output, target in (("sigmoid", 1.0), ("linear", 0.7)):
            net = MLP([3, 5, 4, 1], output, seed=3)
            r = Rng(4)
            for W in net.W:  # move away from the near-symmetric initialisation
                for row in W:
                    for j in range(len(row)):
                        row[j] += 0.3 * r.normal()
            xs = [0.4, -1.1, 0.7]
            gW = [[[0.0] * len(row) for row in W] for W in net.W]
            gb = [[0.0] * len(b) for b in net.b]
            cols = [list(zip(*W)) for W in net.W]
            loss = net._accumulate_gradient(xs, target, gW, gb, cols)

            def loss_now():
                o = net._forward(xs)[-1][0]
                if output == "sigmoid":
                    p = 1.0 / (1.0 + math.exp(-o))
                    return -(target * math.log(p) + (1 - target) * math.log(1 - p))
                return 0.5 * (o - target) ** 2

            self.assertAlmostEqual(loss, loss_now(), places=12)
            h = 1e-6
            checked = 0
            for params, grads in ((net.W, gW), (net.b, gb)):
                for layer, P in enumerate(params):
                    rows = P if isinstance(P[0], list) else [P]
                    grows = grads[layer] if isinstance(P[0], list) else [grads[layer]]
                    for row, grow in zip(rows, grows):
                        for j in range(len(row)):
                            old_v = row[j]
                            row[j] = old_v + h
                            up = loss_now()
                            row[j] = old_v - h
                            dn = loss_now()
                            row[j] = old_v
                            self.assertAlmostEqual(grow[j], (up - dn) / (2 * h), places=6)
                            checked += 1
            self.assertEqual(checked, 3 * 5 + 5 + 5 * 4 + 4 + 4 * 1 + 1)

    def test_random_forest_fits_step(self):
        r = Rng(11)
        X = [[r.uniform() * 2 - 1, r.uniform()] for _ in range(400)]
        y = [1.0 if x[0] > 0.2 else 0.0 for x in X]
        rf = RandomForest(n_trees=10, max_depth=4, min_samples_leaf=3, seed=1).fit(X, y)
        self.assertGreater(rf([0.8, 0.5]), 0.9)
        self.assertLess(rf([-0.6, 0.5]), 0.1)
        clone = RandomForest.from_dict(rf.to_dict())
        self.assertEqual(clone([0.21, 0.3]), rf([0.21, 0.3]))


    def test_gradient_boosting_learns_interaction_and_roundtrips(self):
        r = Rng(15)
        X = [[r.uniform() * 2 - 1, r.uniform() * 2 - 1] for _ in range(600)]
        y = [1.0 if a * b > 0 else 0.0 for a, b in X]  # XOR-like: needs interactions
        gb = GradientBoosting(n_trees=60, learning_rate=0.3, max_depth=3, min_samples_leaf=5).fit(X, y)
        acc = sum((gb(x) > 0.5) == (t > 0.5) for x, t in zip(X, y)) / len(X)
        self.assertGreater(acc, 0.9)
        self.assertTrue(all(0.0 < gb(x) < 1.0 for x in X[:20]))
        clone = GradientBoosting.from_dict(gb.to_dict())
        self.assertEqual(clone([0.3, -0.2]), gb([0.3, -0.2]))


class TestMetrics(unittest.TestCase):
    def test_wilcoxon_matches_normal_approximation(self):
        a = [float(i) for i in range(1, 11)]
        res = wilcoxon_signed_rank(a, [0.0] * 10)
        self.assertEqual(res["W_plus"], 55.0)
        self.assertAlmostEqual(res["z"], 27.5 / math.sqrt(96.25), places=12)
        self.assertAlmostEqual(res["p"], 0.005063, places=5)
        self.assertEqual(wilcoxon_signed_rank([1, 2], [1, 2])["p"], 1.0)

    def test_bootstrap_and_win_rate(self):
        r = Rng(12)
        a = [1.0 + r.normal() for _ in range(200)]
        b = [0.0 + r.normal() for _ in range(200)]
        lo, hi = bootstrap_mean_diff_ci(a, b, Rng(13), n_boot=500)
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 1.0)
        self.assertEqual(win_rate([1, 2, 3], [2, 2, 2]), 0.5)

    def test_misc(self):
        self.assertAlmostEqual(cosine([1, 0], [0, 2]), 0.0)
        g = true_gradient(lambda x: x[0] * x[1], [2.0, 3.0], [1.0, 2.0])
        self.assertAlmostEqual(g[0], 3.0, places=6)
        self.assertAlmostEqual(g[1], 4.0, places=6)
        dirs = unit_directions(3, 5, Rng(14))
        for v in dirs:
            self.assertAlmostEqual(sum(c * c for c in v), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
