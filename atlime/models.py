"""Black-box models used in the experiments, implemented from scratch.

* ``MLP``          - fully-connected tanh network trained with Adam
                     (sigmoid output + cross-entropy, or linear output + MSE).
* ``RandomForest`` - bagged CART regression trees (probability = mean leaf
                     frequency); a deliberately non-smooth black box.
* ``CountingBlackBox`` - wraps any callable and counts queries.

Models serialise to plain JSON-compatible dicts so trained weights can be
cached on disk and shipped to the website.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .rng import Rng


def sigmoid(v: float) -> float:
    if v >= 0.0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


class CountingBlackBox:
    def __init__(self, f: Callable[[Sequence[float]], float]) -> None:
        self.f = f
        self.calls = 0

    def __call__(self, x: Sequence[float]) -> float:
        self.calls += 1
        return self.f(x)


class MLP:
    """tanh hidden layers; ``output`` is 'sigmoid' (probability) or 'linear'."""

    def __init__(self, sizes: Sequence[int], output: str = "sigmoid", seed: int = 0) -> None:
        if output not in ("sigmoid", "linear"):
            raise ValueError("output must be 'sigmoid' or 'linear'")
        self.sizes = list(sizes)
        self.output = output
        rng = Rng(seed)
        self.W: list[list[list[float]]] = []
        self.b: list[list[float]] = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))  # Glorot uniform
            self.W.append([[(2.0 * rng.uniform() - 1.0) * limit for _ in range(fan_in)] for _ in range(fan_out)])
            self.b.append([0.0] * fan_out)
        self.x_mean = [0.0] * self.sizes[0]
        self.x_std = [1.0] * self.sizes[0]
        self.y_mean = 0.0
        self.y_std = 1.0

    # -- inference -------------------------------------------------------------
    def _forward(self, xs: Sequence[float]) -> list[list[float]]:
        acts = [list(xs)]
        a = acts[0]
        last = len(self.W) - 1
        for layer, (W, b) in enumerate(zip(self.W, self.b)):
            z = [bi + math.sumprod(row, a) for row, bi in zip(W, b)]
            a = z if layer == last else [math.tanh(v) for v in z]
            acts.append(a)
        return acts

    def _standardize(self, x: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.x_mean, self.x_std)]

    def predict(self, x: Sequence[float]) -> float:
        o = self._forward(self._standardize(x))[-1][0]
        if self.output == "sigmoid":
            return sigmoid(o)
        return self.y_mean + self.y_std * o

    __call__ = predict

    # -- training --------------------------------------------------------------
    def _accumulate_gradient(self, xs: Sequence[float], target: float, gW: list, gb: list, cols: list) -> float:
        """Backpropagation for one standardised sample: adds dLoss/dW and dLoss/db into gW / gb and returns
        the loss (cross-entropy for a sigmoid output, 0.5 * squared error for a linear one). ``cols`` holds the
        transposed weight matrices."""
        acts = self._forward(xs)
        o = acts[-1][0]
        if self.output == "sigmoid":
            p = sigmoid(o)
            loss = -(target * math.log(max(p, 1e-12)) + (1 - target) * math.log(max(1 - p, 1e-12)))
            delta = [p - target]
        else:
            r = o - target
            loss = 0.5 * r * r
            delta = [r]
        for layer in range(len(self.W) - 1, -1, -1):
            a_prev = acts[layer]
            gWl = gW[layer]
            gbl = gb[layer]
            for i, di in enumerate(delta):
                gWl[i] = [g + di * aj for g, aj in zip(gWl[i], a_prev)]
                gbl[i] += di
            if layer > 0:
                delta = [(1.0 - aj * aj) * math.sumprod(col, delta) for aj, col in zip(a_prev, cols[layer])]
        return loss

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float], epochs: int = 100, batch_size: int = 32,
            lr: float = 0.01, weight_decay: float = 0.0, seed: int = 0, verbose: bool = False) -> list[float]:
        n = len(X)
        d = len(X[0])
        self.x_mean = [math.fsum(r[j] for r in X) / n for j in range(d)]
        self.x_std = [math.sqrt(math.fsum((r[j] - self.x_mean[j]) ** 2 for r in X) / n) or 1.0 for j in range(d)]
        Xs = [self._standardize(r) for r in X]
        if self.output == "linear":
            self.y_mean = math.fsum(y) / n
            self.y_std = math.sqrt(math.fsum((v - self.y_mean) ** 2 for v in y) / n) or 1.0
            ys = [(v - self.y_mean) / self.y_std for v in y]
        else:
            ys = list(y)
        rng = Rng(seed)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        mW = [[[0.0] * len(r) for r in W] for W in self.W]
        vW = [[[0.0] * len(r) for r in W] for W in self.W]
        mb = [[0.0] * len(b) for b in self.b]
        vb = [[0.0] * len(b) for b in self.b]
        step = 0
        history = []
        order = list(range(n))
        L = len(self.W)
        for epoch in range(epochs):
            rng.shuffle(order)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                gW = [[[0.0] * len(r) for r in W] for W in self.W]
                gb = [[0.0] * len(b) for b in self.b]
                cols = [list(zip(*W)) for W in self.W]
                for idx in batch:
                    total_loss += self._accumulate_gradient(Xs[idx], ys[idx], gW, gb, cols)
                step += 1
                bs = len(batch)
                c1 = 1.0 - beta1 ** step
                c2 = 1.0 - beta2 ** step
                for layer in range(L):
                    W, b = self.W[layer], self.b[layer]
                    for i in range(len(W)):
                        row, grow, mrow, vrow = W[i], gW[layer][i], mW[layer][i], vW[layer][i]
                        for j in range(len(row)):
                            g = grow[j] / bs + weight_decay * row[j]
                            mrow[j] = beta1 * mrow[j] + (1 - beta1) * g
                            vrow[j] = beta2 * vrow[j] + (1 - beta2) * g * g
                            row[j] -= lr * (mrow[j] / c1) / (math.sqrt(vrow[j] / c2) + eps_adam)
                        g = gb[layer][i] / bs
                        mb[layer][i] = beta1 * mb[layer][i] + (1 - beta1) * g
                        vb[layer][i] = beta2 * vb[layer][i] + (1 - beta2) * g * g
                        b[i] -= lr * (mb[layer][i] / c1) / (math.sqrt(vb[layer][i] / c2) + eps_adam)
            history.append(total_loss / n)
            if verbose:
                print(f"  epoch {epoch + 1:3d}  loss {total_loss / n:.4f}")
        return history

    # -- serialisation ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "mlp", "sizes": self.sizes, "output": self.output, "W": self.W, "b": self.b,
                "x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_dict(cls, data: dict) -> "MLP":
        m = cls(data["sizes"], data["output"])
        m.W, m.b = data["W"], data["b"]
        m.x_mean, m.x_std = data["x_mean"], data["x_std"]
        m.y_mean, m.y_std = data["y_mean"], data["y_std"]
        return m


class RegressionTree:
    """CART with squared-error splits, stored as flat arrays."""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 5, max_features: int | None = None) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.feature: list[int] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.value: list[float] = []

    def _new_node(self, value: float) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.value) - 1

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float], idx: list[int], rng: Rng) -> "RegressionTree":
        d = len(X[0])
        m = self.max_features or d
        stack = [(self._new_node(math.fsum(y[i] for i in idx) / len(idx)), idx, 0)]
        while stack:
            node, rows, depth = stack.pop()
            if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
                continue
            feats = list(range(d))
            rng.shuffle(feats)
            split = self._best_split(X, y, rows, feats[:m])
            if split is None:
                continue
            j, thr = split
            left_rows = [i for i in rows if X[i][j] <= thr]
            right_rows = [i for i in rows if X[i][j] > thr]
            self.feature[node] = j
            self.threshold[node] = thr
            ln = self._new_node(math.fsum(y[i] for i in left_rows) / len(left_rows))
            rn = self._new_node(math.fsum(y[i] for i in right_rows) / len(right_rows))
            self.left[node] = ln
            self.right[node] = rn
            stack.append((ln, left_rows, depth + 1))
            stack.append((rn, right_rows, depth + 1))
        return self

    def _best_split(self, X, y, rows, feats) -> tuple[int, float] | None:
        n = len(rows)
        total = math.fsum(y[i] for i in rows)
        base = total * total / n
        best_gain, best = 1e-12, None
        leaf = self.min_samples_leaf
        for j in feats:
            order = sorted(rows, key=lambda i: X[i][j])
            left_sum = 0.0
            for k in range(n - 1):
                left_sum += y[order[k]]
                ln = k + 1
                if ln < leaf or n - ln < leaf:
                    continue
                xa, xb = X[order[k]][j], X[order[k + 1]][j]
                if xa == xb:
                    continue
                right_sum = total - left_sum
                gain = left_sum * left_sum / ln + right_sum * right_sum / (n - ln) - base
                if gain > best_gain:
                    best_gain, best = gain, (j, 0.5 * (xa + xb))
        return best

    def leaf(self, x: Sequence[float]) -> int:
        node = 0
        feature, threshold, left, right = self.feature, self.threshold, self.left, self.right
        while feature[node] >= 0:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        return node

    def predict(self, x: Sequence[float]) -> float:
        return self.value[self.leaf(x)]

    def to_dict(self) -> dict:
        return {"feature": self.feature, "threshold": self.threshold, "left": self.left, "right": self.right,
                "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionTree":
        t = cls()
        t.feature, t.threshold, t.left, t.right, t.value = (data["feature"], data["threshold"], data["left"],
                                                            data["right"], data["value"])
        return t


class RandomForest:
    def __init__(self, n_trees: int = 40, max_depth: int = 10, min_samples_leaf: int = 5,
                 max_features: int | None = None, seed: int = 0) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees: list[RegressionTree] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "RandomForest":
        rng = Rng(self.seed)
        n = len(X)
        m = self.max_features or max(1, int(round(math.sqrt(len(X[0])))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = [rng.randint(n) for _ in range(n)]  # bootstrap sample
            tree = RegressionTree(self.max_depth, self.min_samples_leaf, m)
            self.trees.append(tree.fit(X, y, idx, rng))
        return self

    def predict(self, x: Sequence[float]) -> float:
        return math.fsum(t.predict(x) for t in self.trees) / len(self.trees)

    __call__ = predict

    def to_dict(self) -> dict:
        return {"type": "random_forest", "trees": [t.to_dict() for t in self.trees]}

    @classmethod
    def from_dict(cls, data: dict) -> "RandomForest":
        rf = cls(n_trees=len(data["trees"]))
        rf.trees = [RegressionTree.from_dict(td) for td in data["trees"]]
        return rf


class GradientBoosting:
    """Binary gradient-boosted trees on the log-loss (Friedman 2001, "TreeBoost").

    Each round fits a CART regression tree to the residuals y - p and then sets
    every leaf to its Newton step sum(r) / sum(p(1-p)); predictions are
    sigmoid(F0 + learning_rate * sum of tree outputs). Piecewise constant,
    like the random forest, but with a much finer partition.
    """

    def __init__(self, n_trees: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_leaf: int = 10,
                 seed: int = 0) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.f0 = 0.0
        self.trees: list[RegressionTree] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "GradientBoosting":
        rng = Rng(self.seed)
        n = len(X)
        p0 = min(max(math.fsum(y) / n, 1e-6), 1 - 1e-6)
        self.f0 = math.log(p0 / (1 - p0))
        F = [self.f0] * n
        idx = list(range(n))
        self.trees = []
        for _ in range(self.n_trees):
            p = [sigmoid(v) for v in F]
            r = [t - q for t, q in zip(y, p)]
            tree = RegressionTree(self.max_depth, self.min_samples_leaf).fit(X, r, idx, rng)
            num: dict[int, float] = {}
            den: dict[int, float] = {}
            leaves = [tree.leaf(x) for x in X]
            for leaf, ri, pi in zip(leaves, r, p):
                num[leaf] = num.get(leaf, 0.0) + ri
                den[leaf] = den.get(leaf, 0.0) + pi * (1 - pi)
            for leaf in num:
                tree.value[leaf] = num[leaf] / (den[leaf] + 1e-12)
            F = [v + self.learning_rate * tree.value[leaf] for v, leaf in zip(F, leaves)]
            self.trees.append(tree)
        return self

    def predict(self, x: Sequence[float]) -> float:
        return sigmoid(self.f0 + self.learning_rate * math.fsum(t.predict(x) for t in self.trees))

    __call__ = predict

    def to_dict(self) -> dict:
        return {"type": "gradient_boosting", "f0": self.f0, "learning_rate": self.learning_rate,
                "trees": [t.to_dict() for t in self.trees]}

    @classmethod
    def from_dict(cls, data: dict) -> "GradientBoosting":
        gb = cls(n_trees=len(data["trees"]), learning_rate=data["learning_rate"])
        gb.f0 = data["f0"]
        gb.trees = [RegressionTree.from_dict(td) for td in data["trees"]]
        return gb
