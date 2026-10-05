"""Small, dependency-free statistics for paired model comparisons.

Everything here works on plain sequences of floats and a seeded
``numpy.random.Generator``, so results are reproducible run to run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_PERMUTATIONS = 10_000


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict:
        return asdict(self)


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def bootstrap_mean(values: Sequence[float], n_boot: int = DEFAULT_BOOTSTRAP, alpha: float = 0.05,
                   seed: int | None = 0) -> Interval:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    idx = _rng(seed).integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(x.mean()), float(lo), float(hi), int(x.size))


def bootstrap_ratio(num: Sequence[float], den: Sequence[float], n_boot: int = DEFAULT_BOOTSTRAP,
                    alpha: float = 0.05, seed: int | None = 0,
                    transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Interval:
    """CI for sum(num)/sum(den), resampling items (e.g. total NLL / total tokens per text piece).

    ``transform`` is applied to the ratio (e.g. ``np.exp`` to turn mean NLL into perplexity).
    """
    a, b = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if a.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    f = transform or (lambda v: v)
    idx = _rng(seed).integers(0, a.size, size=(n_boot, a.size))
    ratios = f(a[idx].sum(axis=1) / b[idx].sum(axis=1))
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return Interval(float(f(np.asarray(a.sum() / b.sum()))), float(lo), float(hi), int(a.size))


def bootstrap_diff_of_means(x: Sequence[float], y: Sequence[float], n_boot: int = DEFAULT_BOOTSTRAP,
                            alpha: float = 0.05, seed: int | None = 0,
                            clusters_x: Sequence[str] | None = None) -> Interval:
    """CI for mean(x) - mean(y), x and y independent samples (resampled separately).

    With ``clusters_x``, x is resampled by whole clusters (e.g. pieces from one book).
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = _rng(seed)
    if clusters_x is not None:
        sn, sd = _cluster_sums(a, [1.0] * a.size, clusters_x)
        idx = rng.integers(0, sn.size, size=(n_boot, sn.size))
        ma = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    else:
        ma = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    lo, hi = np.quantile(ma - mb, [alpha / 2, 1 - alpha / 2])
    return Interval(float(a.mean() - b.mean()), float(lo), float(hi), int(min(a.size, b.size)))


def sign_test(diffs: Sequence[float], tol: float = 1e-9) -> dict:
    """Exact two-sided binomial sign test of median(diffs) == 0. Ties (|d| <= tol) are dropped."""
    pos = sum(1 for d in diffs if d > tol)
    neg = sum(1 for d in diffs if d < -tol)
    n = pos + neg
    if n == 0:
        return {"positive": 0, "negative": 0, "ties": len(diffs), "p_value": 1.0}
    k = min(pos, neg)
    # P(X <= k) for X ~ Bin(n, 1/2), doubled; exact integer arithmetic, then one division.
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = min(1.0, 2 * tail / 2 ** n)
    return {"positive": pos, "negative": neg, "ties": len(diffs) - n, "p_value": p}


def paired_permutation_test(diffs: Sequence[float], n_perm: int = DEFAULT_PERMUTATIONS,
                            seed: int | None = 0) -> float:
    """Two-sided sign-flip permutation p-value for mean(diffs) == 0.

    Exact enumeration when there are at most 16 differences, Monte Carlo otherwise
    (with the +1 correction so p is never exactly 0).
    """
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    if d.size <= 16:
        signs = np.array([[1 if (m >> i) & 1 else -1 for i in range(d.size)] for m in range(2 ** d.size)])
        null = np.abs((signs * d).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12))
    signs = _rng(seed).choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_perm + 1))


def two_sample_permutation_test(x: Sequence[float], y: Sequence[float], n_perm: int = DEFAULT_PERMUTATIONS,
                                seed: int | None = 0) -> float:
    """Two-sided permutation p-value for mean(x) == mean(y) (labels shuffled)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    rng = _rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        if abs(p[: a.size].mean() - p[a.size:].mean()) >= observed - 1e-12:
            count += 1
    return float((count + 1) / (n_perm + 1))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n)


def binomial_test_greater(successes: int, n: int, p0: float) -> float:
    """One-sided exact binomial p-value P(X >= successes) under X ~ Bin(n, p0)."""
    if n == 0:
        return 1.0
    return float(min(1.0, sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(successes, n + 1))))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    return sign_test([1.0] * b + [-1.0] * c)["p_value"]


def _cluster_sums(num: Sequence[float], den: Sequence[float], clusters: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    keys = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    sn, sd = np.zeros(len(keys)), np.zeros(len(keys))
    for x, w, c in zip(num, den, clusters):
        sn[keys[c]] += x
        sd[keys[c]] += w
    return sn, sd


def cluster_bootstrap_ratio(num: Sequence[float], den: Sequence[float], clusters: Sequence[str],
                            n_boot: int = DEFAULT_BOOTSTRAP, alpha: float = 0.05, seed: int | None = 0,
                            transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Interval:
    """Like bootstrap_ratio, but resamples whole clusters (items sharing a word, pieces sharing a book).

    Items in one cluster are not independent evidence, so this interval is the
    honest one when clusters exist. ``n`` in the result is the number of clusters.
    """
    if len(num) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    sn, sd = _cluster_sums(num, den, clusters)
    f = transform or (lambda v: v)
    idx = _rng(seed).integers(0, sn.size, size=(n_boot, sn.size))
    ratios = f(sn[idx].sum(axis=1) / sd[idx].sum(axis=1))
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return Interval(float(f(np.asarray(sn.sum() / sd.sum()))), float(lo), float(hi), int(sn.size))


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[str], **kw) -> Interval:
    return cluster_bootstrap_ratio(values, [1.0] * len(values), clusters, **kw)


def cluster_permutation_test(diffs: Sequence[float], clusters: Sequence[str], n_perm: int = DEFAULT_PERMUTATIONS,
                             seed: int | None = 0) -> float:
    """Sign-flip test of mean(diffs) == 0 that flips whole clusters together."""
    if len(diffs) == 0:
        return 1.0
    sn, _ = _cluster_sums(diffs, [1.0] * len(diffs), clusters)
    return paired_permutation_test(sn, n_perm=n_perm, seed=seed)
