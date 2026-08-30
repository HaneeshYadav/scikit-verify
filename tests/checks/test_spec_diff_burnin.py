"""Burn-in for the @specifies core loop: handwritten specs diffed
symbolically against traced formulas. Every case here is exactly what
check_formula will do; a spurious nonzero diff means the exit normal
form or the trace drifted from human-written mathematics."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.helpers import axis_idx

I = axis_idx(0)
N = 5
V = sympy.IndexedBase("v")
VALS = np.array([0.7, -1.2, 2.5, 0.3, -0.4])


def _mean():
    j = sympy.Dummy("j", integer=True)
    return sympy.Sum(V[j], (j, 0, N - 1)) / N


def _entrywise_zero(traced, spec, n=N):
    """The check_formula core: per-entry diff at concrete indices."""
    for k in range(n):
        t = traced.subs(I, k) if isinstance(traced, sympy.Basic) else traced
        s = spec.subs(I, k)
        d = sympy.simplify(sympy.expand(t - s).doit())
        assert d == 0, f"entry {k}: {t} vs {s}"


class TestScalarSpecs:
    def test_mean(self):
        out = to_sympy(lambda v: v.mean(), VALS.copy())
        assert sympy.simplify(out.formula.doit() - _mean().doit()) == 0

    def test_norm_squared(self):
        def f(v):
            return (v ** 2).sum()

        out = to_sympy(f, VALS.copy())
        j = sympy.Dummy("j", integer=True)
        spec = sympy.Sum(V[j] ** 2, (j, 0, N - 1))
        assert sympy.simplify(out.formula.doit() - spec.doit()) == 0

    def test_variance_population(self):
        out = to_sympy(lambda v: v.var(), VALS.copy())
        j = sympy.Dummy("j", integer=True)
        spec = sympy.Sum((V[j] - _mean()) ** 2, (j, 0, N - 1)) / N
        d = sympy.simplify(sympy.expand(out.formula.doit() - spec.doit()))
        assert d == 0

    def test_weighted_mean(self):
        W = sympy.IndexedBase("w")
        wvals = np.array([1.0, 2.0, 3.0, 4.0, 0.5])

        def f(v, w):
            return (v * w).sum() / w.sum()

        out = to_sympy(f, VALS.copy(), wvals.copy())
        j = sympy.Dummy("j", integer=True)
        spec = sympy.Sum(V[j] * W[j], (j, 0, N - 1)) / sympy.Sum(
            W[j], (j, 0, N - 1)
        )
        assert sympy.simplify(out.formula.doit() - spec.doit()) == 0

    def test_horner_is_polynomial(self):
        def f(v):
            acc = 0.0
            for k in range(v.shape[0]):
                acc = acc * 2.0 + v[k]
            return acc

        out = to_sympy(f, VALS.copy())
        spec = sum(V[k] * sympy.Integer(2) ** (N - 1 - k) for k in range(N))
        assert sympy.simplify(sympy.expand(out.formula) - spec) == 0


class TestEntrywiseSpecs:
    def test_zscore_float_constant_tier(self):
        # the burn-in finding that shaped the verdict ladder: numpy
        # computes 1/sqrt(5) as a rounded float, so the traced formula
        # is exact about the CODE but differs from exact mathematics
        # at the 16th digit. The honest verdict is tier 2: the diff
        # vanishes numerically at exact sample points, and check_formula
        # must label it "matches (float-constant)", never "exact".
        def f(v):
            return (v - v.mean()) / v.std()

        out = to_sympy(f, VALS.copy())
        j = sympy.Dummy("j", integer=True)
        mean = _mean()
        std = sympy.sqrt(
            sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N
        )
        spec = (V[I] - mean) / std
        rng = np.random.default_rng(7)
        for trial in range(3):
            pt = {V[k]: sympy.Rational(int(x), 100)
                  for k, x in enumerate(rng.integers(-300, 300, N))}
            for k in range(N):
                t = float(out.formula.subs(I, k).doit().xreplace(pt))
                s = float(spec.subs(I, k).doit().xreplace(pt))
                assert abs(t - s) < 1e-10 * max(1.0, abs(s)), (trial, k)

    def test_softmax(self):
        def f(v):
            e = np.exp(v - v.max())
            return e / e.sum()

        out = to_sympy(f, VALS.copy())
        # spec under the traced ordering: max is v[2] for this input
        j = sympy.Dummy("j", integer=True)
        denom = sympy.Sum(sympy.exp(V[j] - V[2]), (j, 0, N - 1))
        spec = sympy.exp(V[I] - V[2]) / denom
        _entrywise_zero(out.formula, spec)

    def test_affine(self):
        def f(v):
            return 3.0 * v + 1.0

        out = to_sympy(f, VALS.copy())
        _entrywise_zero(out.formula, 3 * V[I] + 1)

    def test_diff_spec(self):
        out = to_sympy(lambda v: np.diff(v), VALS.copy())
        _entrywise_zero(out.formula, V[I + 1] - V[I], n=N - 1)


class TestPropertySpecs:
    def test_softmax_normalizes(self):
        # rung 2: a property, no closed form -- entries sum to one
        def f(v):
            e = np.exp(v - v.max())
            return e / e.sum()

        out = to_sympy(f, VALS.copy())
        total = sum(out.formula.subs(I, k) for k in range(N))
        assert sympy.simplify(total.doit()) == 1

    def test_gram_symmetric(self):
        def f(a):
            return a.T @ a

        A = np.arange(6.0).reshape(2, 3) + 1
        out = to_sympy(f, A)
        J = axis_idx(1)
        d = out.formula - out.formula.subs({I: J, J: I}, simultaneous=True)
        assert sympy.simplify(d.doit()) == 0

    def test_centering_kills_the_mean(self):
        # property: centered data has exactly zero mean
        def f(v):
            return v - v.mean()

        out = to_sympy(f, VALS.copy())
        total = sum(out.formula.subs(I, k) for k in range(N))
        assert sympy.simplify(sympy.expand(total.doit())) == 0
