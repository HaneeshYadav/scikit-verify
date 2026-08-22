"""Mechanisms priced by the wild web sample, pinned."""

import numpy as np
import sympy

from skverify import Pair, to_sympy

rng = np.random.default_rng(7)


def test_argsort_scatter_ranks():
    def compute_ranks(x):
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    x = np.array([3.0, 1.0, 2.0])
    out = to_sympy(compute_ranks, x)
    got = out.value if isinstance(out, Pair) else out
    assert np.array_equal(np.asarray(Pair._value_of(np.asarray(got, dtype=object)) if not isinstance(out, Pair) else out.value, dtype=float), compute_ranks(x))


def test_gradient_2d_matches():
    A = rng.uniform(0.5, 2.0, (4, 3))

    def g2(a):
        gy, gx = np.gradient(a)
        return gy.sum() + (gx**2).sum()

    out = to_sympy(g2, A.copy())
    assert np.isclose(float(out.value), g2(A))


def test_eigh_seals_with_role_names():
    A = rng.uniform(0.5, 2.0, (3, 3))

    def f(a):
        w, v = np.linalg.eigh(a @ a.T)
        return w.sum()

    out = to_sympy(f, A.copy())
    assert np.isclose(float(out.value), f(A))
    assert "eigh" in str(out.formula)


def test_tuple_operand_adds_like_numpy():
    def f(c):
        return np.cumsum((0.0,) + c).sum()

    v = rng.uniform(0.5, 2.0, 4)
    out = to_sympy(f, v.copy())
    assert np.isclose(float(out.value), f(v))


def test_linspace_traced_endpoints_exact():
    def f(x):
        return np.linspace(x.min(), x.max(), 5).sum()

    v = rng.uniform(0.5, 2.0, 6)
    out = to_sympy(f, v.copy())
    assert np.isclose(float(out.value), f(v))
    assert isinstance(out.formula, sympy.Basic)
