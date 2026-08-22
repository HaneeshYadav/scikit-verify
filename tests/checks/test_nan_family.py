"""nan-aware reductions reduce the SURVIVORS, never the raw slots.

Regression net for the nanstd silent-wrong the common-numpy battery
caught: with no registry entry, numpy's own body degraded nanstd to
plain std (the NaN slot counted in the denominator).
"""

import numpy as np
import pytest

from skverify import Pair, to_sympy

rng = np.random.default_rng(11)
V = rng.standard_normal(6)
V[2] = np.nan


@pytest.mark.parametrize(
    "fn",
    [
        np.nansum,
        np.nanmean,
        np.nanstd,
        np.nanvar,
        np.nanprod,
        np.nanmax,
        np.nanmedian,
        lambda a: np.nanstd(a, ddof=1),
        lambda a: np.nanvar(a, ddof=1),
    ],
)
def test_nan_reduction_matches_numpy(fn):
    out = to_sympy(fn, V.copy())
    got = float(out.value if isinstance(out, Pair) else out)
    assert np.isclose(got, float(fn(V)), rtol=1e-9)


def test_nanstd_formula_omits_nan_slot():
    out = to_sympy(np.nanstd, V.copy())
    s = str(out.formula)
    assert "a[2]" not in s
    assert "a[0]" in s and "a[5]" in s


def test_digitize_indices_are_concrete_facts():
    edges = np.sort(rng.uniform(-2, 2, 4))

    def gather(a):
        return a[np.digitize(0.0, np.sort(a)) - 1]

    out = to_sympy(gather, V[np.isfinite(V)].copy())
    ref = gather(V[np.isfinite(V)])
    assert np.isclose(float(out.value), ref)


def test_eye_allocates_traced_and_scatter_lands():
    def h(x):
        A = np.eye(3)
        A[0, 2] = x[0]
        return A

    ref = h(np.array([5.0, 1.0]))
    out = to_sympy(h, np.array([5.0, 1.0]))
    got = np.asarray(out.value if isinstance(out, Pair) else out, dtype=float)
    assert np.allclose(got, ref)


def test_truthy_flag_branch_records_guard():
    def g(x, flag):
        if flag:
            return x.sum() * 2.0
        return x.sum()

    out = to_sympy(g, np.array([1.0, 2.0]), np.float64(1.0))
    assert "Ne(flag, 0)" in str(out.preconditions)
