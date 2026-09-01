"""@specifies against real scipy functions, module by module.

Every function here is shipped scipy implemented in Python + NumPy,
and every spec is transcribed from its docstring or the textbook
definition -- never from the trace. The asserted tiers are the
honest ones: exact where the code computes with rationals,
float-constant where numpy rounds an irrational, undecided where the
tracer's formula cannot be interrogated yet.
"""

import numpy as np
import pytest
import sympy

pytest.importorskip("scipy")

import scipy.integrate as si
import scipy.signal as sg
import scipy.spatial.distance as sdist
import scipy.special as sp
import scipy.stats as st

from skverify.testing import check_formula, specifies

N = 5
V = sympy.IndexedBase("v")
i = sympy.Symbol("i", integer=True)
VALS = np.array([0.7, 1.2, 2.5, 0.3, 0.4])  # positive: gmean/hmean domains


def _j():
    return sympy.Dummy("j", integer=True)


def _mean():
    j = _j()
    return sympy.Sum(V[j], (j, 0, N - 1)) / N


class TestStats:
    def test_zscore(self):
        # docstring: (x - mean) / std, population ddof=0. numpy rounds
        # 1/sqrt(5), so exact symbolic equality is impossible: the
        # float-constant tier states that honestly.
        j = _j()
        mean = _mean()
        std = sympy.sqrt(sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N)
        v = check_formula(
            lambda v: st.zscore(v), (VALS.copy(),),
            (V[i] - mean) / std, indices=(i,),
        )
        assert v.tier == "float-constant" and v.matches
        assert v.shape == (N,)

    def test_gmean(self):
        # geometric mean: exp of the mean of logs -- exactly
        j = _j()
        spec = sympy.exp(sympy.Sum(sympy.log(V[j]), (j, 0, N - 1)) / N)
        v = check_formula(lambda v: st.gmean(v), (VALS.copy(),), spec)
        assert v.tier == "exact"

    def test_hmean_on_its_stated_domain(self):
        # harmonic mean: n / sum(1/x), claimed for positive data only
        # (scipy returns nan otherwise -- the traced Piecewise says
        # so). assume= draws the arbitration points on the domain.
        j = _j()
        spec = N / sympy.Sum(1 / V[j], (j, 0, N - 1))
        v = check_formula(
            lambda v: st.hmean(v), (VALS.copy(),), spec,
            assume=[V[k] > 0 for k in range(N)],
        )
        assert v.matches

    def test_hmean_off_domain_does_not_fake_a_pass(self):
        # scipy versions differ here: newer ones return nan for
        # negative data (the trace is a Piecewise over sign patterns),
        # older ones compute n/sum(1/x) unconditionally. The honest
        # assertion follows what THIS scipy's trace contains.
        from skverify import to_sympy

        out = to_sympy(lambda v: st.hmean(v), VALS.copy())
        branched = isinstance(out.formula, sympy.Basic) and out.formula.has(
            sympy.Piecewise
        )
        j = _j()
        spec = N / sympy.Sum(1 / V[j], (j, 0, N - 1))
        v = check_formula(lambda v: st.hmean(v), (VALS.copy(),), spec)
        if branched:
            # per-path semantics: arbitration samples WITHIN the traced
            # path's guards, so the spec matches on the positive path it
            # was traced on -- and the restriction is DISCLOSED: the
            # verdict cannot be an unqualified match
            assert v.matches
            assert v.tier in ("sampled", "float-constant")
            assert out.preconditions is not sympy.true
        else:
            # branch-free implementation IS the spec, everywhere
            assert v.tier == "exact"


class TestIntegrate:
    def test_trapezoid(self):
        # docstring: unit spacing, (y[0] + y[4])/2 + y[1] + y[2] + y[3]
        @specifies(V[0] / 2 + V[1] + V[2] + V[3] + V[4] / 2)
        def check():
            return (lambda v: si.trapezoid(v)), (VALS.copy(),)

        check()

    def test_simpson(self):
        # composite Simpson on five points, unit spacing:
        # (y0 + 4 y1 + 2 y2 + 4 y3 + y4) / 3
        spec = (V[0] + 4 * V[1] + 2 * V[2] + 4 * V[3] + V[4]) / 3
        v = check_formula(lambda v: si.simpson(v), (VALS.copy(),), spec)
        assert v.tier == "exact"

    def test_simpson_wrong_weights_differ(self):
        # the classic transcription slip: trapezoid weights passed
        # off as Simpson. The verdict names a concrete disagreement.
        wrong = V[0] / 2 + V[1] + V[2] + V[3] + V[4] / 2
        v = check_formula(lambda v: si.simpson(v), (VALS.copy(),), wrong)
        assert v.tier == "differs"
        assert "spec value" in v.counterexample


class TestSpecial:
    def test_softmax(self):
        # exp(v - max) / sum(exp(v - max)); on this input the max is
        # v[2] and the traced guard states that ordering
        j = _j()
        den = sympy.Sum(sympy.exp(V[j] - V[2]), (j, 0, N - 1))
        v = check_formula(
            lambda v: sp.softmax(v), (VALS.copy(),),
            sympy.exp(V[i] - V[2]) / den, indices=(i,),
        )
        assert v.tier == "exact"
        assert v.shape == (N,)

    def test_softmax_normalizes(self):
        # rung 2: no closed form needed -- the entries sum to one
        @specifies.property(
            lambda F: sympy.Eq(
                sum(F[k] for k in range(N))
                if isinstance(F, sympy.NDimArray)
                else sum(F.subs(i, k) for k in range(N)),
                1,
            )
        )
        def check():
            return (lambda v: sp.softmax(v)), (VALS.copy(),)

        check()

    def test_logsumexp(self):
        # log(sum(exp(v))): scipy computes via the max-shift trick;
        # the shift cancels mathematically but leaves float constants
        j = _j()
        spec = sympy.log(sympy.Sum(sympy.exp(V[j]), (j, 0, N - 1)))
        v = check_formula(lambda v: sp.logsumexp(v), (VALS.copy(),), spec)
        assert v.matches


class TestSpatial:
    def test_euclidean(self):
        U, W = sympy.IndexedBase("u"), sympy.IndexedBase("w")
        k = _j()
        spec = sympy.sqrt(sympy.Sum((U[k] - W[k]) ** 2, (k, 0, 2)))
        v = check_formula(
            lambda u, w: sdist.euclidean(u, w),
            (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 5.0])),
            spec,
        )
        assert v.matches  # float-constant: numpy's rounded sqrt

    def test_cosine_is_honestly_undecided(self):
        # KNOWN TRACER LIMIT, pinned so it cannot silently change:
        # the cosine trace contains Function('sqrt') -- an undefined
        # lookalike, not sympy.sqrt -- so the check can neither prove
        # nor refute. The contract that matters: never a fake pass.
        U, W = sympy.IndexedBase("u"), sympy.IndexedBase("w")
        k = _j()
        dot = sympy.Sum(U[k] * W[k], (k, 0, 2))
        nu = sympy.sqrt(sympy.Sum(U[k] ** 2, (k, 0, 2)))
        nw = sympy.sqrt(sympy.Sum(W[k] ** 2, (k, 0, 2)))
        v = check_formula(
            lambda u, w: sdist.cosine(u, w),
            (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 5.0])),
            1 - dot / (nu * nw),
        )
        assert v.tier == "undecided"
        assert not v.matches


class TestSignal:
    def test_detrend_constant(self):
        # type="constant" detrending is exactly mean removal
        v = check_formula(
            lambda v: sg.detrend(v, type="constant"), (VALS.copy(),),
            V[i] - _mean(), indices=(i,),
        )
        assert v.tier == "exact"
        assert v.shape == (N,)

    def test_detrend_removes_the_mean(self):
        # the property the docstring implies: output sums to zero
        @specifies.property(
            lambda F: sympy.Eq(sum(F.subs(i, k) for k in range(N)), 0)
        )
        def check():
            return (lambda v: sg.detrend(v, type="constant")), (VALS.copy(),)

        check()


class TestHonestRefusals:
    """Functions chosen because they DON'T trace: what a user hits
    when pointing @specifies at arbitrary scipy. The contract is a
    skip that blames the tracer, never a failure and never a pass."""

    def test_pearsonr_incomplete(self):
        U, W = sympy.IndexedBase("u"), sympy.IndexedBase("w")
        v = check_formula(
            lambda u, w: st.pearsonr(u, w).statistic,
            (np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 5.0, 3.0])),
            U[0] * W[0],  # spec content irrelevant: never reached
        )
        assert v.tier == "incomplete"

    def test_iqr_incomplete(self):
        v = check_formula(lambda v: st.iqr(v), (VALS.copy(),), _mean())
        assert v.tier == "incomplete"

    def test_decorator_skips_not_fails(self):
        @specifies(V[0])
        def check():
            return (lambda v: st.iqr(v)), (VALS.copy(),)

        with pytest.raises(pytest.skip.Exception):
            check()


class TestInterpolate:
    """The north-star module. Coefficient and evaluation claims that
    hold exactly, plus the algebraic crown: two different
    interpolation algorithms proven to compute the same polynomial."""

    XS = np.array([0.0, 1.0, 2.0, 3.0])
    YS = np.array([1.0, 3.0, 2.0, 5.0])

    def test_linear_spline_coefficients_are_the_data(self):
        # k=1 B-spline interpolation: the coefficients ARE the values
        import scipy.interpolate as ip

        Y = sympy.IndexedBase("y")
        v = check_formula(
            lambda x, y: ip.make_interp_spline(x, y, k=1).c,
            (self.XS.copy(), self.YS.copy()),
            Y[i], indices=(i,),
        )
        assert v.tier == "exact"

    def test_cubicspline_constant_row_interpolates(self):
        # c[3, j] is S(x_j): the spline passes through the data,
        # stated as a formula rather than a float comparison
        import scipy.interpolate as ip

        Y = sympy.IndexedBase("y")
        v = check_formula(
            lambda x, y: ip.CubicSpline(x, y).c[3],
            (self.XS.copy(), self.YS.copy()),
            Y[i], indices=(i,),
        )
        assert v.tier == "exact"

    def test_krogh_equals_lagrange(self):
        # KroghInterpolator computes via Newton divided differences;
        # the Lagrange formula is a different algorithm for the same
        # polynomial. Traced Newton == textbook Lagrange, exactly:
        # a theorem between two algorithms, not a point check.
        import scipy.interpolate as ip

        X, Y = sympy.IndexedBase("x"), sympy.IndexedBase("y")
        half = sympy.Rational(1, 2)
        lagrange = sum(
            Y[jj]
            * sympy.prod(
                [(half - X[m]) / (X[jj] - X[m]) for m in range(4) if m != jj]
            )
            for jj in range(4)
        )
        v = check_formula(
            lambda x, y: ip.KroghInterpolator(x, y)(0.5),
            (self.XS.copy(), self.YS.copy()),
            lagrange,
        )
        assert v.tier == "exact"
