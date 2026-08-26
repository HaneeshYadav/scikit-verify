"""Sparse diagonal constructors on traced data: dense in the value
lane, exact in the formula lane. Storage format is bookkeeping.

User POV: code written against scipy.sparse.diags_array traces without
edits, matches the real constructor's dense output entry for entry,
and .tocsc()/.diagonal() behave as the sparse API promises.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.sparse import diags_array

from skverify import Pair, to_sympy


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


CASES = [
    ("tridiagonal", lambda t: diags_array(
        [(t[2:] - t[1:-1]) / 6.0, (t[2:] - t[:-2]) / 3.0],
        offsets=[-1, 0], shape=(4, 4))),
    ("rectangular", lambda t: diags_array(
        [t[:-1], -t[1:]], offsets=[0, -1], shape=(6, 5))),
    ("super", lambda t: diags_array([t[:4]], offsets=[2], shape=(4, 6))),
    ("scalar_offset", lambda t: diags_array(t, offsets=0, shape=(6, 6))),
]


@pytest.mark.parametrize("name,build", CASES, ids=[c[0] for c in CASES])
def test_traced_diags_matches_scipy_dense(name, build):
    t = np.array([0.5, 1.0, 2.0, 3.5, 4.0, 6.0])

    def fn(t):
        M = build(t)
        return M.toarray() if hasattr(M, "toarray") else M

    ref = dense(build(t.copy()))
    out = to_sympy(fn, t.copy())
    got = np.asarray(out.value if isinstance(out, Pair) else out, dtype=float)
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def test_diagonal_offsets_formula_equals_value_per_entry():
    # values matching is not enough: a gather can ship right values
    # with a wrong formula (the diag(-1) index-collapse bug). Every
    # entry's FORMULA must evaluate to that entry's value.
    import sympy

    from skverify.helpers import axis_idx

    def fn(t):
        A = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                A[i, j] = t[i] * 10.0 + t[j]
        return A.diagonal(0), A.diagonal(1), A.diagonal(-2)

    t = np.array([2.0, 3.0, 5.0, 7.0])
    ref = fn(t.copy())
    out = to_sympy(fn, t.copy())
    ts = sympy.IndexedBase("t")
    knots = {ts[k]: t[k] for k in range(4)}
    for g, w in zip(out, ref):
        assert np.allclose(np.asarray(g.value, float), w)
        for pos in range(len(w)):
            fval = g.formula.subs({axis_idx(0): pos}, simultaneous=True)
            fval = float(fval.subs(knots, simultaneous=True).doit())
            assert abs(fval - w[pos]) < 1e-9, (pos, fval, w[pos])


def test_tocsc_is_identity_on_traced():
    def fn(t):
        R = diags_array(t[:3], offsets=0, shape=(3, 3))
        return R.tocsc().diagonal(0)

    t = np.array([1.5, 2.5, 3.5])
    out = to_sympy(fn, t.copy())
    assert np.allclose(np.asarray(out.value, float), t)


def test_untraced_calls_reach_real_constructor():
    # no traced data anywhere: the shim must hand back real scipy sparse
    def fn(t):
        M = diags_array(np.ones(3), offsets=0, shape=(3, 3))
        return t.sum() + M.diagonal(0).sum()

    t = np.array([1.0, 2.0])
    out = to_sympy(fn, t.copy())
    assert np.isclose(float(out.value), 6.0)


def test_chain_through_matmul():
    # the shipped-function shape in miniature: diags -> @ -> diagonal
    def fn(t):
        D = diags_array([t[:3], -t[1:4]], offsets=[0, -1], shape=(4, 3))
        R = diags_array([t[:4]], offsets=[0], shape=(4, 4))
        M = D.T @ R @ D
        return M.diagonal(0)

    t = np.array([0.5, 1.5, 2.5, 3.5])
    ref = fn(t.copy())
    out = to_sympy(fn, t.copy())
    assert np.allclose(np.asarray(out.value, float), dense(ref).ravel())

def test_shim_rejects_what_scipy_rejects():
    # a LIST of diagonals with a scalar offset is a scipy error; the
    # traced path must not be more lenient than the real constructor
    def fn(t):
        return diags_array([t[:3]], offsets=0, shape=(3, 3)).diagonal(0)

    t = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fn(t.copy())          # real scipy rejects
    with pytest.raises(ValueError):
        to_sympy(fn, t.copy())  # traced path rejects identically


def test_banded_penalty_pipeline_closed_form():
    """The full shipped-function shape: masked deboor vectors ->
    diags_array -> chained sparse products -> tocsc -> diagonal loop
    -> banded assembly. The extracted formula must equal the proved
    closed form, not merely the run's values."""
    import sympy

    from skverify.helpers import axis_idx

    def banded_penalty(t):
        order = 4
        m = len(t) - order

        def deboor(o):
            N = len(t) - o
            d = np.zeros(N + 1)
            j = np.arange(N + 1)
            mask = np.where(t[j + o - 1] - t[j] > 0)
            d[mask] = (o - 1) / (t[j[mask] + o - 1] - t[j[mask]])
            return diags_array([d[:N], -d[1:]], offsets=[0, -1],
                               shape=(N + 1, N))

        D1 = deboor(order)
        D2 = deboor(order - 1)
        C = D2 @ D1
        Rn = len(t) - 2
        d0 = (t[2:] - t[:-2]) / 3.0
        d1 = (t[2:-1] - t[1:-2]) / 6.0
        R = diags_array([d1, d0, d1], offsets=[-1, 0, 1], shape=(Rn, Rn))
        omega = (C.T @ R @ C).tocsc()
        banded = np.zeros((4, m))
        for i in range(4):
            banded[i, : m - i] = omega.diagonal(-i)
        return banded

    t9 = np.array([0., 0., 0., 0., 0.6, 2., 2., 2., 2.])
    ref = banded_penalty(t9.copy())
    out = to_sympy(banded_penalty, t9.copy())
    assert np.allclose(np.asarray(out.value, float), dense(ref))

    ts = sympy.IndexedBase("t")
    I, J = axis_idx(0), axis_idx(1)
    pinned = {ts[0]: 0, ts[1]: 0, ts[2]: 0, ts[3]: 0,
              ts[6]: ts[5], ts[7]: ts[5], ts[8]: ts[5]}
    e = out.formula.subs({I: 1, J: 1}, simultaneous=True).subs(pinned)
    e = sympy.cancel(sympy.together(sympy.nsimplify(e.doit(), rational=True)))
    assert e == -12 / (ts[4] ** 2 * ts[5])
