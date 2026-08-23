"""The aliasing session, pinned: in-place ops, views, owned buffers."""

import numpy as np

from skverify import Pair, to_sympy


def test_inplace_through_rollaxis_view_updates_parent():
    def mini_scale(x):
        Xr = np.rollaxis(x, 0)
        Xr -= np.nanmean(x, 0)
        Xr /= np.nanstd(x, 0)
        return x

    d = np.array([-0.8, 0.9, 0.7, -0.6, 0.5])
    out = to_sympy(mini_scale, d.copy())
    got = np.asarray(Pair._value_of(out.value), dtype=float)
    assert np.allclose(got, mini_scale(d.copy()))


def test_windowed_slice_is_a_copy_not_a_view():
    def kern(k):
        big = np.zeros((7, 7))
        for r in range(3):
            for c in range(3):
                big[2 * r:2 * r + 3, 2 * c:2 * c + 3] += k[r, c] * k
        return big[2:-2, 2:-2] / big.sum()

    K = np.random.default_rng(5).uniform(0.1, 1.0, (3, 3))
    out = to_sympy(kern, K.copy())
    got = np.asarray(Pair._value_of(out.value), dtype=float)
    assert np.allclose(got, kern(K.copy()))


def test_neutral_copy_owns_its_buffer():
    def f(a):
        b = np.asarray(a)
        b -= 1.0
        return a.sum()  # the original must be untouched

    v = np.array([1.0, 2.0, 3.0])
    out = to_sympy(f, v.copy())
    assert np.isclose(float(out.value), 6.0)


def test_predict_after_inplace_preprocessing():
    pytest = __import__("pytest")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(7)
    X = rng.standard_normal((10, 2))
    y = X @ np.array([0.5, -1.0]) + 0.1

    def fit_predict(a, b):
        return LinearRegression().fit(a, b).predict(a[:3])

    ref = fit_predict(X.copy(), y.copy())
    out = to_sympy(fit_predict, X.copy(), y.copy())
    got = np.asarray(Pair._value_of(out.value if isinstance(out, Pair) else out), dtype=float)
    assert np.allclose(got, ref)
