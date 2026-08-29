"""table entries + the parametrized differential loop."""

import numpy as np
import pytest
import sympy

from skverify import IDX, Pair
from skverify.registry import UFUNC_TABLE

U = sympy.IndexedBase("u")
V = sympy.IndexedBase("v")
N = sympy.IndexedBase("n")


def make():
    return Pair.array("u", np.random.default_rng(0).uniform(0.15, 0.85, 8))


class TestNamedEntries:
    def test_sin(self):
        u = make()
        assert np.sin(u).formula == sympy.sin(U[IDX])
        assert np.allclose(np.sin(u).value, np.sin(u.value))

    def test_renamed_arcsin(self):
        assert np.arcsin(make()).formula == sympy.asin(U[IDX])

    def test_binary_maximum(self):
        u = make()
        m = np.maximum(u[1:], u[:-1])
        assert m.formula == sympy.Max(U[IDX + 1], U[IDX])
        assert m.domain == (0, 7)

    def test_priority_interop(self):
        u = make()
        assert (2.0 * np.exp(u)).formula == 2.0 * sympy.exp(U[IDX])

ELEMENTWISE = [
    (np_fn, sp_fn)
    for np_fn, sp_fn in UFUNC_TABLE.items()
    if np_fn.nin == 1  # unary only for the loop
]

BINARY = [
    (np_fn, sp_fn)
    for np_fn, sp_fn in UFUNC_TABLE.items()
    if np_fn.nin == 2
]

INTEGER_FIRST = {
    "eval_legendre",
    "eval_chebyt",
    "eval_chebyu",
    "eval_hermite",
    "eval_laguerre",
}

SAFE = {"arccosh": (1.1, 3.0), "arctanh": (-0.9, 0.9)}


def binary_inputs(np_fn):
    u = Pair.array("u", np.linspace(0.25, 0.75, 8))
    v = Pair.array(
        "v",
        np.array([0.25, 0.20, 0.50, 0.40, 0.60, 0.90, 0.70, 0.30]),
    )
    n = Pair.array("n", np.arange(1, 9, dtype=np.int64))

    if np_fn.__name__ == "ldexp":
        return u, n, U, N

    if np_fn.__name__ in INTEGER_FIRST:
        return n, u, N, U

    return u, v, U, V

@pytest.mark.parametrize(
    "np_fn,sp_fn", ELEMENTWISE, ids=[f.__name__ for f, _ in ELEMENTWISE]
)
def test_differential_whole_table(np_fn, sp_fn):
    """Every table entry, forever: formula evaluated == value computed."""
    # NumPy and SymPy may disagree outside the domain.
    lo, hi = SAFE.get(np_fn.__name__, (0.15, 0.85))
    u = Pair.array("u", np.random.default_rng(0).uniform(lo, hi, 8))
    out = np_fn(u)
    for k in range(3):
        evaluated = float(out.formula.subs(U[IDX], u.value[k]))
        assert evaluated == pytest.approx(out.value[k], rel=1e-9)

@pytest.mark.parametrize(
    "np_fn,sp_fn", BINARY, ids=[f.__name__ for f, _ in BINARY]
)
def test_differential_binary_table(np_fn, sp_fn):
    """Every binary table entry: symbolic formula agrees with the computed value."""
    a, b, a_symbol, b_symbol = binary_inputs(np_fn)
    out = np_fn(a, b)

    for k in range(3):
        evaluated = out.formula.subs(
            {
                a_symbol[IDX]: a.value[k],
                b_symbol[IDX]: b.value[k],
            }
        )

        if isinstance(evaluated, sympy.logic.boolalg.Boolean):
            assert bool(evaluated) == bool(out.value[k])
        else:
            assert float(evaluated) == pytest.approx(float(out.value[k]), rel=1e-8)

class TestRefusals:
    def test_frexp_composes_from_exact_pieces(self):
        x = make()
        m, e = np.frexp(x)
        got = np.asarray(m.value, dtype=float) * 2.0 ** np.asarray(e.value, dtype=float)
        assert np.allclose(got, np.asarray(x.value, dtype=float))

    def test_add_reduce_is_sum(self):
        r = np.add.reduce(make())
        assert isinstance(r.formula, sympy.Sum)

    def test_max_reduce_is_lazy_max(self):
        r = np.maximum.reduce(make())
        assert isinstance(r.formula, sympy.Max)
        assert float(r.value) == make().value.max()

    def test_out_into_raw_buffer_writes_through(self):
        u = make()
        buf = np.empty(np.shape(u.value))
        r = np.sin(u, out=buf)
        # the name gets the Pair; the buffer gets the values
        assert hasattr(r, "formula")
        assert np.allclose(buf, np.sin(np.asarray(u.value, dtype=float)))
