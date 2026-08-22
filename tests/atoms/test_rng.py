"""Random draws seal as distribution-tagged atoms.

The concrete lane keeps the numbers actually drawn (same generator
stream as an untraced run); the symbolic lane gets a sympy.stats
random variable for scalar draws, an IndexedBase declared iid for
arrays. sympy.stats then computes E and variance of the certificate.
"""

import numpy as np
import sympy
import sympy.stats as st

from skverify import to_sympy


def test_scalar_draw_is_random_variable():
    def noisy(x):
        rng = np.random.default_rng(0)
        return x * 3.0 + rng.normal(0.0, 2.0)

    out = to_sympy(noisy, np.float64(1.5))
    x = sympy.Symbol("x", real=True)
    assert st.E(out.formula) == 3.0 * x
    assert st.variance(out.formula) == 4.0
    rng = np.random.default_rng(0)
    assert np.isclose(float(out.value), 1.5 * 3.0 + rng.normal(0.0, 2.0))


def test_array_draw_seals_iid_atom():
    def sim(x):
        rng = np.random.default_rng(1)
        return np.mean(x + rng.normal(0.0, 1.0, 4))

    data = np.array([1.0, 2.0, 3.0, 4.0])
    out = to_sympy(sim, data)
    assert "normal_0" in str(out.formula)
    rng = np.random.default_rng(1)
    assert np.isclose(float(out.value), np.mean(data + rng.normal(0.0, 1.0, 4)))
    notes = [r[-1][1] for r in out.unchecked if r[0] == "normal"]
    assert notes and "iid ~ Normal(0.0, 1.0" in notes[0]


def test_traced_parameter_lifts_into_distribution():
    def scaled(x, s):
        rng = np.random.default_rng(2)
        return x + rng.normal(0.0, s)

    out = to_sympy(scaled, np.float64(1.0), np.float64(0.5))
    s = sympy.Symbol("s", real=True)
    assert st.variance(out.formula) == s**2
    rng = np.random.default_rng(2)
    assert np.isclose(float(out.value), 1.0 + rng.normal(0.0, 0.5))


def test_uniform_and_exponential():
    def mixed(x):
        rng = np.random.default_rng(3)
        return x * rng.uniform(2.0, 4.0) + rng.exponential(2.0)

    out = to_sympy(mixed, np.float64(1.0))
    x = sympy.Symbol("x", real=True)
    assert sympy.simplify(st.E(out.formula) - (3.0 * x + 2.0)) == 0


def test_legacy_module_functions_seal():
    def legacy(x):
        np.random.seed(5)
        return x + np.random.normal(0.0, 1.0)

    out = to_sympy(legacy, np.float64(2.0))
    assert st.E(out.formula) == sympy.Symbol("x", real=True)
    np.random.seed(5)
    assert np.isclose(float(out.value), 2.0 + np.random.normal(0.0, 1.0))


def test_patch_restored_after_trace():
    to_sympy(lambda x: x + np.random.default_rng(0).normal(), np.float64(1.0))
    assert type(np.random.default_rng(0)) is np.random.Generator
    assert isinstance(np.random.normal(), float)


def test_unmapped_draws_stay_concrete_and_exact():
    def pick(x):
        rng = np.random.default_rng(4)
        return x + rng.integers(0, 10)

    out = to_sympy(pick, np.float64(1.0))
    rng = np.random.default_rng(4)
    assert np.isclose(float(out.value), 1.0 + rng.integers(0, 10))
