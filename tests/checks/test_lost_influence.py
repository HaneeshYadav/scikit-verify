"""A constant formula from data-dependent code must refuse, never
certify. Legitimate constants (code that really ignores the data)
must keep lifting."""

import numpy as np
import pytest

from skverify import to_sympy


class TestLostInfluenceRefusal:
    def test_scipy_sem_refuses_loudly(self):
        # stats.sem strips the traced array in scipy's array-api layer;
        # the trace watches plain numpy and produces a constant
        stats = pytest.importorskip("scipy.stats")

        def f(v):
            return stats.sem(v)

        with pytest.raises(NotImplementedError, match="constant"):
            to_sympy(f, np.array([0.7, -1.2, 2.5, 0.3, -0.4]))


class TestLegitimateConstantsStillLift:
    def test_data_cancels_exactly(self):
        def f(v):
            return (v - v).sum()  # always 0, honestly

        out = to_sympy(f, np.array([1.0, 2.0, 3.0]))
        assert float(out.value) == 0.0

    def test_shape_only(self):
        def f(v):
            return float(v.shape[0]) * 2.0

        out = to_sympy(f, np.array([1.0, 2.0, 3.0]))
        assert float(out.value) == 6.0

    def test_zeros_like_sum(self):
        def f(v):
            return np.zeros_like(v).sum()

        out = to_sympy(f, np.array([4.0, 5.0]))
        assert float(out.value) == 0.0
