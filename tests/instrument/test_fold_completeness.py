"""Certificates never reference undefined fold symbols.

The accumulator-in-nested-loops pattern once leaked probe and held
symbols into formulas with empty definitions (three stacked causes:
probes embedded in pre-plant scatter targets, inline's fixed pass cap,
and probe repair reintroducing recurrence symbols after inlining).
"""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.session import current as _session


@pytest.mark.parametrize("n,m", [(4, 2), (8, 2), (8, 3), (13, 5), (40, 5)])
def test_accumulator_loops_leave_no_undefined_symbols(n, m):
    def f(x):
        out = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                acc = 0.0
                for k in range(n):
                    acc = acc + x[i % len(x)] * x[k % len(x)]
                out[i, j] = acc
        return out

    x = np.array([2.0, 3.0, 1.5])
    out = to_sympy(f, x.copy())
    defined = set(out.definitions or {})
    for e in (list(out.formula) if isinstance(out.formula, sympy.NDimArray)
              else [out.formula]):
        undefined = {
            s for s in e.free_symbols - defined
            if s in _session.probe_repairs or s in _session.recurrences
            or isinstance(s, sympy.Dummy)
        }
        assert not undefined, f"undefined fold symbols: {undefined}"
    assert np.allclose(np.asarray(out.value, float), f(x.copy()))
