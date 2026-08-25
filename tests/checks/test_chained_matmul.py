"""Chained matmuls over scatter-built matrices trace exactly.

The C.T @ R @ C shape (the penalty-matrix demo) once refused on
sympy's Sum-over-Piecewise hazard; inner sums now unroll with dead
branches pruned, restoring the sparsity the code had.
"""

import numpy as np
import sympy

from skverify import to_sympy


def test_triple_chain_traces_to_closed_form():
    def penalty(t):
        m = len(t) - 4
        D1 = np.zeros((m + 1, m))
        for j in range(m + 1):
            run = t[j + 3] - t[j]
            if run != 0:
                if j - 1 >= 0:
                    D1[j, j - 1] = -3 / run
                if j < m:
                    D1[j, j] = 3 / run
        D2 = np.zeros((m + 2, m + 1))
        for j in range(m + 2):
            run = t[j + 2] - t[j]
            if run != 0:
                if j - 1 >= 0:
                    D2[j, j - 1] = -2 / run
                if j < m + 1:
                    D2[j, j] = 2 / run
        C = D2 @ D1
        R = np.zeros((m + 2, m + 2))
        for p in range(m + 2):
            R[p, p] = (t[p + 2] - t[p]) / 3
            if p + 1 < m + 2:
                R[p, p + 1] = R[p + 1, p] = (t[p + 2] - t[p + 1]) / 6
        return C.T @ R @ C

    t9 = np.array([0., 0., 0., 0., 0.6, 2., 2., 2., 2.])
    ref = penalty(t9.copy())
    out = to_sympy(penalty, t9.copy())
    assert np.allclose(np.asarray(out.value, float), ref)

    from skverify.helpers import axis_idx
    t = sympy.IndexedBase("t")
    pinned = {t[0]: 0, t[1]: 0, t[2]: 0, t[3]: 0,
              t[6]: t[5], t[7]: t[5], t[8]: t[5]}
    e = sympy.cancel(sympy.together(
        out.formula.subs({axis_idx(0): 1, axis_idx(1): 2}).subs(pinned).doit()))
    assert e == -12 / (t[4] ** 2 * t[5])
