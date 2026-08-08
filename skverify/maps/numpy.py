"Module to map NumPy Ops to SymPy Ops."

import numpy as np
import sympy

from ..registry import (
    UFUNC_TABLE,
    FUNCTION_TABLE,
)

from ..pair import Pair, IDX

# UFUNCs, Elementwise
_SAME = "sin cos tan sinh cosh tanh exp log sqrt floor sign".split()
_RENAMED = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "absolute": "Abs",
    "ceil": "ceiling",
}

UFUNC_TABLE.update({getattr(np, n): getattr(sympy, n) for n in _SAME})
UFUNC_TABLE.update({getattr(np, k): getattr(sympy, v) for k, v in _RENAMED.items()})

# Others
UFUNC_TABLE[np.maximum] = sympy.Max
UFUNC_TABLE[np.minimum] = sympy.Min
UFUNC_TABLE[np.arctan2] = sympy.atan2


# FUNCTION TABLE (non-UFUNCS)


def _where(cond, a, b):
    domain = Pair._merge_domains(
        Pair._domain_of(cond), Pair._domain_of(a), Pair._domain_of(b)
    )
    cond_f = Pair._formula_of(cond)
    if not isinstance(
        cond_f, (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational)
    ):
        cond_f = sympy.Ne(cond_f, 0)
    return Pair(
        np.where(Pair._value_of(cond), Pair._value_of(a), Pair._value_of(b)),
        sympy.Piecewise((Pair._formula_of(a), cond_f), (Pair._formula_of(b), True)),
        domain,
    )


def _sum(a, axis=None, **kwargs):
    if kwargs:
        raise NotImplementedError(f"np.sum kwargs {list(kwargs)} not supported")
    if not isinstance(a, Pair) or a.domain is None:
        return np.sum(a)  # plain input, not ours
    if axis not in (None, 0):
        raise NotImplementedError("axis reduction beyond 1-D arrives with N-D")
    lo, hi = a.domain
    j = sympy.Symbol("j", integer=True)
    return Pair(
        np.sum(a.value),
        sympy.Sum(a.formula.subs(IDX, j), (j, lo, hi - 1)),  # Sum bounds INCLUSIVE
        None,  # 1-D reduced fully, scalar
    )


def _zeros(shape, **kw):
    if not isinstance(shape, (int, np.integer)):
        raise NotImplementedError("N-D creation arrives with N-D")
    return Pair(np.zeros(shape), sympy.Integer(0), (0, int(shape)))


FUNCTION_TABLE[np.sum] = _sum
FUNCTION_TABLE[np.where] = _where
