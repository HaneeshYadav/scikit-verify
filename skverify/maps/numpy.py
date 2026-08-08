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


FUNCTION_TABLE[np.where] = _where
