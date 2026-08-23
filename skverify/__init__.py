from skverify.pair import Pair, IDX
from skverify.maps import numpy as _numpy_map
from skverify.maps import special as _special_map
from .api import to_sympy


def latex(expr):
    """sympy.latex with code-ish names (containing underscores)
    rendered as upright text instead of nested subscripts."""
    import sympy as _sym

    if not isinstance(expr, _sym.Basic):
        return str(expr)
    names = {
        s: r"\mathtt{%s}" % str(s).replace("_", r"\_")
        for s in expr.atoms(_sym.Symbol)
        if "_" in str(s)
    }
    return _sym.latex(expr, symbol_names=names)
