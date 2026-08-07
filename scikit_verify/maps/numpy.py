"Module to map NumPy Ops to SymPy Ops."
import numpy as np
import sympy 

# Elementwise Operators

# Same, Name Unchanged
_SAME = "sin cos tan sinh cosh tanh exp log sqrt floor sign".split()
_RENAMED = {"arcsin": "asin", "arccos": "acos", "arctan": "atan",
            "arcsinh": "asinh", "arccosh": "acosh", "arctanh": "atanh",
            "absolute": "Abs", "ceil": "ceiling"}

UFUNC_TABLE = {getattr(np, n): getattr(sympy, n) for n in _SAME}
UFUNC_TABLE |= {getattr(np, k): getattr(sympy, v) for k, v in _RENAMED.items()}

# Others
UFUNC_TABLE[np.maximum] = sympy.Max
UFUNC_TABLE[np.minimum] = sympy.Min
UFUNC_TABLE[np.arctan2] = sympy.atan2

