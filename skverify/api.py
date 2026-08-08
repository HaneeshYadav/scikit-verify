"""Public API."""

import inspect

import numpy as np
import sympy

from .pair import Pair


def to_sympy(fn, *args):
    """Run ``fn`` with tracing values.

    Array arguments become indexed formulas named after the function's
    parameters; scalar arguments become symbols of the same name.
    Returns the traced result: ``.formula``, ``.value``, ``.domain``.
    """
    wrapped = [
        (
            Pair(val, sympy.Symbol(name, real=True))
            if np.isscalar(val)
            else Pair.array(name, val)
        )
        for name, val in _infer_names(fn, args)
    ]
    return fn(*wrapped)


def _infer_names(fn, args):
    """Pair each positional argument with its parameter name from fn's signature."""
    names = list(inspect.signature(fn).parameters)
    if len(args) > len(names):
        raise TypeError(f"{fn.__name__} takes {len(names)} arguments, got {len(args)}")
    return zip(names, args)
