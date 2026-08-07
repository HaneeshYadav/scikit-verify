import numpy as np
import sympy

from .maps.numpy import (
    UFUNC_TABLE,
)

IDX = sympy.Symbol("i", integer=True)


class Pair:
    """Convert math and array style operations to SymPy
    expressions.
    """
    def __init__(self, value, formula, domain=None):
        self.value = value  # the real ndarray/scalar, what executes
        self.formula = formula  # the sympy Expr, what it means
        self.domain = domain  # When this needs to behave like an Array.

    @staticmethod
    def _formula_of(x):
        """Formula for an operand: Pair => its formula, raw number => sympy number."""
        return x.formula if isinstance(x, Pair) else sympy.sympify(x)

    @staticmethod
    def _domain_of(x):
        return x.domain if isinstance(x, Pair) else None

    @staticmethod
    def _merge_domains(*domains):
        """Merge raw domain tuples. None = scalar (compatible with anything).
        All non-None domains must be identical."""
        result = None
        for d in domains:
            if d is None:
                continue
            if result is None:
                result = d
            elif d != result:
                raise ValueError(f"domain mismatch: {result} vs {d}")
        return result

    @staticmethod
    def _binary(inputs, fwd, rev, self):
        a, b = inputs
        return fwd(a, b) if a is self else rev(b, a)

    @staticmethod
    def _value_of(x):
        return x.value if isinstance(x, Pair) else x

    def __add__(self, other):
        return Pair(
            value=self.value + Pair._value_of(other),
            formula=self.formula
            + Pair._formula_of(other),  # sympy dunder does the rest
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __radd__(self, other):  # handles  2 + u
        return self.__add__(other)

    def __sub__(self, other):
        return Pair(
            value=self.value - Pair._value_of(other),
            formula=self.formula - Pair._formula_of(other),
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __rsub__(self, other):  # handles  2 - u   (order matters!)
        return Pair(
            value=Pair._value_of(other) - self.value,
            formula=Pair._formula_of(other) - self.formula,
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __mul__(self, other):
        return Pair(
            value=self.value * Pair._value_of(other),
            formula=self.formula * Pair._formula_of(other),
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    __rmul__ = __mul__

    def __abs__(self):
        return Pair(abs(self.value), sympy.Abs(self.formula))

    def __bool__(self):
        raise NotImplementedError(
            "data-dependent branch on a traced value"  # guard-logging comes later
        )

    def __truediv__(self, other):       # self / other  (if not added yet)
        return Pair(
            value=self.value / Pair._value_of(other),
            formula=self.formula / Pair._formula_of(other),
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __rtruediv__(self, other):      # other / self
        return Pair(
            value=Pair._value_of(other) / self.value,
            formula=Pair._formula_of(other) / self.formula,
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __pow__(self, other):           # self ** other
        return Pair(
            value=self.value ** Pair._value_of(other),
            formula=self.formula ** Pair._formula_of(other),
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __rpow__(self, other):          # other ** self
        return Pair(
            value=Pair._value_of(other) ** self.value,
            formula=Pair._formula_of(other) ** self.formula,
            domain=Pair._merge_domains(self, Pair._domain_of(other)),
        )

    def __neg__(self):                  # -self
        return Pair(-self.value, -self.formula)

    @classmethod
    def array(cls, name, value):
        value = np.asarray(value)
        if value.ndim != 1:
            raise NotImplementedError(
                "Currently only 1D array's are supported.",
            )
        return cls(value, sympy.IndexedBase(name)[IDX], domain=(0, len(value)))

    def __len__(self):
        n = self.value.shape[0]                       # truth: the real array
        assert n == self.domain[1] - self.domain[0], "domain drifted from value"
        return n

    def __getitem__(self, key):
        """Handle slicing of unstrided arrays."""
        if self.domain is None:
            raise TypeError("scalar Pair is not subscriptable")
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise NotImplementedError("only step-1 slices are supported for now.")
        n = len(self)
        start = key.start or 0
        stop = key.stop if key.stop is not None else n
        if start < 0: start += n
        if stop < 0: stop += n
        return Pair(
            value=self.value[key],
            formula=self.formula.subs(IDX, IDX + start),
            domain=(0, stop - start),
        )

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        for input in inputs:
            if isinstance(input, np.ndarray):
                if input.ndim > 1:
                    raise NotImplementedError("")

        if method != "__call__" or kwargs.get("out") is not None:
            raise NotImplementedError(f"{ufunc.__name__}.{method} not supported")

        if ufunc is np.add:        return Pair._binary(inputs, Pair.__add__, Pair.__radd__, self)
        if ufunc is np.subtract:   return Pair._binary(inputs, Pair.__sub__, Pair.__rsub__, self)
        if ufunc is np.multiply:   return Pair._binary(inputs, Pair.__mul__, Pair.__rmul__, self)
        if ufunc is np.true_divide:return Pair._binary(inputs, Pair.__truediv__, Pair.__rtruediv__, self)
        if ufunc is np.power:      return Pair._binary(inputs, Pair.__pow__, Pair.__rpow__, self)
        if ufunc is np.negative:   return -inputs[0]

        target = UFUNC_TABLE.get(ufunc)
        if target is None:
            raise NotImplementedError(f"ufunc {ufunc.__name__} not mapped")

        values   = [Pair._value_of(x) for x in inputs]
        formulas = [Pair._formula_of(x) for x in inputs]    
        domain   = Pair._merge_domains(*(Pair._domain_of(x) for x in inputs))

        return Pair(ufunc(*values), target(*formulas), domain)
