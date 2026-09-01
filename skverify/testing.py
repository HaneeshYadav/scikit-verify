"""Check code against the mathematics you believe it implements.

The spec comes from OUTSIDE the code -- a paper, a docstring, your
derivation. skverify traces the function into a formula and compares
the two symbolically, entry by entry. The spec must not be derived
from the trace itself: checking the code against its own output would
always pass.

Minimal scaffold: scalar and entrywise equality specs, the four-tier
verdict, the pytest decorator. Piecewise seams, property rungs beyond
a bare callable, assume-driven normalization and coverage proofs land
on top of this skeleton.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy

from .api import to_sympy
from .helpers import axis_idx


@dataclass
class Verdict:
    """The outcome of one spec check. Never a bare boolean: the tier
    and the shape it was decided at are part of the result."""

    tier: str  # "exact" | "float-constant" | "differs" | "incomplete"
    shape: tuple
    spec: object = None
    traced: object = None
    counterexample: dict = field(default_factory=dict)
    detail: str = ""

    @property
    def matches(self):
        return self.tier in ("exact", "float-constant")

    def message(self):
        head = f"verdict: {self.tier} (at shape {self.shape})"
        if self.matches or self.tier == "incomplete":
            return head + (f"\n  {self.detail}" if self.detail else "")
        lines = [head]
        lines.append(f"  your spec:  {self.spec}")
        lines.append(f"  the code:   {self.traced}")
        if self.counterexample:
            lines.append("  counterexample:")
            for k, v in self.counterexample.items():
                lines.append(f"      {k} = {v}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def check_formula(fn, args, spec, indices=(), assume=(), samples=3):
    """Trace ``fn`` on ``args`` and compare against ``spec`` per entry.

    ``indices`` binds the spec's index symbols to output axes in
    order; ``assume`` carries the derivation's preconditions (stored
    on the verdict today, consumed by normalization as it lands).
    The verdict is decided AT the traced shape -- the same honesty
    concolic testing states about fixed inputs.
    """
    try:
        out = to_sympy(fn, *args)
    except NotImplementedError as e:
        return Verdict(
            tier="incomplete",
            shape=tuple(np.shape(args[0])),
            detail=f"the tracer refused: {e} (a tracer limit, not a code bug)",
        )
    shape = tuple(np.shape(out.value))
    bound = {sym: axis_idx(k) for k, sym in enumerate(indices)}
    spec_b = spec.xreplace(bound) if bound else spec

    entries = list(np.ndindex(shape)) if shape else [()]
    sampled = False
    for entry in entries:
        at = {axis_idx(k): int(v) for k, v in enumerate(entry)}
        t = out.formula.subs(at) if at else out.formula
        s = spec_b.subs(at) if at else spec_b
        verdict, used_sampling = _entry_equal(t, s, out, entry, samples)
        sampled = sampled or used_sampling
        if verdict is not None:
            verdict.shape = shape
            return verdict
    # the tier states HOW the decision was made: exact means every
    # entry's difference vanished symbolically; float-constant means
    # at least one entry needed arbitration at exact sample points
    # (the code computes with rounded irrationals, so symbolic zero
    # is impossible there)
    tier = "float-constant" if sampled else "exact"
    detail = (
        f"{len(entries)}/{len(entries)} entries agree"
        + (
            "; code computes with rounded float constants, agreement "
            f"verified at {samples} exact rational points"
            if sampled
            else ""
        )
    )
    return Verdict(tier=tier, shape=shape, spec=spec, detail=detail)


def _entry_equal(t, s, out, entry, samples):
    """(verdict, used_sampling): verdict is None when the entry
    agrees. Exact tier first, sample-point arbitration second."""
    try:
        d = sympy.simplify(sympy.expand((t - s).doit()))
        if d == 0:
            return None, False
    except Exception:
        pass
    rng = np.random.default_rng(0)
    # unroll reductions FIRST: a spec's Sum binds a dummy, and only
    # after doit() do its terms carry concrete indices a sample point
    # can bind
    td, sd = t.doit(), s.doit()
    slots = sorted(
        {
            e
            for x in (td, sd)
            for e in x.atoms(sympy.Indexed)
            if all(ix.is_Integer for ix in e.indices)
        },
        key=str,
    )
    syms = sorted(
        (td - sd).free_symbols - set(sympy.symbols("i j k l m")), key=str
    )
    agree = True
    point = {}
    for _ in range(samples):
        subs = {
            e: sympy.Rational(int(rng.integers(-300, 300)), 100)
            for e in slots
        }
        for sy in syms:
            if isinstance(sy, sympy.Symbol):
                subs[sy] = sympy.Rational(int(rng.integers(-300, 300)), 100)
        try:
            tv = float(sympy.N(td.xreplace(subs)))
            sv = float(sympy.N(sd.xreplace(subs)))
        except (TypeError, ValueError):
            return Verdict(
                tier="undecided",
                shape=(),
                spec=s,
                traced=t,
                detail=f"entry {entry}: could not decide symbolically or numerically",
            ), True
        if abs(tv - sv) > 1e-10 * max(1.0, abs(sv)):
            agree = False
            point = {str(k): v for k, v in subs.items()}
            point["spec value"] = sv
            point["code value"] = tv
            break
    if agree:
        return None, True  # float-constant tier, labeled by the caller
    return Verdict(
        tier="differs",
        shape=(),
        spec=s,
        traced=t,
        counterexample=point,
        detail=f"first disagreement at entry {entry}",
    ), True


def specifies(spec, indices=(), assume=()):
    """Pytest decorator. The test function RETURNS ``(fn, args)``; the
    trace stays under skverify's control::

        @specifies((x[i] - mean) / std, indices=(i,))
        def test_scale():
            return scale, (np.array([1.0, 4.0, 2.0, 8.0, 5.0]),)
    """

    def deco(test_fn):
        def wrapper():
            fn, args = test_fn()
            v = check_formula(fn, args, spec, indices=indices, assume=assume)
            if v.tier == "incomplete":
                import pytest

                pytest.skip(v.message())
            assert v.matches, v.message()

        wrapper.__name__ = test_fn.__name__
        wrapper.__doc__ = test_fn.__doc__
        return wrapper

    return deco


def _property(prop, assume=()):
    """Property rung: assert a fact about the traced certificate
    itself, no closed form needed::

        @specifies.property(lambda F: sympy.Sum(F[i], (i, 0, 4)) == 0)
        def test_centered():
            return center, (data,)
    """

    def deco(test_fn):
        def wrapper():
            fn, args = test_fn()
            out = to_sympy(fn, *args)
            claim = prop(out.formula)
            if claim in (True, sympy.true):
                return
            d = sympy.simplify(
                (claim.lhs - claim.rhs).doit()
                if isinstance(claim, sympy.Eq)
                else claim
            )
            assert d in (0, sympy.true), (
                f"property does not hold: {claim} (residual: {d})"
            )

        wrapper.__name__ = test_fn.__name__
        return wrapper

    return deco


specifies.property = _property
