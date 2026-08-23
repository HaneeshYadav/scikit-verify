"""The obvious hypothesis x skverify uses, and nothing clever.

explore(fn, like)      -- one witness input per distinct certificate
                          path: hypothesis draws, skverify signs each
                          draw with its preconditions, new signature =
                          new path kept.
edge_cases(fn, like)   -- inputs sitting exactly ON precondition
                          boundaries (ties, zero denominators): each
                          guard is solved for one input element and
                          the draw is projected onto the boundary.
verify(fn, *args)      -- for use inside your own @given test: trace,
                          compare against the untraced run, fail loud.
"""

import numpy as np
import sympy
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from skverify import Pair, to_sympy


def _like_strategy(a, lo=-3.0, hi=3.0):
    if isinstance(a, np.ndarray):
        return arrays(
            np.float64,
            a.shape,
            elements=st.floats(lo, hi, allow_nan=False, width=64),
        )
    return st.floats(lo, hi, allow_nan=False, width=64)


def _guards(out):
    pre = getattr(out, "preconditions", sympy.true)
    if pre is sympy.true:
        return []
    return list(pre.args) if isinstance(pre, sympy.And) else [pre]


def explore(fn, like, max_examples=200, lo=-3.0, hi=3.0, per_draw_seconds=15):
    """One witness per distinct certificate path.

    Draws inputs shaped like `like` (a tuple of example arguments),
    traces each, and keeps the first witness of every new
    precondition signature. Returns [(args, certificate), ...].
    Refusals and errors are counted, never raised.
    """
    import signal

    rng = np.random.default_rng(0)

    def draw_one(a):
        if isinstance(a, np.ndarray):
            return rng.uniform(lo, hi, a.shape)
        return float(rng.uniform(lo, hi))

    seen, found, skipped = set(), [], 0
    last_reason = None
    dry = 0

    class _TO(Exception):
        pass

    old_handler = signal.signal(
        signal.SIGALRM, lambda s_, f_: (_ for _ in ()).throw(_TO())
    )
    try:
        for _ in range(max_examples):
            if dry >= 25:
                break  # 25 draws with nothing new: the paths have dried up
            args = tuple(draw_one(a) for a in like)
            signal.alarm(per_draw_seconds)
            try:
                out = to_sympy(fn, *[np.copy(a) if isinstance(a, np.ndarray)
                                     else a for a in args])
            except _TO:
                skipped += 1
                last_reason = f"trace exceeded {per_draw_seconds}s"
                dry += 1
                continue
            except Exception as e:
                skipped += 1
                last_reason = f"{type(e).__name__}: {str(e)[:120]}"
                dry += 1
                continue
            finally:
                signal.alarm(0)
            sig = str(getattr(out, "preconditions", sympy.true))
            if sig not in seen:
                seen.add(sig)
                found.append((args, out))
                dry = 0
            else:
                dry += 1
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    if skipped:
        # silence hides problems: say what happened and why
        print(f"[skverify-hypothesis] {skipped} draws did "
              f"not trace; last reason: {last_reason}")
    return found


def edge_cases(fn, like, max_paths=20, lo=-3.0, hi=3.0, paths=None):
    """Inputs exactly ON precondition boundaries.

    For every guard discovered by explore(), solve its equality form
    for one input element and project a drawn input onto the
    boundary: `a[0] <= a[2]` yields an input with a[0] == a[2], a
    `Ne(d, 0)` guard yields one with d == 0. Returns
    [(guard, args, outcome), ...] where outcome is the REAL
    function's behavior at the boundary (value or exception).
    """
    if paths is None:
        paths = explore(fn, like, max_examples=max_paths * 10, lo=lo, hi=hi)
    param_names = [
        getattr(a, "name", None) for a in ()
    ]
    cases = []
    done = set()
    for args, out in paths:
        for g in _guards(out):
            key = str(g)
            if key in done:
                continue
            done.add(key)
            expr = None
            if isinstance(g, sympy.Not) and isinstance(g.args[0], (sympy.Eq, sympy.Ne)):
                g = g.args[0]
            if isinstance(g, sympy.Eq):
                # indicator equality: Eq(Piecewise((1, cond), (0, True)), v)
                # is just cond (or its negation); the boundary is cond's
                for side in (g.lhs, g.rhs):
                    if (
                        isinstance(side, sympy.Piecewise)
                        and len(side.args) == 2
                        and side.args[0][0] == 1
                    ):
                        g = side.args[0][1]
                        break
            if isinstance(g, (sympy.StrictLessThan, sympy.LessThan,
                              sympy.StrictGreaterThan, sympy.GreaterThan,
                              sympy.Eq, sympy.Ne)):
                expr = g.lhs - g.rhs
            if expr is None:
                continue
            try:
                expr = expr.doit()  # finite Sums become indexed terms
            except Exception:
                pass
            idx = sorted(expr.atoms(sympy.Indexed), key=str)
            syms = sorted(expr.atoms(sympy.Symbol) - {s for i in idx for s in i.atoms(sympy.Symbol)}, key=str)
            target = idx[0] if idx else (syms[0] if syms else None)
            if target is None:
                continue
            try:
                sol = sympy.solve(sympy.Eq(expr, 0), target)
            except Exception:
                continue
            if not sol:
                continue
            new_args = [np.copy(a) if isinstance(a, np.ndarray) else a for a in args]
            subs = {}
            names = _arg_names(fn, new_args)
            for other in idx[1:] + [i for i in [target] if False]:
                subs[other] = _read(names, new_args, other)
            for s_ in syms:
                subs[s_] = _read(names, new_args, s_)
            try:
                val = float(sympy.N(sol[0].subs(subs)))
            except Exception:
                continue
            if not np.isfinite(val):
                continue
            _write(names, new_args, target, val)
            try:
                outcome = fn(*[np.copy(a) if isinstance(a, np.ndarray) else a
                               for a in new_args])
                outcome = np.asarray(outcome, dtype=float).tolist()
            except Exception as e:
                outcome = f"raises {type(e).__name__}: {e}"
            cases.append((key, new_args, outcome))
    return cases


def _arg_names(fn, args):
    import inspect

    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = [f"a{i}" for i in range(len(args))]
    return dict(zip(params, args))


def _base_name(term):
    if isinstance(term, sympy.Indexed):
        return str(term.base.label), tuple(int(i) for i in term.indices)
    return str(term), None


def _read(names, args, term):
    name, pos = _base_name(term)
    a = names.get(name)
    if a is None:
        return sympy.nan
    if pos is None:
        return float(a)
    return float(np.asarray(a)[pos])


def _write(names, args, term, val):
    name, pos = _base_name(term)
    a = names.get(name)
    if a is None:
        return
    if pos is None:
        for i, x in enumerate(args):
            if x is a:
                args[i] = val
        return
    np.asarray(a)[pos] = val


def verify(fn, *args, rtol=1e-9):
    """Inside your own @given test: trace fn and compare against the
    plain run. Refusal fails with the one-sentence reason; mismatch
    fails with both values."""
    ref = fn(*[np.copy(a) if isinstance(a, np.ndarray) else a for a in args])
    out = to_sympy(fn, *[np.copy(a) if isinstance(a, np.ndarray) else a
                         for a in args])
    got = out.value if isinstance(out, Pair) else out
    if isinstance(got, np.ndarray) and got.dtype == object:
        got = np.asarray(Pair._value_of(got), dtype=float)
    assert np.allclose(
        np.asarray(got, dtype=float), np.asarray(ref, dtype=float),
        rtol=rtol, equal_nan=True,
    ), f"certificate value {got} != library value {ref}\nformula: {out.formula}"
    return out
