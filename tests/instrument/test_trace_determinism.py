"""A testing tool must give the same verdict every run.

Three nets: trace-order independence across fresh processes (the twin
caches are process-global, so order effects only show between
processes), repeated-trace stability inside one process, and session
isolation under the repeated to_sympy calls check_formula will make.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import sympy

from skverify import to_sympy

PY = sys.executable

MENU = textwrap.dedent(
    """
    import warnings, sys, json
    warnings.filterwarnings("ignore")
    sys.setrecursionlimit(20000)
    import numpy as np
    from skverify import to_sympy

    def f_std(v): return (v - v.mean()) / v.std()
    def f_soft(v):
        e = np.exp(v - v.max())
        return e / e.sum()
    def f_median(v): return np.median(v)
    def f_gram(a): return a.T @ a
    def f_horner(v):
        acc = 0.0
        for k in range(v.shape[0]):
            acc = acc * 2.0 + v[k]
        return acc

    FNS = {"std": f_std, "soft": f_soft, "median": f_median,
           "gram": f_gram, "horner": f_horner}
    v = np.array([0.7, -1.2, 2.5, 0.3, -0.4])
    A = np.arange(6.0).reshape(2, 3) + 1
    ARGS = {"gram": (A,)}

    order = sys.argv[1].split(",")
    certs = {}
    for name in order:
        out = to_sympy(FNS[name], *ARGS.get(name, (v.copy(),)))
        f = out.formula
        try:
            import sympy
            body = (" ; ".join(str(e) for e in f)
                    if isinstance(f, sympy.NDimArray) else str(f))
        except Exception:
            body = str(f)
        certs[name] = body + " ## " + str(out.preconditions)
    print(json.dumps(certs))
    """
)


def _run(order):
    r = subprocess.run(
        [PY, "-c", MENU, order], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class TestTraceOrderIndependence:
    def test_two_orders_identical(self):
        a = _run("std,soft,median,gram,horner")
        b = _run("horner,gram,median,soft,std")
        assert a == b

    def test_warm_cache_matches_cold(self):
        # tracing gram twice: second (warm-twin) trace must equal first
        a = _run("gram,gram")
        assert a["gram"] == _run("gram")["gram"]


class TestRepeatedTraceStability:
    def test_same_function_thrice_interleaved(self):
        def f(v):
            return (v - v.mean()) / v.std()

        def g(v):
            return np.sort(v)[1] * 2.0

        v = np.array([0.7, -1.2, 2.5, 0.3, -0.4])
        first = str(to_sympy(f, v.copy()).formula)
        to_sympy(g, v.copy())
        second = str(to_sympy(f, v.copy()).formula)
        to_sympy(g, v.copy())
        third = str(to_sympy(f, v.copy()).formula)
        assert first == second == third

    def test_check_formula_style_loop(self):
        # the @specifies pattern: many traces in a tight loop, sessions
        # must not leak guards or definitions across calls
        def f(v):
            if v.sum() > 0:
                return v * 2.0
            return v

        v = np.array([1.0, 2.0, 3.0])
        results = [to_sympy(f, v.copy()) for _ in range(5)]
        pres = {str(r.preconditions) for r in results}
        forms = {str(r.formula) for r in results}
        assert len(pres) == 1, pres
        assert len(forms) == 1, forms

    def test_guards_do_not_accumulate(self):
        def guarded(v):
            if v[0] > 0:
                return v.sum()
            return 0.0

        v = np.array([1.0, 2.0])
        n_guards = []
        for _ in range(3):
            out = to_sympy(guarded, v.copy())
            pre = out.preconditions
            n = len(pre.args) if isinstance(pre, sympy.And) else 1
            n_guards.append(n)
        assert n_guards[0] == n_guards[1] == n_guards[2], n_guards
