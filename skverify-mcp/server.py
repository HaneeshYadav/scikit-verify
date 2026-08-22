"""skverify MCP server: to_sympy over MCP, nothing else.

One tool traces a Python+NumPy function and returns its certificate:
formula, preconditions, definitions, checked-atom records, value.
Derivation is a separate tool (it can be large; the certificate stays
small). Refusals come back as one sentence in the "refused" field,
never as a stack trace. Nothing is truncated silently: any elision
announces itself.
"""

import linecache

import numpy as np
import sympy
from mcp.server.fastmcp import FastMCP

from skverify import Pair, to_sympy

mcp = FastMCP("skverify")

_LAST = {}  # session state: the most recent trace, for derivation()


def _load(source, function):
    fname = f"<mcp {function}>"
    linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
    ns = {"np": np, "numpy": np}
    exec(compile(source, fname, "exec"), ns)
    fn = ns.get(function)
    if fn is None:
        raise ValueError(f"source defines no function named {function!r}")
    return fn


def _convert(a):
    if isinstance(a, list):
        return np.asarray(a, dtype=float)
    if isinstance(a, (int, float)):
        return float(a)
    raise ValueError(f"arguments must be numbers or (nested) lists, got {type(a).__name__}")


def _cap(text, limit=20000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... elided {len(text) - limit} characters; the full text exceeds the response cap]"


@mcp.tool()
def trace(source: str, function: str, args: list) -> dict:
    """Translate a Python+NumPy function to symbolic mathematics.

    source: Python source defining the function (imports numpy as np
    allowed; the namespace already has np). function: its name. args:
    example inputs, numbers or nested lists (traced as named symbols
    and indexed arrays). Returns the certificate: formula (sympy text
    and latex), preconditions (hypotheses from branches the data took),
    definitions (folded loop lemmas), checked (compiled-call records
    with contract verdicts), value (the concrete result). If the
    function cannot be translated exactly, returns {"refused": <one
    sentence>} -- the code itself is fine, only the translation
    refused.
    """
    try:
        fn = _load(source, function)
        conv = [_convert(a) for a in args]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        out = to_sympy(fn, *conv)
    except NotImplementedError as e:
        return {"refused": str(e)}
    except Exception as e:
        return {"error": f"your code raised {type(e).__name__}: {e}"}
    _LAST["out"] = out
    formula = getattr(out, "formula", None)
    pre = getattr(out, "preconditions", sympy.true)
    pre_list = list(pre.args) if isinstance(pre, sympy.And) else (
        [] if pre is sympy.true else [pre]
    )
    value = out.value if isinstance(out, Pair) else out
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = np.asarray(Pair._value_of(value), dtype=float)
    return {
        "formula": _cap(str(formula)),
        "latex": _cap(sympy.latex(formula) if isinstance(formula, sympy.Basic) else str(formula)),
        "preconditions": [_cap(str(g), 2000) for g in pre_list],
        "definitions": {
            str(k): _cap(str(v), 4000)
            for k, v in (getattr(out, "definitions", {}) or {}).items()
        },
        "checked": [
            {"atom": str(r[0]), "verdicts": [list(map(str, v)) for v in r[1]],
             "definition": _cap(str(r[-1][1]), 2000)}
            for r in getattr(out, "unchecked", ())
            if isinstance(r, tuple) and len(r) >= 2
        ],
        "value": np.asarray(value, dtype=float).tolist()
        if value is not None else None,
    }


@mcp.tool()
def derivation(max_chars: int = 40000) -> dict:
    """Step-by-step derivation of the MOST RECENT trace: how each
    intermediate was computed from its parents, loops folded into
    rules. Large by nature; capped at max_chars with the elision
    announced."""
    out = _LAST.get("out")
    if out is None:
        return {"error": "no trace yet: call trace first"}
    try:
        text = out.derivation()
    except Exception as e:
        return {"error": f"derivation unavailable: {type(e).__name__}: {e}"}
    return {"derivation": _cap(str(text), max_chars)}


if __name__ == "__main__":
    mcp.run()
