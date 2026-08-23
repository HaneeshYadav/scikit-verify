# skverify-hypothesis

Three helpers for testing numerical code with
[Hypothesis](https://hypothesis.readthedocs.io).

```python
from skverify_hypothesis import explore, edge_cases, verify
```

**`explore(fn, like)` -- find every branch of your function, with an
input that reaches it.**

```python
paths = explore(np.median, (np.zeros(4),))
# 24 paths: median takes a different route for each ordering of 4 numbers.
# You get one example input per route.
```

You wrote one test input; your function has 24 behaviors. This finds
all of them so your tests can cover them.

**`edge_cases(fn, like)` -- generate the inputs most likely to break
your function.**

```python
cases = edge_cases(my_metric, (np.zeros(4),))
# inputs with exact ties, denominators that are exactly zero,
# values exactly on every if-condition in your code
```

Bugs live on boundaries: the tie, the zero denominator, the equal
endpoints. This reads your code's actual conditions and builds inputs
that sit exactly on them, then shows you what your function does
there. (This is how we found a NaN bug in scipy.stats.)

**`verify(fn, *args)` -- one extra line that makes a Hypothesis test
check the math, not just "it didn't crash."**

```python
@given(arrays(np.float64, 4, elements=st.floats(-2, 2, allow_nan=False)))
def test_median(a):
    verify(np.median, a)   # fails loudly if the math is ever wrong
```

Requires `hypothesis` and `scikit-verify`.
