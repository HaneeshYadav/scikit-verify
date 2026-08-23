# skverify-hypothesis

The obvious uses of [Hypothesis](https://hypothesis.readthedocs.io)
with scikit-verify certificates.

```python
from skverify_hypothesis import explore, edge_cases, verify
```

* `explore(fn, like)`: one witness input per distinct certificate
  path. Hypothesis draws inputs, each trace is signed by its
  preconditions, every new signature is a new branch of your code
  with its own formula. `np.median` on 4 points comes back as 24
  paths, each with its ordering hypotheses.

* `edge_cases(fn, like)`: inputs sitting exactly ON precondition
  boundaries. Each guard is solved for one input element and a drawn
  input is projected onto the boundary: `a[0] <= a[2]` yields a tie,
  `Ne(sum(b) - 1, 0)` yields an input whose denominator is exactly
  zero. Returns the real function's behavior at each boundary.

* `verify(fn, *args)`: inside your own `@given` test -- trace, check
  the certificate against the plain run, fail loudly with both
  values and the formula.

```python
@given(arrays(np.float64, 4, elements=st.floats(-2, 2, allow_nan=False)))
def test_median(a):
    verify(np.median, a)
```

Requires `hypothesis` and `scikit-verify`.
