import numpy as np, warnings
warnings.filterwarnings("ignore")
from skverify import to_sympy
from skverify.pair import Pair

rng = np.random.default_rng(5)
x = np.sort(rng.uniform(0, 4, 10))
y = np.sin(x) + 0.05 * rng.standard_normal(10)
v = rng.standard_normal(8)
M = rng.standard_normal((4, 4)); M = M @ M.T + 4 * np.eye(4)
b = rng.standard_normal(4)

import scipy.interpolate as si
import scipy.integrate as sint
import scipy.signal as ss
import scipy.linalg as sl
import scipy.stats as sst
import scipy.spatial.distance as sd
import scipy.special as sp
import scipy.fft as sfft

MENU = []
def add(name, fn, *args): MENU.append((name, fn, args))

add("interp1d_linear", lambda x, y: si.interp1d(x, y)(np.array([1.0, 2.0])), x, y)
add("make_interp_spline(k1)", lambda x, y: si.make_interp_spline(x, y, k=1).c, x, y)
add("CubicSpline.c0", lambda x, y: si.CubicSpline(x, y).c[0], x, y)
add("trapezoid", sint.trapezoid, v)
add("simpson", lambda v: sint.simpson(v), v)
add("cumulative_trapezoid", lambda v: sint.cumulative_trapezoid(v), v)
add("detrend", ss.detrend, v)
add("convolve_same", lambda v: ss.convolve(v, np.array([0.25, 0.5, 0.25]), mode="same"), v)
add("solve", lambda M, b: sl.solve(M, b), M, b)
add("lstsq", lambda M, b: sl.lstsq(M, b)[0], M, b)
add("cho_solve", lambda M, b: sl.cho_solve(sl.cho_factor(M), b), M, b)
add("expm_diag", lambda b: sl.expm(np.diag(b))[0, 0], b)
add("zscore", sst.zscore, v)
add("gmean", lambda v: sst.gmean(np.abs(v)), v)
add("hmean", lambda v: sst.hmean(np.abs(v) + 1), v)
add("rankdata", lambda v: sst.rankdata(v).astype(float) + 0*v, v)
add("iqr", sst.iqr, v)
add("skew", sst.skew, v)
add("kurtosis", sst.kurtosis, v)
add("sem", sst.sem, v)
add("pearsonr", lambda v: sst.pearsonr(v, v[::-1])[0], v)
add("moment2", lambda v: sst.moment(v, 2), v)
add("norm.pdf", lambda v: sst.norm.pdf(v), v)
add("norm.cdf", lambda v: sst.norm.cdf(v), v)
add("t.sf", lambda v: sst.t.sf(np.abs(v), 5), v)
add("euclidean", lambda v: sd.euclidean(v[:4], v[4:]), v)
add("cosine_dist", lambda v: sd.cosine(v[:4], v[4:]), v)
add("erf", lambda v: sp.erf(v), v)
add("gammaln", lambda v: sp.gammaln(np.abs(v) + 1), v)
add("expit", lambda v: sp.expit(v), v)
add("xlogy", lambda v: sp.xlogy(np.abs(v), np.abs(v) + 1), v)
add("softmax", lambda v: sp.softmax(v), v)
add("logsumexp", lambda v: sp.logsumexp(v), v)
add("fft_real", lambda v: np.real(sfft.fft(v)), v)

lift_ok, unver, refused, died = [], [], [], []
for name, fn, args in MENU:
    try:
        ref = fn(*args)
        r = to_sympy(fn, *args)
        got = r.value if isinstance(r, Pair) else r
        if isinstance(got, np.ndarray) and got.dtype == object:
            got = np.asarray([Pair._value_of(e) for e in np.ravel(got)], dtype=float).reshape(np.shape(got))
        try:
            ok = np.allclose(np.asarray(got, float), np.asarray(ref, float), rtol=1e-6, atol=1e-8, equal_nan=True)
        except Exception:
            ok = None
        (lift_ok if ok else unver).append(name)
    except NotImplementedError as e:
        refused.append((name, str(e)[:48]))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:48]}"))

print(f"TOTAL {len(MENU)} | LIFT+match {len(lift_ok)} | unverified {len(unver)} | refused {len(refused)} | died {len(died)}")
print("LIFT:", ", ".join(lift_ok))
print("UNVERIFIED:", unver)
print("REFUSED:");  [print("  ", a, "|", m) for a, m in refused]
print("DIED:");     [print("  ", a, "|", m) for a, m in died]
