import numpy as np, warnings
warnings.filterwarnings("ignore")
import skverify
from skverify.pair import Pair

def mini_scale(x):
    Xr = np.rollaxis(x, 0)
    mean_ = np.nanmean(x, 0)
    Xr -= mean_
    std_ = np.nanstd(x, 0)
    Xr /= std_
    return x

d = np.array([-0.8, 0.9, 0.7, -0.6, 0.5])
out = skverify.to_sympy(mini_scale, d)
v = out.value if isinstance(out, Pair) else out
print("traced:", np.round(np.asarray(Pair._value_of(v), float), 4))
print("real:  ", np.round(mini_scale(d.copy()), 4))
