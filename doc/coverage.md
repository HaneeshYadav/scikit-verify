# Coverage

Functions we have run through `to_sympy`, and what happened.
**works** means the traced value matched the library exactly on
that call; **refuses** means a one-sentence refusal (see
[sharp bits](sharp-bits.md)). Regenerate with the batteries in
`batteries/`.

## NumPy / SciPy

**Works (28):** `make_interp_spline(k1)`, `CubicSpline.c0`, `trapezoid`, `simpson`, `cumulative_trapezoid`, `detrend`, `convolve_same`, `solve`, `lstsq`, `cho_solve`, `expm_diag`, `zscore`, `gmean`, `hmean`, `rankdata`, `skew`, `kurtosis`, `sem`, `moment2`, `euclidean`, `cosine_dist`, `erf`, `gammaln`, `expit`, `xlogy`, `softmax`, `logsumexp`, `fft_real`

**Refuses (4):** `interp1d_linear`, `iqr`, `pearsonr`, `t.sf`

## scikit-learn

**Works (57):** `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `median_absolute_error`, `max_error`, `mean_squared_log_error`, `root_mean_squared_log_error`, `mean_absolute_percentage_error`, `r2_score`, `explained_variance_score`, `mean_poisson_deviance`, `mean_gamma_deviance`, `mean_tweedie_deviance`, `mean_pinball_loss`, `d2_absolute_error_score`, `d2_pinball_score`, `d2_tweedie_score`, `accuracy_score`, `zero_one_loss`, `hamming_loss`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score_2`, `matthews_corrcoef`, `jaccard_score`, `balanced_accuracy`, `cohen_kappa`, `log_loss`, `roc_auc_score`, `average_precision`, `euclidean_distances`, `cosine_similarity`, `cosine_distances`, `linear_kernel`, `polynomial_kernel`, `rbf_kernel`, `paired_euclidean`, `paired_cosine`, `minmax_scale_fn`, `maxabs_scale_fn`, `robust_scale`, `normalize_l2`, `normalize_l1`, `normalize_max`, `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `PolynomialFeatures`, `KernelCenterer`, `LinearRegression`, `Ridge`, `BayesianRidge`, `LinearRegression.predict`, `Ridge.predict`

