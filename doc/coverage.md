# Coverage

Every function we have run through `to_sympy`, and what happened.
**Works** means the traced value matched the library exactly on
that call. **Refuses** means a one-sentence refusal instead of a
result -- the rules behind refusals are in
[sharp bits](sharp-bits.md). Regenerate this page with the
batteries in `batteries/`.

## NumPy / SciPy

**Works (28):** `make_interp_spline(k1)`, `CubicSpline.c0`, `trapezoid`, `simpson`, `cumulative_trapezoid`, `detrend`, `convolve_same`, `solve`, `lstsq`, `cho_solve`, `expm_diag`, `zscore`, `gmean`, `hmean`, `rankdata`, `skew`, `kurtosis`, `sem`, `moment2`, `euclidean`, `cosine_dist`, `erf`, `gammaln`, `expit`, `xlogy`, `softmax`, `logsumexp`, `fft_real`

**Refuses (4):** `interp1d_linear`, `iqr`, `pearsonr`, `t.sf`

## scikit-learn

**Works (57):** `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `median_absolute_error`, `max_error`, `mean_squared_log_error`, `root_mean_squared_log_error`, `mean_absolute_percentage_error`, `r2_score`, `explained_variance_score`, `mean_poisson_deviance`, `mean_gamma_deviance`, `mean_tweedie_deviance`, `mean_pinball_loss`, `d2_absolute_error_score`, `d2_pinball_score`, `d2_tweedie_score`, `accuracy_score`, `zero_one_loss`, `hamming_loss`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score_2`, `matthews_corrcoef`, `jaccard_score`, `balanced_accuracy`, `cohen_kappa`, `log_loss`, `roc_auc_score`, `average_precision`, `euclidean_distances`, `cosine_similarity`, `cosine_distances`, `linear_kernel`, `polynomial_kernel`, `rbf_kernel`, `paired_euclidean`, `paired_cosine`, `minmax_scale_fn`, `maxabs_scale_fn`, `robust_scale`, `normalize_l2`, `normalize_l1`, `normalize_max`, `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `PolynomialFeatures`, `KernelCenterer`, `LinearRegression`, `Ridge`, `BayesianRidge`, `LinearRegression.predict`, `Ridge.predict`

## statsmodels

**Works (30):** `durbin_watson`, `robust_skewness`, `medcouple`, `mad`, `add_constant`, `pinv_extended`, `OLS.params`, `OLS.resid`, `OLS.fittedvalues`, `OLS.rsquared`, `OLS.rsquared_adj`, `OLS.bse`, `OLS.tvalues`, `OLS.aic`, `OLS.bic`, `OLS.llf`, `OLS.predict`, `OLS.mse_resid`, `OLS.ess`, `OLS.centered_tss`, `OLS.fvalue`, `WLS.params`, `GLS.params`, `OLS.cov_params`, `OLS.HC0_se`, `Huber.rho`, `Huber.psi`, `Tukey.rho`, `acf`, `zscore-ish`

**Refuses (14):** `jarque_bera`, `omni_normtest`, `robust_kurtosis`, `OLS.pvalues`, `GLM.gaussian.params`, `GLM.poisson.params`, `GLM.binomial.params`, `GLM.gaussian.mu`, `Poisson.params`, `acovf`, `pacf_yw`, `adfuller_stat`, `RLM.params`, `quantile_reg`

**Known walls (1):** `Logit.params`

