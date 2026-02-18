# linear_regression (#1)

Extends `sklearn.linear_model.LinearRegression` with t-statistics and p-values for each coefficient.

## Approach

`LinearRegressionStats` subclasses `LinearRegression` and overrides `fit()` to compute, after the standard OLS fit:

1. Residual mean squared error (MSE) with appropriate degrees of freedom.
2. The variance-covariance matrix of coefficients via `MSE * (X'X)^{-1}`.
3. Standard errors from the diagonal of the covariance matrix.
4. t-statistics as `coef / SE` and two-sided p-values from the t-distribution.

Results are stored as `t_` and `p_` attributes.

## Key design decisions

- The `__init__` signature mirrors `LinearRegression` exactly so that `sklearn.base.clone()` and other sklearn utilities work correctly.
- When `fit_intercept=True`, the design matrix is augmented with a column of ones for the covariance calculation, but only regressor standard errors are kept (intercept statistics are excluded from `t_` and `p_`).
- Validated against `statsmodels.OLS` to confirm correctness.
