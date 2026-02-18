"""LinearRegressionStats — sklearn LinearRegression extended with t-statistics and p-values."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


class LinearRegressionStats(LinearRegression):
    """Linear regression that computes t-statistics and p-values for each coefficient.

    Extends :class:`sklearn.linear_model.LinearRegression` so that after
    calling :meth:`fit`, the attributes ``t_`` and ``p_`` are available,
    containing the t-statistic and two-sided p-value for every regressor.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model and compute t-statistics and p-values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        self
        """
        super().fit(X, y, sample_weight=sample_weight)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        # Predicted values and residuals
        y_pred = self.predict(X)
        residuals = y - y_pred

        # Degrees of freedom
        if self.fit_intercept:
            dof = n - p - 1
        else:
            dof = n - p

        # Residual sum of squares → MSE
        mse = np.sum(residuals ** 2) / dof

        # Variance-covariance matrix of coefficients
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
            cov = mse * np.linalg.inv(X_design.T @ X_design)
            # Standard errors for regressors only (skip intercept)
            se = np.sqrt(np.diag(cov)[1:])
        else:
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))

        # t-statistics and p-values
        self.t_ = self.coef_ / se
        self.p_ = 2.0 * stats.t.sf(np.abs(self.t_), df=dof)

        return self


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = LinearRegressionStats().fit(X, y)

    print("Linear Regression with t-statistics and p-values")
    print("=" * 60)
    print(f"R² = {model.score(X, y):.4f}")
    print(f"Intercept = {model.intercept_:.4f}")
    print()
    print(f"{'Feature':>10}  {'Coef':>10}  {'t-stat':>10}  {'p-value':>12}")
    print("-" * 48)
    feature_names = load_diabetes().feature_names
    for name, coef, t, p in zip(feature_names, model.coef_, model.t_, model.p_):
        sig = "*" if p < 0.05 else ""
        print(f"{name:>10}  {coef:>10.4f}  {t:>10.4f}  {p:>12.6f} {sig}")
