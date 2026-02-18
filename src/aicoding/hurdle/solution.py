"""HurdleRegression — a two-part hurdle model with sklearn-style interface.

The hurdle model handles zero-inflated count data in two parts:
1. A binary (logistic) model for P(y > 0 | X).
2. A truncated-at-zero Poisson model for f(y | y > 0, X).

Parameters are estimated via joint maximum likelihood following the
formulation in Zeileis, Kleiber & Jackman (2008), "Regression Models for
Count Data in R" (pscl countreg vignette).

The final prediction is: P(y > 0 | X) * E[y | y > 0, X].
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from sklearn.base import BaseEstimator, RegressorMixin


class _BinaryComponent:
    """Fitted binary classifier exposing predict_proba for the hurdle part."""

    def __init__(self, coef, intercept):
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.atleast_1d(intercept)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_.ravel() + self.intercept_[0]
        p_pos = expit(z)
        return np.column_stack([1 - p_pos, p_pos])


class _TruncatedPoissonComponent:
    """Fitted truncated-at-zero Poisson exposing predict for the count part."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef)
        self.intercept_ = float(intercept)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        eta = X @ self.coef_ + self.intercept_
        mu = np.exp(np.clip(eta, -20, 20))
        # E[Y | Y > 0] for zero-truncated Poisson = mu / (1 - exp(-mu))
        return mu / np.maximum(1 - np.exp(-mu), 1e-15)


def _hurdle_nll(params, X, y, n_features):
    """Negative joint log-likelihood for the hurdle model.

    params layout:
        [0 .. n_features]       = gamma  (binary intercept + coefficients)
        [n_features+1 .. end]   = beta   (count intercept + coefficients)
    """
    d = n_features + 1
    gamma = params[:d]
    beta = params[d:]

    X_aug = np.column_stack([np.ones(X.shape[0]), X])

    # Binary part: P(y > 0) = sigmoid(X_aug @ gamma)
    p_pos = expit(X_aug @ gamma)

    # Count part: mu = exp(X_aug @ beta)
    eta = np.clip(X_aug @ beta, -20, 20)
    mu = np.exp(eta)

    zero_mask = y == 0
    pos_mask = ~zero_mask

    # Zeros: log P(y=0) = log(1 - p_pos)
    ll = np.sum(np.log(np.maximum(1 - p_pos[zero_mask], 1e-15)))

    # Positives: log P(y>0) + log Poisson(y;mu) - log(1 - exp(-mu))
    if np.any(pos_mask):
        y_pos = y[pos_mask]
        mu_pos = mu[pos_mask]

        ll += np.sum(np.log(np.maximum(p_pos[pos_mask], 1e-15)))
        ll += np.sum(
            -mu_pos
            + y_pos * np.log(np.maximum(mu_pos, 1e-15))
            - gammaln(y_pos + 1)
        )
        ll -= np.sum(np.log(np.maximum(1 - np.exp(-mu_pos), 1e-15)))

    return -ll


def _truncated_poisson_nll(beta, X, y):
    """Negative log-likelihood for truncated-at-zero Poisson only."""
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    eta = np.clip(X_aug @ beta, -20, 20)
    mu = np.exp(eta)
    ll = np.sum(
        -mu + y * np.log(np.maximum(mu, 1e-15)) - gammaln(y + 1)
    )
    ll -= np.sum(np.log(np.maximum(1 - np.exp(-mu), 1e-15)))
    return -ll


class HurdleRegression(BaseEstimator, RegressorMixin):
    """Hurdle regression model estimated via joint maximum likelihood.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum iterations for the L-BFGS-B optimizer.
    tol : float, default=1e-8
        Convergence tolerance.
    """

    def __init__(self, max_iter=1000, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Fit the hurdle model via joint MLE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Non-negative response variable.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        y_binary = (y > 0).astype(int)
        n_classes = len(np.unique(y_binary))

        if n_classes < 2:
            # Degenerate: all zeros or all positive
            gamma, beta = self._fit_degenerate(X, y, d)
        else:
            gamma, beta = self._fit_joint(X, y, d)

        self.clf_ = _BinaryComponent(gamma[1:], gamma[0])
        self.reg_ = _TruncatedPoissonComponent(beta[1:], beta[0])
        return self

    def _fit_degenerate(self, X, y, d):
        """Handle degenerate case where all values are same class."""
        all_positive = y[0] > 0
        if all_positive:
            # P(y>0) ≈ 1: set large positive intercept
            gamma = np.zeros(d + 1)
            gamma[0] = 10.0
            # Optimize only the count part on positive values
            beta0 = np.zeros(d + 1)
            beta0[0] = np.log(max(np.mean(y), 1e-5))
            result = minimize(
                _truncated_poisson_nll,
                beta0,
                args=(X, y),
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            beta = result.x
        else:
            # All zeros
            gamma = np.zeros(d + 1)
            gamma[0] = -10.0
            beta = np.zeros(d + 1)
        return gamma, beta

    def _fit_joint(self, X, y, d):
        """Joint MLE for the standard case with both zeros and positives."""
        # Initialization
        gamma0 = np.zeros(d + 1)
        beta0 = np.zeros(d + 1)

        p_pos_emp = np.mean(y > 0)
        gamma0[0] = np.log(p_pos_emp / max(1 - p_pos_emp, 1e-5))

        y_pos = y[y > 0]
        if len(y_pos) > 0:
            beta0[0] = np.log(max(np.mean(y_pos), 1e-5))

        params0 = np.concatenate([gamma0, beta0])

        result = minimize(
            _hurdle_nll,
            params0,
            args=(X, y, d),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        gamma = result.x[: d + 1]
        beta = result.x[d + 1 :]
        return gamma, beta

    def predict(self, X):
        """Predict expected values: P(y > 0 | X) * E[y | y > 0, X].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        p_nonzero = self.clf_.predict_proba(X)[:, 1]
        expected_positive = self.reg_.predict(X)
        return p_nonzero * expected_positive


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 3)

    # Generate zero-inflated Poisson data
    p_nonzero = 1 / (1 + np.exp(-1.0 - 2.0 * X[:, 0]))
    is_nonzero = rng.binomial(1, p_nonzero)
    lam = np.exp(0.5 + 0.8 * X[:, 1])
    counts = rng.poisson(lam)
    counts = np.maximum(counts, 1)
    y = (is_nonzero * counts).astype(float)

    model = HurdleRegression().fit(X, y)
    preds = model.predict(X)

    print("Hurdle Regression Model (Joint MLE)")
    print("=" * 50)
    print(f"N samples:       {n}")
    print(f"N zeros:         {np.sum(y == 0)} ({np.mean(y == 0) * 100:.1f}%)")
    print(f"N positive:      {np.sum(y > 0)} ({np.mean(y > 0) * 100:.1f}%)")
    print(f"Mean y:          {y.mean():.3f}")
    print(f"Mean prediction: {preds.mean():.3f}")
    print(f"R² score:        {model.score(X, y):.4f}")
    print()
    print("Classifier (zero vs positive) coefficients:")
    print(f"  {model.clf_.coef_[0]}")
    print("Regressor (truncated Poisson) coefficients:")
    print(f"  {model.reg_.coef_}")
