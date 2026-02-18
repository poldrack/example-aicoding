"""Tests for LinearRegressionStats — extends sklearn LinearRegression with t-stats and p-values."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.base import clone

from aicoding.linear_regression.solution import LinearRegressionStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Simple 2-feature regression dataset with known structure."""
    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 2)
    # y = 3*x0 + 0*x1 + noise  →  x0 significant, x1 not
    y = 3.0 * X[:, 0] + 0.0 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


@pytest.fixture
def perfect_fit_data():
    """Data with perfect linear relationship (no noise)."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0
    return X, y


@pytest.fixture
def single_feature_data():
    """Single-feature regression dataset."""
    rng = np.random.RandomState(0)
    n = 50
    X = rng.randn(n, 1)
    y = 5.0 * X[:, 0] + rng.randn(n) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# Class structure tests
# ---------------------------------------------------------------------------

class TestClassStructure:
    def test_is_subclass_of_linear_regression(self):
        assert issubclass(LinearRegressionStats, LinearRegression)

    def test_instance_of_linear_regression(self):
        model = LinearRegressionStats()
        assert isinstance(model, LinearRegression)

    def test_default_construction(self):
        model = LinearRegressionStats()
        assert model is not None

    def test_sklearn_clone(self):
        """Must be clonable by sklearn (proper __init__ signature)."""
        model = LinearRegressionStats()
        cloned = clone(model)
        assert isinstance(cloned, LinearRegressionStats)

    def test_accepts_fit_intercept_param(self):
        model = LinearRegressionStats(fit_intercept=False)
        assert model.fit_intercept is False

    def test_get_params(self):
        model = LinearRegressionStats()
        params = model.get_params()
        assert "fit_intercept" in params


# ---------------------------------------------------------------------------
# Fit and predict tests
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats()
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_coef_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_intercept_is_scalar(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        assert np.isscalar(model.intercept_) or model.intercept_.ndim == 0

    def test_predictions_match_sklearn(self, simple_data):
        """Predictions should be identical to base LinearRegression."""
        X, y = simple_data
        stats_model = LinearRegressionStats().fit(X, y)
        base_model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(
            stats_model.predict(X), base_model.predict(X), atol=1e-10
        )


# ---------------------------------------------------------------------------
# t-statistic and p-value tests
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_tvalues_exist_after_fit(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        assert hasattr(model, "t_") or hasattr(model, "tvalues_")

    def test_pvalues_exist_after_fit(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        assert hasattr(model, "p_") or hasattr(model, "pvalues_")

    def _get_tvalues(self, model):
        return getattr(model, "t_", getattr(model, "tvalues_", None))

    def _get_pvalues(self, model):
        return getattr(model, "p_", getattr(model, "pvalues_", None))

    def test_tvalues_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        t = self._get_tvalues(model)
        assert t.shape == (X.shape[1],)

    def test_pvalues_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        p = self._get_pvalues(model)
        assert p.shape == (X.shape[1],)

    def test_pvalues_between_0_and_1(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        p = self._get_pvalues(model)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_significant_regressor_low_pvalue(self, simple_data):
        """x0 has coef=3, should be significant (p < 0.01)."""
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        p = self._get_pvalues(model)
        assert p[0] < 0.01

    def test_null_regressor_high_pvalue(self, simple_data):
        """x1 has coef=0, should not be significant (p > 0.05)."""
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        p = self._get_pvalues(model)
        assert p[1] > 0.05

    def test_significant_regressor_large_tvalue(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        t = self._get_tvalues(model)
        assert abs(t[0]) > 2.0

    def test_single_feature(self, single_feature_data):
        X, y = single_feature_data
        model = LinearRegressionStats().fit(X, y)
        t = self._get_tvalues(model)
        p = self._get_pvalues(model)
        assert t.shape == (1,)
        assert p.shape == (1,)
        assert p[0] < 0.001  # strong signal

    def test_tvalues_are_finite(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        t = self._get_tvalues(model)
        assert np.all(np.isfinite(t))

    def test_pvalues_are_finite(self, simple_data):
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)
        p = self._get_pvalues(model)
        assert np.all(np.isfinite(p))


# ---------------------------------------------------------------------------
# Statistical correctness — compare with statsmodels OLS
# ---------------------------------------------------------------------------

class TestAgainstStatsmodels:
    def test_tvalues_match_statsmodels(self, simple_data):
        """t-values should match statsmodels OLS results."""
        import statsmodels.api as sm
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)

        X_sm = sm.add_constant(X)
        ols = sm.OLS(y, X_sm).fit()

        t_ours = getattr(model, "t_", getattr(model, "tvalues_", None))
        t_sm = ols.tvalues[1:]  # skip intercept
        np.testing.assert_allclose(np.abs(t_ours), np.abs(t_sm), rtol=0.05)

    def test_pvalues_match_statsmodels(self, simple_data):
        """p-values should match statsmodels OLS results."""
        import statsmodels.api as sm
        X, y = simple_data
        model = LinearRegressionStats().fit(X, y)

        X_sm = sm.add_constant(X)
        ols = sm.OLS(y, X_sm).fit()

        p_ours = getattr(model, "p_", getattr(model, "pvalues_", None))
        p_sm = ols.pvalues[1:]  # skip intercept
        np.testing.assert_allclose(p_ours, p_sm, rtol=0.05)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_intercept(self):
        """Model with fit_intercept=False should still produce stats."""
        rng = np.random.RandomState(7)
        X = rng.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + rng.randn(50) * 0.1
        model = LinearRegressionStats(fit_intercept=False).fit(X, y)
        p = getattr(model, "p_", getattr(model, "pvalues_", None))
        assert p.shape == (3,)
        assert np.all(p < 0.01)

    def test_many_features(self):
        """Works with more features (but still n > p)."""
        rng = np.random.RandomState(99)
        X = rng.randn(200, 20)
        coef = np.zeros(20)
        coef[:5] = np.arange(1, 6)
        y = X @ coef + rng.randn(200) * 0.5
        model = LinearRegressionStats().fit(X, y)
        p = getattr(model, "p_", getattr(model, "pvalues_", None))
        assert p.shape == (20,)
        # first 5 should be significant
        assert np.all(p[:5] < 0.05)
        # most of the rest should not be
        assert np.sum(p[5:] > 0.05) >= 10

    def test_collinear_features_still_works(self):
        """With near-collinear features, stats should still be finite."""
        rng = np.random.RandomState(1)
        X = rng.randn(100, 2)
        X[:, 1] = X[:, 0] + rng.randn(100) * 1e-5  # nearly collinear
        y = X[:, 0] + rng.randn(100) * 0.1
        model = LinearRegressionStats().fit(X, y)
        t = getattr(model, "t_", getattr(model, "tvalues_", None))
        # t-values may be large but should be finite
        assert np.all(np.isfinite(t))
