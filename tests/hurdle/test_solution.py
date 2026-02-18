"""Tests for HurdleRegression — a two-part hurdle model with sklearn-style interface."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from aicoding.hurdle.solution import HurdleRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_inflated_count_data():
    """Generate zero-inflated count data for testing."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 3)
    # Probability of being non-zero depends on X[:,0]
    p_nonzero = 1 / (1 + np.exp(-1.0 - 2.0 * X[:, 0]))
    is_nonzero = rng.binomial(1, p_nonzero)
    # Counts for non-zero part depend on X[:,1]
    lam = np.exp(0.5 + 0.8 * X[:, 1])
    counts = rng.poisson(lam)
    counts = np.maximum(counts, 1)  # truncate at 1 for non-zero part
    y = is_nonzero * counts
    return X, y.astype(float)


@pytest.fixture
def all_positive_data():
    """Data with no zeros at all."""
    rng = np.random.RandomState(7)
    n = 100
    X = rng.randn(n, 2)
    y = np.exp(0.5 + X[:, 0]) + rng.exponential(0.5, n)
    y = np.maximum(y, 0.1)  # ensure all positive
    return X, y


@pytest.fixture
def mostly_zero_data():
    """Data that is mostly zeros."""
    rng = np.random.RandomState(99)
    n = 200
    X = rng.randn(n, 2)
    y = np.zeros(n)
    # Only 10% non-zero
    nonzero_idx = rng.choice(n, size=20, replace=False)
    y[nonzero_idx] = rng.poisson(3, size=20).astype(float) + 1
    return X, y


# ---------------------------------------------------------------------------
# Class structure tests
# ---------------------------------------------------------------------------

class TestClassStructure:
    def test_can_instantiate(self):
        model = HurdleRegression()
        assert model is not None

    def test_sklearn_clone(self):
        model = HurdleRegression()
        cloned = clone(model)
        assert isinstance(cloned, HurdleRegression)

    def test_has_fit_method(self):
        assert hasattr(HurdleRegression, "fit")

    def test_has_predict_method(self):
        assert hasattr(HurdleRegression, "predict")

    def test_get_params(self):
        model = HurdleRegression()
        params = model.get_params()
        assert isinstance(params, dict)

    def test_set_params(self):
        model = HurdleRegression()
        params = model.get_params()
        model.set_params(**params)


# ---------------------------------------------------------------------------
# Fit and predict tests
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_returns_self(self, zero_inflated_count_data):
        X, y = zero_inflated_count_data
        model = HurdleRegression()
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, zero_inflated_count_data):
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predictions_non_negative(self, zero_inflated_count_data):
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert np.all(preds >= 0)

    def test_predictions_finite(self, zero_inflated_count_data):
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_predictions_reasonable_range(self, zero_inflated_count_data):
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.mean() > 0
        assert preds.mean() < y.max() * 2


# ---------------------------------------------------------------------------
# Two-part model behavior
# ---------------------------------------------------------------------------

class TestHurdleBehavior:
    def test_has_classification_component(self, zero_inflated_count_data):
        """The model should have a binary classification sub-model."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        assert hasattr(model, "clf_") or hasattr(model, "classifier_")

    def test_has_regression_component(self, zero_inflated_count_data):
        """The model should have a regression sub-model for positive values."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        assert hasattr(model, "reg_") or hasattr(model, "regressor_")

    def test_prediction_is_product_of_components(self, zero_inflated_count_data):
        """Hurdle prediction = P(y>0) * E[y|y>0]."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)

        clf = getattr(model, "clf_", getattr(model, "classifier_", None))
        reg = getattr(model, "reg_", getattr(model, "regressor_", None))

        p_nonzero = clf.predict_proba(X)[:, 1]
        expected_positive = reg.predict(X)
        combined = p_nonzero * expected_positive
        np.testing.assert_allclose(preds, combined, rtol=1e-10)

    def test_mostly_zero_data_low_predictions(self, mostly_zero_data):
        """When data is mostly zeros, predictions should be low."""
        X, y = mostly_zero_data
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.mean() < 1.0

    def test_all_positive_data_no_zeros_predicted(self, all_positive_data):
        """When data has no zeros, P(y>0) should be high for all."""
        X, y = all_positive_data
        model = HurdleRegression().fit(X, y)
        clf = getattr(model, "clf_", getattr(model, "classifier_", None))
        p_nonzero = clf.predict_proba(X)[:, 1]
        assert np.mean(p_nonzero > 0.5) > 0.8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_feature(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 1)
        y = np.where(X[:, 0] > 0, np.exp(X[:, 0]), 0.0)
        model = HurdleRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_new_data_prediction(self, zero_inflated_count_data):
        """Predict on unseen data."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        X_new = np.random.RandomState(5).randn(10, 3)
        preds = model.predict(X_new)
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_score_method_exists(self, zero_inflated_count_data):
        """Should have a score method (sklearn convention)."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        score = model.score(X, y)
        assert np.isfinite(score)

    def test_reproducibility(self, zero_inflated_count_data):
        """Same data should give same predictions."""
        X, y = zero_inflated_count_data
        model1 = HurdleRegression().fit(X, y)
        model2 = HurdleRegression().fit(X, y)
        np.testing.assert_allclose(model1.predict(X), model2.predict(X))


# ---------------------------------------------------------------------------
# Joint MLE and truncated Poisson tests
# ---------------------------------------------------------------------------

class TestJointMLE:
    def test_uses_joint_mle_not_separate_sklearn(self, zero_inflated_count_data):
        """Model should use joint MLE, not separately fitted sklearn models."""
        X, y = zero_inflated_count_data
        model = HurdleRegression().fit(X, y)
        clf = getattr(model, "clf_", getattr(model, "classifier_", None))
        reg = getattr(model, "reg_", getattr(model, "regressor_", None))
        assert not isinstance(clf, LogisticRegression)
        assert not isinstance(reg, PoissonRegressor)

    def test_count_model_uses_truncated_poisson(self):
        """Count component should use truncated-at-zero Poisson.

        For small mu, the truncated Poisson mean mu/(1-exp(-mu)) differs
        substantially from mu. The model's estimated rate should be close
        to the true generating mu, not to the observed mean of positives.
        """
        rng = np.random.RandomState(42)
        n = 2000
        X = np.zeros((n, 1))

        true_mu = 0.5
        p_nonzero = 0.5

        is_nonzero = rng.binomial(1, p_nonzero, size=n).astype(bool)
        y = np.zeros(n)
        # Generate positive values from truncated Poisson (reject zeros)
        samples = []
        while len(samples) < is_nonzero.sum():
            c = rng.poisson(true_mu)
            if c > 0:
                samples.append(c)
        y[is_nonzero] = np.array(samples[: is_nonzero.sum()], dtype=float)

        model = HurdleRegression().fit(X, y)

        # The count component's underlying rate (intercept) should recover
        # log(true_mu) ≈ -0.693, not log(mean_positive) ≈ 0.239
        reg = getattr(model, "reg_", getattr(model, "regressor_", None))
        estimated_intercept = reg.intercept_
        expected_truncated = np.log(true_mu)  # -0.693
        naive_estimate = np.log(np.mean(y[y > 0]))  # ~0.239
        assert abs(estimated_intercept - expected_truncated) < abs(
            estimated_intercept - naive_estimate
        )
