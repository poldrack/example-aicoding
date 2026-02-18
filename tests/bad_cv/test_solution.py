"""Tests for the bad_cv module.

This module demonstrates incorrect cross-validation where feature selection
is performed outside the CV loop (leaking information) compared to the
correct approach where feature selection is done inside each fold.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from aicoding.mk_classification_data.solution import make_classification_data
from aicoding.bad_cv.solution import bad_crossvalidation, good_crossvalidation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Generate the standard classification dataset used across tests."""
    X, y = make_classification_data(
        n_samples=200, n_features=1000, n_informative=10, random_state=42
    )
    return X, y


@pytest.fixture
def small_data():
    """A smaller dataset for faster tests that still has many noise features."""
    X, y = make_classification_data(
        n_samples=100, n_features=500, n_informative=5, random_state=7
    )
    return X, y


# ---------------------------------------------------------------------------
# Test return types and structure
# ---------------------------------------------------------------------------

class TestReturnTypes:
    """Both functions should return a dict with documented keys."""

    def test_bad_cv_returns_dict(self, classification_data):
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        assert isinstance(result, dict)

    def test_good_cv_returns_dict(self, classification_data):
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert isinstance(result, dict)

    def test_bad_cv_has_mean_accuracy(self, classification_data):
        """Result dict must contain 'mean_accuracy'."""
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        assert "mean_accuracy" in result

    def test_good_cv_has_mean_accuracy(self, classification_data):
        """Result dict must contain 'mean_accuracy'."""
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert "mean_accuracy" in result

    def test_bad_cv_has_fold_accuracies(self, classification_data):
        """Result dict must contain 'fold_accuracies' as an array-like."""
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        assert "fold_accuracies" in result
        assert hasattr(result["fold_accuracies"], "__len__")

    def test_good_cv_has_fold_accuracies(self, classification_data):
        """Result dict must contain 'fold_accuracies' as an array-like."""
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert "fold_accuracies" in result
        assert hasattr(result["fold_accuracies"], "__len__")


# ---------------------------------------------------------------------------
# Test accuracy ranges
# ---------------------------------------------------------------------------

class TestAccuracyRanges:
    """Accuracy values should be valid probabilities."""

    def test_bad_cv_mean_accuracy_in_valid_range(self, classification_data):
        """Mean accuracy must be between 0 and 1."""
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_good_cv_mean_accuracy_in_valid_range(self, classification_data):
        """Mean accuracy must be between 0 and 1."""
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_bad_cv_fold_accuracies_in_valid_range(self, classification_data):
        """Each fold accuracy must be between 0 and 1."""
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        for acc in result["fold_accuracies"]:
            assert 0.0 <= acc <= 1.0

    def test_good_cv_fold_accuracies_in_valid_range(self, classification_data):
        """Each fold accuracy must be between 0 and 1."""
        X, y = classification_data
        result = good_crossvalidation(X, y)
        for acc in result["fold_accuracies"]:
            assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Test number of folds
# ---------------------------------------------------------------------------

class TestNumberOfFolds:
    """The number of fold accuracies should match the k parameter."""

    def test_bad_cv_default_10_folds(self, classification_data):
        """Default k=10 should produce 10 fold accuracies."""
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        assert len(result["fold_accuracies"]) == 10

    def test_good_cv_default_10_folds(self, classification_data):
        """Default k=10 should produce 10 fold accuracies."""
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert len(result["fold_accuracies"]) == 10

    def test_bad_cv_custom_k(self, small_data):
        """Custom k=5 should produce 5 fold accuracies."""
        X, y = small_data
        result = bad_crossvalidation(X, y, k=5)
        assert len(result["fold_accuracies"]) == 5

    def test_good_cv_custom_k(self, small_data):
        """Custom k=5 should produce 5 fold accuracies."""
        X, y = small_data
        result = good_crossvalidation(X, y, k=5)
        assert len(result["fold_accuracies"]) == 5


# ---------------------------------------------------------------------------
# Test mean matches fold accuracies
# ---------------------------------------------------------------------------

class TestMeanConsistency:
    """The mean_accuracy should equal the mean of fold_accuracies."""

    def test_bad_cv_mean_consistent(self, classification_data):
        X, y = classification_data
        result = bad_crossvalidation(X, y)
        expected_mean = np.mean(result["fold_accuracies"])
        np.testing.assert_almost_equal(result["mean_accuracy"], expected_mean, decimal=10)

    def test_good_cv_mean_consistent(self, classification_data):
        X, y = classification_data
        result = good_crossvalidation(X, y)
        expected_mean = np.mean(result["fold_accuracies"])
        np.testing.assert_almost_equal(result["mean_accuracy"], expected_mean, decimal=10)


# ---------------------------------------------------------------------------
# Core test: bad CV should show inflated accuracy vs good CV
# ---------------------------------------------------------------------------

class TestInflatedAccuracy:
    """The central claim: bad CV (feature selection outside loop) inflates accuracy."""

    def test_bad_cv_higher_than_good_cv(self, classification_data):
        """Bad CV should report higher accuracy than good CV.

        When feature selection is done on the full dataset before CV,
        noise features that are spuriously correlated with y leak information
        into the model, inflating the accuracy estimate.
        """
        X, y = classification_data
        bad_result = bad_crossvalidation(X, y)
        good_result = good_crossvalidation(X, y)
        assert bad_result["mean_accuracy"] > good_result["mean_accuracy"], (
            f"Expected bad CV ({bad_result['mean_accuracy']:.3f}) to be higher "
            f"than good CV ({good_result['mean_accuracy']:.3f}) due to data leakage"
        )

    def test_bad_cv_substantially_higher(self, classification_data):
        """The inflation should be substantial (at least 5 percentage points).

        With 200 samples, 1000 features, and only 10 informative, the
        leakage from feature selection outside CV should produce a
        noticeable accuracy boost.
        """
        X, y = classification_data
        bad_result = bad_crossvalidation(X, y)
        good_result = good_crossvalidation(X, y)
        diff = bad_result["mean_accuracy"] - good_result["mean_accuracy"]
        assert diff >= 0.05, (
            f"Expected at least 5pp inflation, got {diff:.3f} "
            f"(bad={bad_result['mean_accuracy']:.3f}, good={good_result['mean_accuracy']:.3f})"
        )


# ---------------------------------------------------------------------------
# Test good CV is near chance for mostly-noise data
# ---------------------------------------------------------------------------

class TestGoodCVRealistic:
    """Good CV should give honest (lower) accuracy on high-noise data."""

    def test_good_cv_not_near_perfect(self, classification_data):
        """Good CV should not achieve near-perfect accuracy on this dataset.

        With 200 samples and 1000 features (only 10 informative), honest
        CV should not exceed ~0.85 accuracy.
        """
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert result["mean_accuracy"] < 0.85, (
            f"Good CV accuracy {result['mean_accuracy']:.3f} suspiciously high"
        )

    def test_good_cv_above_chance(self, classification_data):
        """Good CV should still beat random chance (0.5) since there are
        informative features.
        """
        X, y = classification_data
        result = good_crossvalidation(X, y)
        assert result["mean_accuracy"] > 0.50, (
            f"Good CV accuracy {result['mean_accuracy']:.3f} at or below chance"
        )


# ---------------------------------------------------------------------------
# Test reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Results should be deterministic when random_state is controlled."""

    def test_bad_cv_reproducible(self):
        """Running bad_crossvalidation twice with same data should give same result."""
        X, y = make_classification_data(random_state=42)
        r1 = bad_crossvalidation(X, y)
        r2 = bad_crossvalidation(X, y)
        np.testing.assert_array_almost_equal(
            r1["fold_accuracies"], r2["fold_accuracies"]
        )

    def test_good_cv_reproducible(self):
        """Running good_crossvalidation twice with same data should give same result."""
        X, y = make_classification_data(random_state=42)
        r1 = good_crossvalidation(X, y)
        r2 = good_crossvalidation(X, y)
        np.testing.assert_array_almost_equal(
            r1["fold_accuracies"], r2["fold_accuracies"]
        )


# ---------------------------------------------------------------------------
# Test with different data shapes
# ---------------------------------------------------------------------------

class TestDifferentShapes:
    """Functions should work with various dataset sizes."""

    def test_bad_cv_with_small_data(self, small_data):
        """bad_crossvalidation should work with smaller datasets."""
        X, y = small_data
        result = bad_crossvalidation(X, y, k=5)
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert len(result["fold_accuracies"]) == 5

    def test_good_cv_with_small_data(self, small_data):
        """good_crossvalidation should work with smaller datasets."""
        X, y = small_data
        result = good_crossvalidation(X, y, k=5)
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert len(result["fold_accuracies"]) == 5

    def test_works_with_fewer_features_than_samples(self):
        """Should work even when n_features < n_samples (low-dimensional)."""
        X, y = make_classification_data(
            n_samples=100, n_features=20, n_informative=5, random_state=99
        )
        bad_result = bad_crossvalidation(X, y, k=5)
        good_result = good_crossvalidation(X, y, k=5)
        assert 0.0 <= bad_result["mean_accuracy"] <= 1.0
        assert 0.0 <= good_result["mean_accuracy"] <= 1.0
