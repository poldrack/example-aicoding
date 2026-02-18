"""Tests for the mk_classification_data module.

This module generates synthetic classification data suitable for
demonstrating incorrect cross-validation with feature selection
performed outside the CV loop (problem #9, bad_cv).
"""

import numpy as np
import pytest

from aicoding.mk_classification_data.solution import make_classification_data


class TestReturnTypes:
    """Verify that make_classification_data returns the correct types."""

    def test_returns_tuple(self):
        """Function should return a tuple of (X, y)."""
        result = make_classification_data()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_X_is_numpy_array(self):
        """Features matrix X should be a numpy ndarray."""
        X, _ = make_classification_data()
        assert isinstance(X, np.ndarray)

    def test_y_is_numpy_array(self):
        """Target vector y should be a numpy ndarray."""
        _, y = make_classification_data()
        assert isinstance(y, np.ndarray)


class TestDefaultShape:
    """Verify default dimensions of the generated dataset."""

    def test_default_n_samples(self):
        """Default dataset should have 200 samples."""
        X, y = make_classification_data()
        assert X.shape[0] == 200
        assert y.shape[0] == 200

    def test_default_n_features(self):
        """Default dataset should have many features (at least 500).

        A large number of features is needed so that feature selection
        outside the CV loop can introduce optimistic bias.
        """
        X, _ = make_classification_data()
        assert X.shape[1] >= 500

    def test_X_is_2d(self):
        """X should be a 2-dimensional array."""
        X, _ = make_classification_data()
        assert X.ndim == 2

    def test_y_is_1d(self):
        """y should be a 1-dimensional array."""
        _, y = make_classification_data()
        assert y.ndim == 1


class TestCustomShape:
    """Verify that n_samples and n_features can be customized."""

    def test_custom_n_samples(self):
        """User should be able to set the number of samples."""
        X, y = make_classification_data(n_samples=50)
        assert X.shape[0] == 50
        assert y.shape[0] == 50

    def test_custom_n_features(self):
        """User should be able to set the number of features."""
        X, _ = make_classification_data(n_features=100)
        assert X.shape[1] == 100

    def test_custom_both(self):
        """Both n_samples and n_features should be customizable together."""
        X, y = make_classification_data(n_samples=80, n_features=300)
        assert X.shape == (80, 300)
        assert y.shape == (80,)


class TestBinaryClassification:
    """Verify the target is binary classification."""

    def test_target_has_two_classes(self):
        """y should contain exactly two unique values."""
        _, y = make_classification_data()
        assert len(np.unique(y)) == 2

    def test_target_values_are_zero_and_one(self):
        """y should contain only 0 and 1."""
        _, y = make_classification_data()
        assert set(np.unique(y)) == {0, 1}

    def test_classes_are_roughly_balanced(self):
        """Both classes should be represented with rough balance."""
        _, y = make_classification_data(n_samples=200, random_state=42)
        counts = np.bincount(y)
        # Each class should have at least 30% of samples
        assert counts[0] >= 0.3 * len(y)
        assert counts[1] >= 0.3 * len(y)


class TestReproducibility:
    """Verify that the random_state parameter controls reproducibility."""

    def test_same_seed_gives_same_data(self):
        """Same random_state should produce identical datasets."""
        X1, y1 = make_classification_data(random_state=123)
        X2, y2 = make_classification_data(random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seed_gives_different_data(self):
        """Different random_state should produce different datasets."""
        X1, _ = make_classification_data(random_state=1)
        X2, _ = make_classification_data(random_state=2)
        assert not np.array_equal(X1, X2)


class TestInformativeFeatures:
    """Verify that the dataset has a mix of informative and noise features."""

    def test_n_informative_parameter(self):
        """Function should accept n_informative parameter."""
        X, y = make_classification_data(
            n_features=100, n_informative=5, random_state=42
        )
        assert X.shape[1] == 100

    def test_not_all_features_are_informative(self):
        """With many features and few informative, most should be noise.

        This is critical for the bad_cv demonstration: when feature
        selection is done on the full dataset, noise features can appear
        correlated with y by chance, inflating accuracy.
        """
        X, y = make_classification_data(
            n_samples=200, n_features=500, n_informative=10, random_state=42
        )
        # Compute absolute correlation of each feature with y
        correlations = np.array(
            [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
        )
        # Most features should have low correlation (noise)
        n_low_corr = np.sum(correlations < 0.2)
        assert n_low_corr > 0.8 * X.shape[1], (
            "Expected most features to be noise (low correlation with target)"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Function should work with a very small dataset."""
        X, y = make_classification_data(n_samples=10, n_features=5, n_informative=2)
        assert X.shape == (10, 5)
        assert y.shape == (10,)
        assert len(np.unique(y)) == 2

    def test_single_informative_feature(self):
        """Should work with just one informative feature."""
        X, y = make_classification_data(
            n_samples=50, n_features=20, n_informative=1, random_state=42
        )
        assert X.shape == (50, 20)
        assert len(np.unique(y)) == 2

    def test_features_are_finite(self):
        """All feature values should be finite (no NaN or inf)."""
        X, y = make_classification_data()
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))
