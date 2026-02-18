"""Tests for PCA via power iteration implementation."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from aicoding.PCA_poweriteration.solution import (
    PowerIterationPCA,
    compare_with_sklearn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate reproducible 2-D data with clear principal directions."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 5
    # Create data with a dominant direction so power iteration converges easily
    X = rng.randn(n_samples, n_features)
    # Amplify first two directions
    X[:, 0] *= 10
    X[:, 1] *= 5
    return X


@pytest.fixture
def simple_2d_data():
    """Simple 2-D data for basic checks."""
    rng = np.random.RandomState(0)
    t = rng.randn(100)
    X = np.column_stack([2 * t + rng.randn(100) * 0.1,
                         t + rng.randn(100) * 0.1])
    return X


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestPowerIterationPCAInterface:
    """Verify the class exposes an sklearn-compatible interface."""

    def test_has_fit_method(self):
        pca = PowerIterationPCA()
        assert callable(getattr(pca, "fit", None))

    def test_has_transform_method(self):
        pca = PowerIterationPCA()
        assert callable(getattr(pca, "transform", None))

    def test_has_fit_transform_method(self):
        pca = PowerIterationPCA()
        assert callable(getattr(pca, "fit_transform", None))

    def test_default_n_components(self):
        pca = PowerIterationPCA()
        assert pca.n_components == 2

    def test_custom_n_components(self):
        pca = PowerIterationPCA(n_components=3)
        assert pca.n_components == 3

    def test_custom_max_iter(self):
        pca = PowerIterationPCA(max_iter=500)
        assert pca.max_iter == 500

    def test_custom_tol(self):
        pca = PowerIterationPCA(tol=1e-8)
        assert pca.tol == 1e-8


# ---------------------------------------------------------------------------
# Attribute tests (after fitting)
# ---------------------------------------------------------------------------

class TestPowerIterationPCAAttributes:
    """After calling fit, certain attributes must exist."""

    def test_components_attribute(self, sample_data):
        pca = PowerIterationPCA(n_components=2)
        pca.fit(sample_data)
        assert hasattr(pca, "components_")

    def test_components_shape(self, sample_data):
        n_components = 3
        pca = PowerIterationPCA(n_components=n_components)
        pca.fit(sample_data)
        assert pca.components_.shape == (n_components, sample_data.shape[1])

    def test_explained_variance_attribute(self, sample_data):
        pca = PowerIterationPCA(n_components=2)
        pca.fit(sample_data)
        assert hasattr(pca, "explained_variance_")

    def test_explained_variance_length(self, sample_data):
        n_components = 3
        pca = PowerIterationPCA(n_components=n_components)
        pca.fit(sample_data)
        assert len(pca.explained_variance_) == n_components

    def test_explained_variance_descending(self, sample_data):
        pca = PowerIterationPCA(n_components=3)
        pca.fit(sample_data)
        ev = pca.explained_variance_
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1], "Explained variances must be in descending order"

    def test_mean_attribute(self, sample_data):
        pca = PowerIterationPCA(n_components=2)
        pca.fit(sample_data)
        assert hasattr(pca, "mean_")
        np.testing.assert_allclose(pca.mean_, sample_data.mean(axis=0))

    def test_fit_returns_self(self, sample_data):
        pca = PowerIterationPCA(n_components=2)
        result = pca.fit(sample_data)
        assert result is pca


# ---------------------------------------------------------------------------
# Transform / output shape tests
# ---------------------------------------------------------------------------

class TestPowerIterationPCATransform:
    """Verify transform produces correct shapes and fit_transform consistency."""

    def test_transform_shape(self, sample_data):
        n_components = 3
        pca = PowerIterationPCA(n_components=n_components)
        pca.fit(sample_data)
        X_transformed = pca.transform(sample_data)
        assert X_transformed.shape == (sample_data.shape[0], n_components)

    def test_fit_transform_equals_fit_then_transform(self, sample_data):
        pca1 = PowerIterationPCA(n_components=2, max_iter=2000)
        X_ft = pca1.fit_transform(sample_data)

        pca2 = PowerIterationPCA(n_components=2, max_iter=2000)
        pca2.fit(sample_data)
        X_t = pca2.transform(sample_data)

        # Same random seed / algorithm => identical results
        np.testing.assert_allclose(X_ft, X_t, atol=1e-10)

    def test_transform_on_new_data(self, sample_data):
        """Transform on unseen data should still produce correct shape."""
        pca = PowerIterationPCA(n_components=2)
        pca.fit(sample_data)
        rng = np.random.RandomState(99)
        X_new = rng.randn(10, sample_data.shape[1])
        X_transformed = pca.transform(X_new)
        assert X_transformed.shape == (10, 2)


# ---------------------------------------------------------------------------
# Correctness tests — compare with sklearn PCA
# ---------------------------------------------------------------------------

class TestPowerIterationPCACorrectness:
    """The power-iteration PCA should agree with sklearn's PCA up to sign flips."""

    def _align_signs(self, A, B):
        """Flip signs of rows in B to best match rows of A."""
        B_aligned = B.copy()
        for i in range(A.shape[0]):
            if np.dot(A[i], B_aligned[i]) < 0:
                B_aligned[i] *= -1
        return B_aligned

    def test_components_close_to_sklearn(self, sample_data):
        n_components = 2
        pca_pi = PowerIterationPCA(n_components=n_components, max_iter=2000)
        pca_pi.fit(sample_data)

        pca_sk = SklearnPCA(n_components=n_components)
        pca_sk.fit(sample_data)

        aligned = self._align_signs(pca_sk.components_, pca_pi.components_)
        np.testing.assert_allclose(aligned, pca_sk.components_, atol=1e-3)

    def test_explained_variance_close_to_sklearn(self, sample_data):
        n_components = 3
        pca_pi = PowerIterationPCA(n_components=n_components, max_iter=2000)
        pca_pi.fit(sample_data)

        pca_sk = SklearnPCA(n_components=n_components)
        pca_sk.fit(sample_data)

        np.testing.assert_allclose(
            pca_pi.explained_variance_, pca_sk.explained_variance_, rtol=0.05
        )

    def test_transformed_data_close_to_sklearn(self, sample_data):
        n_components = 2
        pca_pi = PowerIterationPCA(n_components=n_components, max_iter=2000)
        X_pi = pca_pi.fit_transform(sample_data)

        pca_sk = SklearnPCA(n_components=n_components)
        X_sk = pca_sk.fit_transform(sample_data)

        # Fix sign ambiguity column-wise
        for j in range(n_components):
            if np.corrcoef(X_pi[:, j], X_sk[:, j])[0, 1] < 0:
                X_pi[:, j] *= -1

        np.testing.assert_allclose(X_pi, X_sk, atol=0.5)

    def test_orthogonal_components(self, sample_data):
        """Principal components must be mutually orthogonal."""
        pca = PowerIterationPCA(n_components=3, max_iter=2000)
        pca.fit(sample_data)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# Edge case: single component
# ---------------------------------------------------------------------------

class TestPowerIterationPCASingleComponent:
    """Edge case where n_components=1."""

    def test_single_component_shape(self, sample_data):
        pca = PowerIterationPCA(n_components=1, max_iter=2000)
        pca.fit(sample_data)
        assert pca.components_.shape == (1, sample_data.shape[1])
        assert len(pca.explained_variance_) == 1

    def test_single_component_transform(self, sample_data):
        pca = PowerIterationPCA(n_components=1, max_iter=2000)
        X_t = pca.fit_transform(sample_data)
        assert X_t.shape == (sample_data.shape[0], 1)

    def test_single_component_matches_sklearn(self, simple_2d_data):
        pca_pi = PowerIterationPCA(n_components=1, max_iter=2000)
        pca_pi.fit(simple_2d_data)

        pca_sk = SklearnPCA(n_components=1)
        pca_sk.fit(simple_2d_data)

        aligned = pca_pi.components_.copy()
        if np.dot(aligned[0], pca_sk.components_[0]) < 0:
            aligned[0] *= -1
        np.testing.assert_allclose(aligned, pca_sk.components_, atol=1e-3)


# ---------------------------------------------------------------------------
# compare_with_sklearn function
# ---------------------------------------------------------------------------

class TestCompareWithSklearn:
    """Test the standalone comparison function."""

    def test_returns_dict(self, sample_data):
        result = compare_with_sklearn(sample_data, n_components=2)
        assert isinstance(result, dict)

    def test_keys_present(self, sample_data):
        result = compare_with_sklearn(sample_data, n_components=2)
        expected_keys = {
            "power_iteration_components",
            "sklearn_components",
            "component_cosine_similarity",
            "explained_variance_power_iteration",
            "explained_variance_sklearn",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_cosine_similarities_high(self, sample_data):
        result = compare_with_sklearn(sample_data, n_components=2)
        sims = result["component_cosine_similarity"]
        for s in sims:
            assert abs(s) > 0.99, f"Cosine similarity too low: {s}"

    def test_compare_single_component(self, sample_data):
        result = compare_with_sklearn(sample_data, n_components=1)
        assert len(result["component_cosine_similarity"]) == 1
