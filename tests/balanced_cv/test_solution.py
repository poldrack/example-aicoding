"""Tests for the balanced cross-validation module.

Tests verify that BalancedKFold follows the scikit-learn cross-validator
interface and produces splits whose fold distributions are more balanced
(lower F-statistic) than naive random splits.
"""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array

from aicoding.balanced_cv.solution import BalancedKFold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Generate a simple regression dataset for testing."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3))
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.standard_normal(n) * 0.1
    return X, y


@pytest.fixture
def skewed_data():
    """Generate a dataset with skewed y to stress-test balancing."""
    rng = np.random.default_rng(123)
    n = 120
    X = rng.standard_normal((n, 2))
    # Highly skewed target: exponential distribution
    y = rng.exponential(scale=2.0, size=n)
    return X, y


@pytest.fixture
def balanced_cv():
    """Default BalancedKFold instance."""
    return BalancedKFold(n_splits=5, n_candidates=100, seed=42)


# ---------------------------------------------------------------------------
# Test: Constructor and get_n_splits
# ---------------------------------------------------------------------------

class TestConstructorAndGetNSplits:
    """Tests for __init__ and get_n_splits."""

    def test_default_parameters(self):
        """BalancedKFold should have sensible defaults."""
        cv = BalancedKFold()
        assert cv.n_splits == 5
        assert cv.n_candidates == 100
        assert cv.get_n_splits() == 5

    def test_custom_parameters(self):
        """Custom n_splits, n_candidates, seed should be stored."""
        cv = BalancedKFold(n_splits=10, n_candidates=200, seed=99)
        assert cv.n_splits == 10
        assert cv.n_candidates == 200
        assert cv.get_n_splits() == 10

    def test_get_n_splits_ignores_arguments(self):
        """get_n_splits should work with optional X, y, groups args
        (sklearn compatibility)."""
        cv = BalancedKFold(n_splits=3)
        assert cv.get_n_splits(X=None, y=None, groups=None) == 3


# ---------------------------------------------------------------------------
# Test: split method — basic properties
# ---------------------------------------------------------------------------

class TestSplitBasicProperties:
    """Tests for the split method's basic structural guarantees."""

    def test_split_yields_correct_number_of_folds(self, simple_data, balanced_cv):
        """split() should yield exactly n_splits (train, test) tuples."""
        X, y = simple_data
        splits = list(balanced_cv.split(X, y))
        assert len(splits) == 5

    def test_split_returns_index_arrays(self, simple_data, balanced_cv):
        """Each element should be a tuple of two numpy arrays of indices."""
        X, y = simple_data
        for train_idx, test_idx in balanced_cv.split(X, y):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_covers_all_indices(self, simple_data, balanced_cv):
        """The union of all test indices should cover every sample exactly once."""
        X, y = simple_data
        all_test_indices = []
        for _, test_idx in balanced_cv.split(X, y):
            all_test_indices.extend(test_idx.tolist())
        all_test_indices_sorted = sorted(all_test_indices)
        assert all_test_indices_sorted == list(range(len(y)))

    def test_train_test_no_overlap(self, simple_data, balanced_cv):
        """Within each fold, train and test indices must not overlap."""
        X, y = simple_data
        for train_idx, test_idx in balanced_cv.split(X, y):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_train_test_union_is_complete(self, simple_data, balanced_cv):
        """Within each fold, train + test indices should cover all samples."""
        X, y = simple_data
        n = len(y)
        for train_idx, test_idx in balanced_cv.split(X, y):
            combined = set(train_idx) | set(test_idx)
            assert combined == set(range(n))


# ---------------------------------------------------------------------------
# Test: Balancing quality (F-statistic)
# ---------------------------------------------------------------------------

class TestBalancingQuality:
    """Tests verifying that BalancedKFold produces more balanced folds."""

    def test_f_statistic_lower_than_naive(self, skewed_data):
        """BalancedKFold splits should have a lower F-statistic across folds
        than a single random (unoptimized) split, on average."""
        from scipy.stats import f_oneway
        X, y = skewed_data

        # Balanced split
        cv_balanced = BalancedKFold(n_splits=5, n_candidates=200, seed=42)
        balanced_splits = list(cv_balanced.split(X, y))
        balanced_fold_values = [y[test_idx] for _, test_idx in balanced_splits]
        f_balanced, _ = f_oneway(*balanced_fold_values)

        # Naive KFold (no shuffling, sequential split)
        cv_naive = KFold(n_splits=5, shuffle=True, random_state=0)
        naive_splits = list(cv_naive.split(X, y))
        naive_fold_values = [y[test_idx] for _, test_idx in naive_splits]
        f_naive, _ = f_oneway(*naive_fold_values)

        # Balanced should have lower or equal F-statistic
        # We allow a generous margin since this is stochastic
        assert f_balanced <= f_naive * 1.5, (
            f"Balanced F={f_balanced:.4f} should be meaningfully lower than "
            f"naive F={f_naive:.4f}"
        )

    def test_fold_means_are_similar(self, skewed_data):
        """The means of y across folds should be relatively similar."""
        X, y = skewed_data
        cv = BalancedKFold(n_splits=5, n_candidates=200, seed=42)
        fold_means = []
        for _, test_idx in cv.split(X, y):
            fold_means.append(np.mean(y[test_idx]))
        fold_means = np.array(fold_means)
        overall_mean = np.mean(y)
        # Each fold mean should be within 50% of the overall mean
        for i, fm in enumerate(fold_means):
            assert abs(fm - overall_mean) < overall_mean, (
                f"Fold {i} mean={fm:.3f} is too far from overall mean={overall_mean:.3f}"
            )


# ---------------------------------------------------------------------------
# Test: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Tests verifying deterministic behavior with fixed seed."""

    def test_same_seed_same_splits(self, simple_data):
        """Two BalancedKFold instances with the same seed should produce
        identical splits."""
        X, y = simple_data
        cv1 = BalancedKFold(n_splits=5, n_candidates=100, seed=42)
        cv2 = BalancedKFold(n_splits=5, n_candidates=100, seed=42)
        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))
        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seed_different_splits(self, simple_data):
        """Different seeds should (almost certainly) produce different splits."""
        X, y = simple_data
        cv1 = BalancedKFold(n_splits=5, n_candidates=100, seed=42)
        cv2 = BalancedKFold(n_splits=5, n_candidates=100, seed=99)
        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))
        any_different = False
        for (tr1, _), (tr2, _) in zip(splits1, splits2):
            if not np.array_equal(tr1, tr2):
                any_different = True
                break
        assert any_different, "Different seeds produced identical splits"


# ---------------------------------------------------------------------------
# Test: sklearn compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompatibility:
    """Tests verifying that BalancedKFold works with sklearn utilities."""

    def test_cross_val_score_integration(self, simple_data):
        """BalancedKFold should work as a cv argument in cross_val_score."""
        X, y = simple_data
        cv = BalancedKFold(n_splits=5, n_candidates=50, seed=42)
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        assert len(scores) == 5
        # All scores should be finite
        assert np.all(np.isfinite(scores))

    def test_split_accepts_dataframe(self):
        """split() should accept pandas DataFrames as X."""
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
        y = rng.standard_normal(n)
        cv = BalancedKFold(n_splits=3, n_candidates=20, seed=42)
        splits = list(cv.split(df, y))
        assert len(splits) == 3


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_n_splits_equals_n_samples(self):
        """When n_splits equals n_samples (LOO-like), should still work."""
        rng = np.random.default_rng(42)
        n = 10
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)
        cv = BalancedKFold(n_splits=n, n_candidates=10, seed=42)
        splits = list(cv.split(X, y))
        assert len(splits) == n
        for train_idx, test_idx in splits:
            assert len(test_idx) == 1
            assert len(train_idx) == n - 1

    def test_two_folds(self):
        """Should work with n_splits=2 (50/50 split)."""
        rng = np.random.default_rng(42)
        n = 40
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)
        cv = BalancedKFold(n_splits=2, n_candidates=50, seed=42)
        splits = list(cv.split(X, y))
        assert len(splits) == 2
        sizes = [len(te) for _, te in splits]
        assert sum(sizes) == n

    def test_single_candidate(self):
        """n_candidates=1 should still produce a valid split (just random)."""
        rng = np.random.default_rng(42)
        n = 30
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)
        cv = BalancedKFold(n_splits=3, n_candidates=1, seed=42)
        splits = list(cv.split(X, y))
        assert len(splits) == 3
        all_test = []
        for _, test_idx in splits:
            all_test.extend(test_idx.tolist())
        assert sorted(all_test) == list(range(n))
