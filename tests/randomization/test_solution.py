"""Tests for the randomization (permutation test) module."""

import numpy as np
import pytest
from aicoding.randomization.solution import (
    compute_correlation,
    compute_empirical_pvalue,
    generate_correlated_data,
    permutation_test,
)


class TestGenerateCorrelatedData:
    """Test bivariate correlated data generation."""

    def test_returns_correct_shape(self):
        """Generated data should have shape (100, 2) by default."""
        data = generate_correlated_data(n=100, r=0.1, seed=42)
        assert data.shape == (100, 2)

    def test_custom_sample_size(self):
        """Generated data should respect custom sample size."""
        data = generate_correlated_data(n=50, r=0.1, seed=42)
        assert data.shape == (50, 2)

    def test_approximate_correlation(self):
        """Generated data should have approximate correlation of 0.1."""
        data = generate_correlated_data(n=100, r=0.1, seed=42)
        observed_r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        # With n=100, the sample correlation should be within ~0.2 of the target
        assert abs(observed_r - 0.1) < 0.25

    def test_reproducible_with_seed(self):
        """Same seed should produce identical data."""
        data1 = generate_correlated_data(n=100, r=0.1, seed=42)
        data2 = generate_correlated_data(n=100, r=0.1, seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        data1 = generate_correlated_data(n=100, r=0.1, seed=42)
        data2 = generate_correlated_data(n=100, r=0.1, seed=99)
        assert not np.array_equal(data1, data2)


class TestComputeCorrelation:
    """Test Pearson correlation computation."""

    def test_returns_two_values(self):
        """compute_correlation should return (r, p_value) tuple."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        result = compute_correlation(x, y)
        assert len(result) == 2

    def test_r_is_float_in_range(self):
        """Correlation coefficient should be a float between -1 and 1."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        r, _ = compute_correlation(x, y)
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0

    def test_p_value_between_0_and_1(self):
        """Parametric p-value should be between 0 and 1."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        _, p = compute_correlation(x, y)
        assert 0.0 <= p <= 1.0

    def test_perfect_positive_correlation(self):
        """Perfectly correlated data should have r close to 1."""
        x = np.arange(100, dtype=float)
        y = x * 2.0 + 3.0
        r, p = compute_correlation(x, y)
        assert abs(r - 1.0) < 1e-10
        assert p < 0.001

    def test_perfect_negative_correlation(self):
        """Perfectly anti-correlated data should have r close to -1."""
        x = np.arange(100, dtype=float)
        y = -x * 2.0 + 3.0
        r, p = compute_correlation(x, y)
        assert abs(r - (-1.0)) < 1e-10
        assert p < 0.001


class TestPermutationTest:
    """Test the permutation (randomization) test."""

    def test_null_distribution_length(self):
        """Null distribution should have n_permutations entries."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        null_dist = permutation_test(x, y, n_permutations=5000, seed=42)
        assert len(null_dist) == 5000

    def test_returns_numpy_array(self):
        """Null distribution should be a numpy array."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        null_dist = permutation_test(x, y, n_permutations=100, seed=42)
        assert isinstance(null_dist, np.ndarray)

    def test_null_values_in_valid_range(self):
        """All null correlations should be between -1 and 1."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        null_dist = permutation_test(x, y, n_permutations=500, seed=42)
        assert np.all(null_dist >= -1.0)
        assert np.all(null_dist <= 1.0)

    def test_null_distribution_centered_near_zero(self):
        """Mean of null distribution should be close to zero."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        null_dist = permutation_test(x, y, n_permutations=5000, seed=42)
        assert abs(np.mean(null_dist)) < 0.05

    def test_reproducible_with_seed(self):
        """Same seed should produce identical null distributions."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        null1 = permutation_test(x, y, n_permutations=100, seed=42)
        null2 = permutation_test(x, y, n_permutations=100, seed=42)
        np.testing.assert_array_equal(null1, null2)


class TestComputeEmpiricalPvalue:
    """Test empirical p-value computation."""

    def test_p_value_between_0_and_1(self):
        """Empirical p-value should be between 0 and 1."""
        null_dist = np.random.default_rng(42).normal(0, 0.1, 5000)
        p = compute_empirical_pvalue(0.05, null_dist)
        assert 0.0 <= p <= 1.0

    def test_large_observed_r_gives_small_p(self):
        """An observed r far from zero should have a small p-value."""
        null_dist = np.random.default_rng(42).normal(0, 0.1, 5000)
        p = compute_empirical_pvalue(0.9, null_dist)
        assert p < 0.01

    def test_zero_observed_r_gives_large_p(self):
        """An observed r near zero should have a large p-value."""
        null_dist = np.random.default_rng(42).normal(0, 0.1, 5000)
        p = compute_empirical_pvalue(0.0, null_dist)
        assert p > 0.5

    def test_consistent_with_parametric_p(self):
        """Empirical p-value should be roughly consistent with parametric."""
        data = generate_correlated_data(n=100, r=0.1, seed=42)
        x, y = data[:, 0], data[:, 1]
        r, parametric_p = compute_correlation(x, y)
        null_dist = permutation_test(x, y, n_permutations=5000, seed=42)
        empirical_p = compute_empirical_pvalue(r, null_dist)
        # Both should agree in order of magnitude; allow generous tolerance
        # since n=100 with r=0.1 is a weak effect
        assert abs(empirical_p - parametric_p) < 0.15

    def test_negative_observed_r(self):
        """Empirical p-value should work correctly with negative r."""
        null_dist = np.random.default_rng(42).normal(0, 0.1, 5000)
        p_positive = compute_empirical_pvalue(0.5, null_dist)
        p_negative = compute_empirical_pvalue(-0.5, null_dist)
        # Two-sided test: both should yield similar p-values
        assert abs(p_positive - p_negative) < 0.05


class TestZeroCorrelation:
    """Edge case: data with zero target correlation."""

    def test_zero_correlation_data(self):
        """Data generated with r=0 should have near-zero correlation."""
        data = generate_correlated_data(n=100, r=0.0, seed=42)
        observed_r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        assert abs(observed_r) < 0.25

    def test_zero_correlation_high_p_value(self):
        """With r=0, the parametric p-value should generally be large."""
        data = generate_correlated_data(n=100, r=0.0, seed=42)
        x, y = data[:, 0], data[:, 1]
        _, p = compute_correlation(x, y)
        # Most of the time p should be > 0.05 for truly uncorrelated data
        # Use a generous threshold since sampling variability exists
        assert p > 0.01

    def test_zero_correlation_empirical_p_large(self):
        """Empirical p-value for zero-correlation data should be large."""
        data = generate_correlated_data(n=100, r=0.0, seed=42)
        x, y = data[:, 0], data[:, 1]
        r, _ = compute_correlation(x, y)
        null_dist = permutation_test(x, y, n_permutations=5000, seed=42)
        empirical_p = compute_empirical_pvalue(r, null_dist)
        assert empirical_p > 0.01
