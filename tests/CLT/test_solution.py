"""Tests for the Central Limit Theorem simulation module."""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from aicoding.CLT.solution import (
    simulate_sample_means,
    run_clt_simulation,
    plot_clt,
)

DISTRIBUTIONS = ["normal", "uniform", "chi-squared", "poisson", "exponential", "beta"]


class TestSimulateSampleMeans:
    """Test the simulate_sample_means function."""

    def test_returns_numpy_array(self):
        """simulate_sample_means should return a numpy array."""
        result = simulate_sample_means("normal", n_samples=100, sample_size=30, seed=42)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_n_samples(self):
        """The returned array should have length equal to n_samples."""
        result = simulate_sample_means("uniform", n_samples=500, sample_size=30, seed=42)
        assert len(result) == 500

    def test_custom_sample_size(self):
        """Different sample_size values should produce different results."""
        means_small = simulate_sample_means("normal", n_samples=200, sample_size=5, seed=42)
        means_large = simulate_sample_means("normal", n_samples=200, sample_size=100, seed=42)
        # Larger sample size -> smaller variance of sample means (CLT)
        assert np.std(means_large) < np.std(means_small)

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        means1 = simulate_sample_means("normal", n_samples=100, sample_size=30, seed=42)
        means2 = simulate_sample_means("normal", n_samples=100, sample_size=30, seed=42)
        np.testing.assert_array_equal(means1, means2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        means1 = simulate_sample_means("normal", n_samples=100, sample_size=30, seed=42)
        means2 = simulate_sample_means("normal", n_samples=100, sample_size=30, seed=99)
        assert not np.array_equal(means1, means2)

    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_all_distributions_supported(self, dist):
        """Each of the 6 required distributions should be supported."""
        result = simulate_sample_means(dist, n_samples=50, sample_size=30, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50

    def test_invalid_distribution_raises(self):
        """An unsupported distribution name should raise a ValueError."""
        with pytest.raises(ValueError):
            simulate_sample_means("unknown_dist", n_samples=100, sample_size=30, seed=42)

    def test_normal_mean_approximately_zero(self):
        """For standard normal samples, mean of means should be close to 0."""
        means = simulate_sample_means("normal", n_samples=5000, sample_size=30, seed=42)
        assert abs(np.mean(means)) < 0.1

    def test_uniform_mean_approximately_half(self):
        """For uniform(0,1) samples, mean of means should be near 0.5."""
        means = simulate_sample_means("uniform", n_samples=5000, sample_size=30, seed=42)
        assert abs(np.mean(means) - 0.5) < 0.05

    def test_poisson_mean_positive(self):
        """For Poisson samples, all means should be non-negative."""
        means = simulate_sample_means("poisson", n_samples=1000, sample_size=30, seed=42)
        assert np.all(means >= 0)

    def test_exponential_mean_positive(self):
        """For exponential samples, all means should be positive."""
        means = simulate_sample_means("exponential", n_samples=1000, sample_size=30, seed=42)
        assert np.all(means > 0)

    def test_chi_squared_mean_positive(self):
        """For chi-squared samples, all means should be positive."""
        means = simulate_sample_means("chi-squared", n_samples=1000, sample_size=30, seed=42)
        assert np.all(means > 0)

    def test_beta_mean_between_zero_and_one(self):
        """For beta samples, all means should be between 0 and 1."""
        means = simulate_sample_means("beta", n_samples=1000, sample_size=30, seed=42)
        assert np.all(means > 0)
        assert np.all(means < 1)


class TestRunCLTSimulation:
    """Test the run_clt_simulation function."""

    def test_returns_dict(self):
        """run_clt_simulation should return a dictionary."""
        results = run_clt_simulation(n_samples=50, sample_size=30, seed=42)
        assert isinstance(results, dict)

    def test_dict_has_all_six_distributions(self):
        """The returned dict should contain all 6 distribution keys."""
        results = run_clt_simulation(n_samples=50, sample_size=30, seed=42)
        for dist in DISTRIBUTIONS:
            assert dist in results, f"Missing key: {dist}"

    def test_dict_has_only_six_keys(self):
        """The returned dict should contain exactly 6 keys."""
        results = run_clt_simulation(n_samples=50, sample_size=30, seed=42)
        assert len(results) == 6

    def test_each_value_is_numpy_array(self):
        """Each value in the results dict should be a numpy array."""
        results = run_clt_simulation(n_samples=50, sample_size=30, seed=42)
        for dist, means in results.items():
            assert isinstance(means, np.ndarray), f"{dist} value is not ndarray"

    def test_each_array_has_correct_length(self):
        """Each array should have length n_samples."""
        n = 200
        results = run_clt_simulation(n_samples=n, sample_size=30, seed=42)
        for dist, means in results.items():
            assert len(means) == n, f"{dist} array length != {n}"

    def test_reproducible_with_seed(self):
        """Same seed should yield identical results."""
        r1 = run_clt_simulation(n_samples=100, sample_size=30, seed=42)
        r2 = run_clt_simulation(n_samples=100, sample_size=30, seed=42)
        for dist in DISTRIBUTIONS:
            np.testing.assert_array_equal(r1[dist], r2[dist])


class TestPlotCLT:
    """Test the plot_clt function."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self):
        """Use non-interactive backend for testing."""
        matplotlib.use("Agg")
        yield
        plt.close("all")

    @pytest.fixture
    def sample_results(self):
        """Generate a small results dict for plotting tests."""
        return run_clt_simulation(n_samples=100, sample_size=30, seed=42)

    def test_returns_figure(self, sample_results):
        """plot_clt should return a matplotlib Figure."""
        fig = plot_clt(sample_results)
        assert isinstance(fig, Figure)

    def test_figure_has_six_axes(self, sample_results):
        """The figure should have 6 subplots (one per distribution)."""
        fig = plot_clt(sample_results)
        axes = fig.get_axes()
        assert len(axes) == 6

    def test_each_subplot_has_title(self, sample_results):
        """Each subplot should have a non-empty title."""
        fig = plot_clt(sample_results)
        axes = fig.get_axes()
        for ax in axes:
            title = ax.get_title()
            assert title != "", "A subplot has an empty title"

    def test_subplot_titles_contain_distribution_names(self, sample_results):
        """Each subplot title should reference a distribution name."""
        fig = plot_clt(sample_results)
        axes = fig.get_axes()
        titles = [ax.get_title().lower() for ax in axes]
        for dist in DISTRIBUTIONS:
            # Allow flexible matching (e.g., "chi-squared" or "chi squared")
            dist_key = dist.replace("-", "").replace(" ", "")
            found = any(
                dist_key in t.replace("-", "").replace(" ", "") for t in titles
            )
            assert found, f"No subplot title references '{dist}'"
