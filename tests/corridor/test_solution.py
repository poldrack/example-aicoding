"""Tests for corridor — Schönbrodt corridor of stability simulation."""

import os

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aicoding.corridor.solution import (
    simulate_corridor_of_stability,
    plot_corridor_of_stability,
)


class TestSimulateCorridor:
    def test_returns_dict(self):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=50, seed=42)
        assert isinstance(result, dict)

    def test_has_sample_sizes(self):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=50, seed=42)
        assert "sample_sizes" in result

    def test_has_correlations(self):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=50, seed=42)
        assert "correlations" in result

    def test_sample_sizes_range(self):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=200, n_simulations=20, seed=42)
        sizes = result["sample_sizes"]
        assert min(sizes) >= 3  # need at least 3 for correlation
        assert max(sizes) <= 200

    def test_correlations_shape(self):
        n_sims = 30
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=n_sims, seed=42)
        corrs = result["correlations"]
        assert len(corrs) == n_sims  # one array per simulation
        assert len(corrs[0]) == len(result["sample_sizes"])

    def test_correlations_bounded(self):
        result = simulate_corridor_of_stability(true_r=0.5, max_n=100, n_simulations=20, seed=42)
        for arr in result["correlations"]:
            assert np.all(np.array(arr) >= -1)
            assert np.all(np.array(arr) <= 1)

    def test_variance_decreases_with_n(self):
        """Correlation estimates should become less variable as n increases."""
        result = simulate_corridor_of_stability(true_r=0.3, max_n=500, n_simulations=200, seed=42)
        corrs = np.array(result["correlations"])
        sizes = result["sample_sizes"]
        # Compare variance at small n vs large n
        small_idx = np.argmin(np.abs(np.array(sizes) - 20))
        large_idx = np.argmin(np.abs(np.array(sizes) - 400))
        var_small = np.var(corrs[:, small_idx])
        var_large = np.var(corrs[:, large_idx])
        assert var_large < var_small

    def test_converges_to_true_r(self):
        """At large n, mean correlation should be close to true r."""
        result = simulate_corridor_of_stability(true_r=0.5, max_n=500, n_simulations=200, seed=42)
        corrs = np.array(result["correlations"])
        mean_at_end = np.mean(corrs[:, -1])
        assert abs(mean_at_end - 0.5) < 0.1

    def test_zero_correlation(self):
        result = simulate_corridor_of_stability(true_r=0.0, max_n=200, n_simulations=50, seed=42)
        corrs = np.array(result["correlations"])
        mean_at_end = np.mean(corrs[:, -1])
        assert abs(mean_at_end) < 0.15

    def test_reproducibility(self):
        r1 = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=20, seed=7)
        r2 = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=20, seed=7)
        np.testing.assert_array_equal(r1["correlations"], r2["correlations"])


class TestPlotCorridor:
    def test_creates_figure(self):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=50, seed=42)
        fig = plot_corridor_of_stability(result, true_r=0.3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        result = simulate_corridor_of_stability(true_r=0.3, max_n=100, n_simulations=50, seed=42)
        output = str(tmp_path / "corridor.png")
        fig = plot_corridor_of_stability(result, true_r=0.3, save_path=output)
        assert os.path.exists(output)
        plt.close(fig)
