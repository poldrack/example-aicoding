"""Tests for logisticmap — logistic map simulation and bifurcation plot."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aicoding.logisticmap.solution import logistic_map, bifurcation_data, plot_bifurcation


class TestLogisticMap:
    def test_returns_array(self):
        result = logistic_map(r=3.5, x0=0.5, n_steps=100)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self):
        result = logistic_map(r=3.5, x0=0.5, n_steps=100)
        assert len(result) == 100

    def test_values_between_0_and_1(self):
        result = logistic_map(r=2.5, x0=0.5, n_steps=200)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_fixed_point_r_below_1(self):
        """For r < 1, iterates should converge to 0."""
        result = logistic_map(r=0.5, x0=0.5, n_steps=200)
        assert result[-1] < 0.01

    def test_fixed_point_r_2(self):
        """For r=2, should converge to (r-1)/r = 0.5."""
        result = logistic_map(r=2.0, x0=0.1, n_steps=500)
        assert abs(result[-1] - 0.5) < 0.01

    def test_period_2_cycle(self):
        """For r=3.2, should oscillate between two values."""
        result = logistic_map(r=3.2, x0=0.5, n_steps=500)
        last_values = result[-50:]
        unique_approx = len(set(np.round(last_values, 4)))
        assert unique_approx <= 3  # 2 values + possible rounding

    def test_chaos_r_4(self):
        """For r=4, should be chaotic (many distinct values)."""
        result = logistic_map(r=4.0, x0=0.2, n_steps=200)
        unique = len(set(np.round(result[-100:], 6)))
        assert unique > 10


class TestBifurcationData:
    def test_returns_arrays(self):
        r_vals, x_vals = bifurcation_data(r_min=2.5, r_max=4.0, r_steps=50,
                                           n_iterations=200, n_last=50)
        assert isinstance(r_vals, np.ndarray)
        assert isinstance(x_vals, np.ndarray)

    def test_shapes_match(self):
        r_vals, x_vals = bifurcation_data(r_min=2.5, r_max=4.0, r_steps=50,
                                           n_iterations=200, n_last=50)
        assert len(r_vals) == len(x_vals)

    def test_r_range(self):
        r_vals, x_vals = bifurcation_data(r_min=2.5, r_max=4.0, r_steps=50,
                                           n_iterations=200, n_last=50)
        assert np.min(r_vals) >= 2.5
        assert np.max(r_vals) <= 4.0

    def test_x_range(self):
        r_vals, x_vals = bifurcation_data(r_min=2.5, r_max=4.0, r_steps=50,
                                           n_iterations=200, n_last=50)
        assert np.all(x_vals >= 0) and np.all(x_vals <= 1)


class TestPlotBifurcation:
    def test_creates_figure(self):
        fig = plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=50)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        output = str(tmp_path / "bifurcation.png")
        fig = plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=50, save_path=output)
        import os
        assert os.path.exists(output)
        plt.close(fig)

    def test_axes_labels(self):
        fig = plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=50)
        ax = fig.axes[0]
        assert ax.get_xlabel() != "" or ax.get_ylabel() != ""
        plt.close(fig)
