"""Tests for ddm_collapsing_plot — DDM with collapsing bounds."""

import numpy as np
import pytest

from aicoding.ddm_collapsing_plot.solution import simulate_ddm_collapsing


class TestSimulateDDMCollapsing:
    def test_returns_dict(self):
        result = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert isinstance(result, dict)

    def test_has_rt(self):
        result = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert "rt" in result

    def test_has_choices(self):
        result = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert "choice" in result

    def test_has_timeseries(self):
        """Must return the full timeseries of diffusion steps."""
        result = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert "timeseries" in result or "trajectories" in result

    def test_timeseries_is_list(self):
        result = simulate_ddm_collapsing(n_trials=10, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        ts = result.get("timeseries", result.get("trajectories"))
        assert isinstance(ts, list)
        assert len(ts) == 10

    def test_timeseries_entries_are_arrays(self):
        result = simulate_ddm_collapsing(n_trials=10, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        ts = result.get("timeseries", result.get("trajectories"))
        for traj in ts:
            assert isinstance(traj, np.ndarray)
            assert len(traj) > 0

    def test_rt_shape(self):
        result = simulate_ddm_collapsing(n_trials=100, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert len(result["rt"]) == 100

    def test_rt_positive(self):
        result = simulate_ddm_collapsing(n_trials=100, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        assert np.all(np.array(result["rt"]) > 0)

    def test_choices_binary(self):
        result = simulate_ddm_collapsing(n_trials=100, drift_rate=0.5, initial_boundary=2.0,
                                          collapse_rate=0.01, seed=42)
        choices = np.array(result["choice"])
        assert set(choices).issubset({0, 1})

    def test_collapsing_bounds_faster_than_fixed(self):
        """Collapsing bounds should produce shorter mean RTs than fixed bounds."""
        fixed = simulate_ddm_collapsing(n_trials=500, drift_rate=0.3, initial_boundary=2.0,
                                         collapse_rate=0.0, seed=42)
        collapsing = simulate_ddm_collapsing(n_trials=500, drift_rate=0.3, initial_boundary=2.0,
                                              collapse_rate=0.05, seed=42)
        assert np.mean(collapsing["rt"]) < np.mean(fixed["rt"])

    def test_reproducibility(self):
        r1 = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                      collapse_rate=0.01, seed=7)
        r2 = simulate_ddm_collapsing(n_trials=50, drift_rate=0.5, initial_boundary=2.0,
                                      collapse_rate=0.01, seed=7)
        np.testing.assert_array_equal(r1["rt"], r2["rt"])
