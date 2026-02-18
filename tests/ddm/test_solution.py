"""Tests for DDM — drift diffusion model simulation and EZ-diffusion parameter recovery."""

import numpy as np
import pytest

from aicoding.ddm.solution import simulate_ddm, ez_diffusion


class TestSimulateDDM:
    def test_returns_dict(self):
        result = simulate_ddm(n_trials=100, drift_rate=0.3, boundary=1.5, start_point=0.5)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = simulate_ddm(n_trials=100, drift_rate=0.3, boundary=1.5, start_point=0.5)
        assert "rt" in result
        assert "choice" in result or "response" in result or "choices" in result

    def test_rt_shape(self):
        n = 200
        result = simulate_ddm(n_trials=n, drift_rate=0.3, boundary=1.5, start_point=0.5)
        assert len(result["rt"]) == n

    def test_rt_positive(self):
        result = simulate_ddm(n_trials=500, drift_rate=0.3, boundary=1.5, start_point=0.5)
        assert np.all(np.array(result["rt"]) > 0)

    def test_choices_binary(self):
        result = simulate_ddm(n_trials=500, drift_rate=0.3, boundary=1.5, start_point=0.5)
        choices = np.array(result.get("choice", result.get("response", result.get("choices"))))
        unique = set(choices)
        assert len(unique) <= 2  # upper and lower boundary
        assert len(unique) >= 1

    def test_positive_drift_upper_bias(self):
        """Positive drift should produce more upper-boundary responses."""
        result = simulate_ddm(n_trials=1000, drift_rate=1.0, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(result.get("choice", result.get("response", result.get("choices"))))
        # 1 = upper boundary
        upper_frac = np.mean(choices == 1)
        assert upper_frac > 0.6

    def test_zero_drift_roughly_balanced(self):
        """Zero drift should produce roughly 50/50 choices."""
        result = simulate_ddm(n_trials=2000, drift_rate=0.0, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(result.get("choice", result.get("response", result.get("choices"))))
        upper_frac = np.mean(choices == 1)
        assert 0.3 < upper_frac < 0.7

    def test_higher_boundary_slower_rt(self):
        """Higher boundaries should produce longer RTs on average."""
        r_low = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=0.5, start_point=0.5, seed=42)
        r_high = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=2.0, start_point=0.5, seed=42)
        assert np.mean(r_high["rt"]) > np.mean(r_low["rt"])

    def test_reproducibility(self):
        r1 = simulate_ddm(n_trials=100, drift_rate=0.3, boundary=1.5, start_point=0.5, seed=7)
        r2 = simulate_ddm(n_trials=100, drift_rate=0.3, boundary=1.5, start_point=0.5, seed=7)
        np.testing.assert_array_equal(r1["rt"], r2["rt"])


class TestEZDiffusion:
    def test_returns_dict(self):
        sim = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        result = ez_diffusion(sim["rt"], np.array(sim.get("choice", sim.get("response", sim.get("choices")))))
        assert isinstance(result, dict)

    def test_has_drift_rate(self):
        sim = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        assert "drift_rate" in result or "v" in result

    def test_has_boundary(self):
        sim = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        assert "boundary" in result or "a" in result

    def test_has_nondecision_time(self):
        sim = simulate_ddm(n_trials=1000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        assert "nondecision_time" in result or "t0" in result or "ter" in result

    def test_recovers_drift_rate_direction(self):
        """Recovered drift rate should have the same sign as the true drift."""
        sim = simulate_ddm(n_trials=2000, drift_rate=0.8, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        v_hat = result.get("drift_rate", result.get("v"))
        assert v_hat > 0

    def test_recovers_drift_rate_approximate(self):
        """Recovered drift rate should be in the right ballpark."""
        sim = simulate_ddm(n_trials=5000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        v_hat = result.get("drift_rate", result.get("v"))
        assert abs(v_hat - 0.5) < 0.3  # approximate recovery

    def test_recovers_boundary_approximate(self):
        """Recovered boundary should be in the right ballpark."""
        sim = simulate_ddm(n_trials=5000, drift_rate=0.5, boundary=1.5, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        a_hat = result.get("boundary", result.get("a"))
        assert abs(a_hat - 1.5) < 0.5  # approximate recovery

    def test_all_values_finite(self):
        sim = simulate_ddm(n_trials=1000, drift_rate=0.3, boundary=1.0, start_point=0.5, seed=42)
        choices = np.array(sim.get("choice", sim.get("response", sim.get("choices"))))
        result = ez_diffusion(sim["rt"], choices)
        for v in result.values():
            assert np.isfinite(v)
