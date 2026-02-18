"""Tests for the PC causal inference algorithm module."""

import numpy as np
import pytest

from aicoding.PC.solution import (
    PCAlgorithm,
    generate_synthetic_data,
    partial_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate the standard synthetic dataset used across tests."""
    return generate_synthetic_data(n=1000, seed=42)


@pytest.fixture
def simple_correlated_data():
    """Two perfectly correlated variables plus one independent variable."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500) * 0.01  # almost perfect correlation
    z = rng.standard_normal(500)              # independent
    return np.column_stack([x, y, z])


@pytest.fixture
def pc():
    """Return a default PCAlgorithm instance."""
    return PCAlgorithm(alpha=0.05)


# ---------------------------------------------------------------------------
# Tests for generate_synthetic_data
# ---------------------------------------------------------------------------

class TestGenerateSyntheticData:
    """Tests for the synthetic data generator."""

    def test_returns_numpy_array(self, synthetic_data):
        """generate_synthetic_data should return a numpy ndarray."""
        assert isinstance(synthetic_data, np.ndarray)

    def test_shape(self, synthetic_data):
        """Default call should produce (1000, 4) array."""
        assert synthetic_data.shape == (1000, 4)

    def test_custom_n(self):
        """Requesting a different n should change the number of rows."""
        data = generate_synthetic_data(n=200, seed=0)
        assert data.shape[0] == 200

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        d1 = generate_synthetic_data(n=100, seed=7)
        d2 = generate_synthetic_data(n=100, seed=7)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        d1 = generate_synthetic_data(n=100, seed=1)
        d2 = generate_synthetic_data(n=100, seed=2)
        assert not np.array_equal(d1, d2)

    def test_at_least_four_variables(self):
        """The synthetic dataset must have at least 4 columns."""
        data = generate_synthetic_data(n=50, seed=0)
        assert data.shape[1] >= 4


# ---------------------------------------------------------------------------
# Tests for partial_correlation
# ---------------------------------------------------------------------------

class TestPartialCorrelation:
    """Tests for the partial_correlation function."""

    def test_unconditional_correlation_positive(self, simple_correlated_data):
        """Two highly correlated vars should have a large positive partial corr
        when conditioning set is empty."""
        r, p = partial_correlation(simple_correlated_data, 0, 1, [])
        assert r > 0.9
        assert p < 0.01

    def test_independent_variables(self, simple_correlated_data):
        """An independent variable should have a near-zero partial corr and a
        high p-value."""
        r, p = partial_correlation(simple_correlated_data, 0, 2, [])
        assert abs(r) < 0.15
        assert p > 0.01

    def test_returns_tuple_of_two_floats(self, synthetic_data):
        """partial_correlation should return (r, p) as floats."""
        result = partial_correlation(synthetic_data, 0, 1, [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        r, p = result
        assert isinstance(r, float)
        assert isinstance(p, float)

    def test_correlation_bounded(self, synthetic_data):
        """Partial correlation coefficient must be in [-1, 1]."""
        r, _ = partial_correlation(synthetic_data, 0, 1, [2])
        assert -1.0 <= r <= 1.0

    def test_p_value_bounded(self, synthetic_data):
        """p-value must be in [0, 1]."""
        _, p = partial_correlation(synthetic_data, 0, 1, [2])
        assert 0.0 <= p <= 1.0

    def test_conditioning_on_mediator_removes_association(self):
        """If X -> M -> Y (pure mediation), conditioning on M should make
        the partial correlation between X and Y near zero."""
        rng = np.random.default_rng(99)
        n = 2000
        x = rng.standard_normal(n)
        m = 2.0 * x + rng.standard_normal(n) * 0.3
        y = 1.5 * m + rng.standard_normal(n) * 0.3
        data = np.column_stack([x, m, y])

        # Unconditional: X and Y should be correlated
        r_unc, p_unc = partial_correlation(data, 0, 2, [])
        assert abs(r_unc) > 0.5

        # Conditional on M: X and Y should be (near-)independent
        r_cond, p_cond = partial_correlation(data, 0, 2, [1])
        assert abs(r_cond) < 0.15

    def test_symmetry(self, synthetic_data):
        """partial_correlation(X, i, j, S) == partial_correlation(X, j, i, S)."""
        r1, p1 = partial_correlation(synthetic_data, 0, 2, [1])
        r2, p2 = partial_correlation(synthetic_data, 2, 0, [1])
        np.testing.assert_almost_equal(r1, r2, decimal=10)
        np.testing.assert_almost_equal(p1, p2, decimal=10)


# ---------------------------------------------------------------------------
# Tests for PCAlgorithm
# ---------------------------------------------------------------------------

class TestPCAlgorithm:
    """Tests for the PCAlgorithm class."""

    def test_fit_returns_self(self, pc, synthetic_data):
        """fit() should return the PCAlgorithm instance for chaining."""
        result = pc.fit(synthetic_data)
        assert result is pc

    def test_skeleton_is_set_of_frozensets(self, pc, synthetic_data):
        """After fit, skeleton should be a set of frozensets (undirected edges)."""
        pc.fit(synthetic_data)
        assert isinstance(pc.skeleton, set)
        for edge in pc.skeleton:
            assert isinstance(edge, frozenset)
            assert len(edge) == 2

    def test_separation_sets_stored(self, pc, synthetic_data):
        """After fit, separation_sets should be a dict mapping node-pairs to
        conditioning sets."""
        pc.fit(synthetic_data)
        assert isinstance(pc.separation_sets, dict)

    def test_number_of_nodes(self, pc, synthetic_data):
        """The algorithm should know how many variables are in the data."""
        pc.fit(synthetic_data)
        assert pc.num_nodes == synthetic_data.shape[1]

    def test_detects_known_edges_in_synthetic_data(self, pc, synthetic_data):
        """For the known DAG X0->X2, X1->X2, X2->X3, the skeleton must contain
        edges (0,2), (1,2), (2,3)."""
        pc.fit(synthetic_data)
        expected_edges = {frozenset({0, 2}), frozenset({1, 2}), frozenset({2, 3})}
        for edge in expected_edges:
            assert edge in pc.skeleton, (
                f"Expected edge {edge} not found in skeleton {pc.skeleton}"
            )

    def test_removes_non_adjacent_edges(self, pc, synthetic_data):
        """For the known DAG, edge (0,1) should NOT be in the skeleton
        because X0 and X1 are independent."""
        pc.fit(synthetic_data)
        assert frozenset({0, 1}) not in pc.skeleton

    def test_alpha_affects_skeleton(self, synthetic_data):
        """A very small alpha should remove more edges (more conservative)."""
        pc_strict = PCAlgorithm(alpha=1e-10)
        pc_strict.fit(synthetic_data)
        pc_loose = PCAlgorithm(alpha=0.5)
        pc_loose.fit(synthetic_data)
        # The loose skeleton should have at least as many edges
        assert len(pc_loose.skeleton) >= len(pc_strict.skeleton)

    def test_fit_on_independent_data(self):
        """Fitting on fully independent columns should yield a very sparse
        skeleton (ideally empty, but allow for rare false positives)."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((2000, 4))
        pc = PCAlgorithm(alpha=0.01)  # stricter alpha to reduce false positives
        pc.fit(data)
        # With independent data and strict alpha, at most 1 spurious edge
        assert len(pc.skeleton) <= 1

    def test_fit_on_two_variable_chain(self):
        """Simple two-variable case: X -> Y should yield one edge."""
        rng = np.random.default_rng(55)
        n = 1000
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.5
        data = np.column_stack([x, y])

        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        assert frozenset({0, 1}) in pc.skeleton

    def test_no_self_loops(self, pc, synthetic_data):
        """Skeleton edges should never be self-loops."""
        pc.fit(synthetic_data)
        for edge in pc.skeleton:
            nodes = list(edge)
            assert len(nodes) == 2
            assert nodes[0] != nodes[1]

    def test_default_alpha(self):
        """Default alpha should be 0.05."""
        pc = PCAlgorithm()
        assert pc.alpha == 0.05


# ---------------------------------------------------------------------------
# Edge-case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_single_variable_returns_empty_skeleton(self):
        """A single-column dataset has no pairs, so skeleton should be empty."""
        data = np.random.default_rng(0).standard_normal((100, 1))
        pc = PCAlgorithm()
        pc.fit(data)
        assert len(pc.skeleton) == 0

    def test_large_conditioning_set_does_not_crash(self):
        """Even when conditioning set size approaches p-2, no crash should occur."""
        rng = np.random.default_rng(10)
        data = rng.standard_normal((200, 5))
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        # Just verify it runs without error; skeleton content is not prescribed.
        assert isinstance(pc.skeleton, set)
