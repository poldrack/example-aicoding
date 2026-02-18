"""Tests for GES — greedy equivalence search for causal discovery."""

import numpy as np
import pytest

from aicoding.GES.solution import (
    greedy_equivalence_search,
    generate_test_data,
    bic_score,
)


@pytest.fixture
def simple_chain_data():
    """Generate data from X -> Y -> Z."""
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n)
    Y = 0.8 * X + rng.randn(n) * 0.3
    Z = 0.6 * Y + rng.randn(n) * 0.3
    return np.column_stack([X, Y, Z])


@pytest.fixture
def independent_data():
    """Three independent variables."""
    rng = np.random.RandomState(42)
    n = 500
    return rng.randn(n, 3)


class TestGenerateTestData:
    def test_returns_array(self):
        data, graph = generate_test_data(n_samples=100, n_variables=4, seed=42)
        assert isinstance(data, np.ndarray)

    def test_correct_shape(self):
        data, graph = generate_test_data(n_samples=200, n_variables=5, seed=42)
        assert data.shape == (200, 5)

    def test_graph_is_dict_or_matrix(self):
        data, graph = generate_test_data(n_samples=100, n_variables=4, seed=42)
        assert isinstance(graph, (dict, np.ndarray))

    def test_reproducibility(self):
        d1, g1 = generate_test_data(n_samples=100, n_variables=4, seed=7)
        d2, g2 = generate_test_data(n_samples=100, n_variables=4, seed=7)
        np.testing.assert_array_equal(d1, d2)


class TestBicScore:
    def test_returns_float(self, simple_chain_data):
        # BIC for Y given X
        score = bic_score(simple_chain_data, child=1, parents=[0])
        assert isinstance(score, float)

    def test_finite(self, simple_chain_data):
        score = bic_score(simple_chain_data, child=1, parents=[0])
        assert np.isfinite(score)

    def test_better_parent_higher_score(self, simple_chain_data):
        """True parent should give a better (higher) BIC score than no parent."""
        score_with = bic_score(simple_chain_data, child=1, parents=[0])
        score_without = bic_score(simple_chain_data, child=1, parents=[])
        assert score_with > score_without

    def test_empty_parents(self, simple_chain_data):
        score = bic_score(simple_chain_data, child=0, parents=[])
        assert np.isfinite(score)


class TestGES:
    def test_returns_adjacency(self, simple_chain_data):
        result = greedy_equivalence_search(simple_chain_data)
        assert isinstance(result, (dict, np.ndarray))

    def test_detects_edge_in_chain(self, simple_chain_data):
        """Should detect X-Y and Y-Z edges in X->Y->Z."""
        result = greedy_equivalence_search(simple_chain_data)
        if isinstance(result, np.ndarray):
            # Should have edges involving node 1 (Y)
            assert result[0, 1] != 0 or result[1, 0] != 0  # X-Y edge
            assert result[1, 2] != 0 or result[2, 1] != 0  # Y-Z edge
        else:
            # Dict: check adjacency
            adj_0 = result.get(0, set())
            adj_1 = result.get(1, set())
            assert 1 in adj_0 or 0 in adj_1  # X-Y

    def test_sparse_for_independent_data(self, independent_data):
        """Independent variables should produce a sparse/empty graph."""
        result = greedy_equivalence_search(independent_data)
        if isinstance(result, np.ndarray):
            n_edges = np.sum(result != 0)
            assert n_edges <= 2  # allow small false positives
        else:
            total = sum(len(v) for v in result.values())
            assert total <= 2

    def test_generated_data_recoverable(self):
        """GES should recover some edges from generated test data."""
        data, true_graph = generate_test_data(n_samples=1000, n_variables=4, seed=42)
        result = greedy_equivalence_search(data)
        # At least one edge should be found
        if isinstance(result, np.ndarray):
            assert np.sum(result != 0) > 0
        else:
            total = sum(len(v) for v in result.values())
            assert total > 0

    def test_handles_two_variables(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200)
        Y = 0.9 * X + rng.randn(200) * 0.2
        data = np.column_stack([X, Y])
        result = greedy_equivalence_search(data)
        assert result is not None

    def test_result_is_dag(self):
        """Estimated graph should be a DAG (no bidirectional or cyclic edges)."""
        data, true_adj = generate_test_data(n_samples=1000, n_variables=5, seed=42)
        adj = greedy_equivalence_search(data)
        # No bidirectional edges: if adj[i,j]==1, then adj[j,i]==0
        for i in range(adj.shape[0]):
            for j in range(i + 1, adj.shape[1]):
                assert not (adj[i, j] != 0 and adj[j, i] != 0), (
                    f"Bidirectional edge between {i} and {j}"
                )
        # No cycles: topological sort should succeed
        p = adj.shape[0]
        in_degree = np.sum(adj != 0, axis=0)
        queue = list(np.where(in_degree == 0)[0])
        visited = 0
        remaining = in_degree.copy()
        while queue:
            node = queue.pop(0)
            visited += 1
            for j in range(p):
                if adj[node, j] != 0:
                    remaining[j] -= 1
                    if remaining[j] == 0:
                        queue.append(j)
        assert visited == p, "Graph contains a cycle"

    def test_recovers_correct_skeleton(self):
        """GES should recover the correct edge skeleton (ignoring direction)."""
        data, true_adj = generate_test_data(n_samples=1000, n_variables=5, seed=42)
        adj = greedy_equivalence_search(data)
        true_edges = {frozenset(e) for e in zip(*np.where(true_adj != 0))}
        est_edges = {frozenset(e) for e in zip(*np.where(adj != 0))}
        # At least half the true edges should be recovered
        overlap = true_edges & est_edges
        assert len(overlap) >= len(true_edges) / 2

    def test_recovers_v_structure_directions(self):
        """GES should orient edges correctly at v-structures (colliders).

        Data: 0→2←1 (v-structure at node 2, nodes 0 and 1 independent).
        """
        rng = np.random.RandomState(42)
        n = 2000
        X0 = rng.randn(n)
        X1 = rng.randn(n)
        X2 = 0.8 * X0 + 0.8 * X1 + rng.randn(n) * 0.3
        data = np.column_stack([X0, X1, X2])

        adj = greedy_equivalence_search(data)

        # Both 0 and 1 should be parents of 2
        assert adj[0, 2] != 0, "Missing edge 0→2"
        assert adj[1, 2] != 0, "Missing edge 1→2"
        # 0 and 1 should NOT be adjacent
        assert adj[0, 1] == 0 and adj[1, 0] == 0, "Spurious edge 0-1"
