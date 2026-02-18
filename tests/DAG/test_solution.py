"""Tests for the DAG (directed graph / d-separation) module."""

import pytest
import networkx as nx
from aicoding.DAG.solution import DirectedGraph, d_separated


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_graph():
    """A -> B -> C  (simple chain)."""
    g = DirectedGraph()
    for node in ["A", "B", "C"]:
        g.add_node(node)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    return g


@pytest.fixture
def fork_graph():
    """B <- A -> C  (fork / common cause)."""
    g = DirectedGraph()
    for node in ["A", "B", "C"]:
        g.add_node(node)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    return g


@pytest.fixture
def collider_graph():
    """A -> C <- B  (collider / common effect)."""
    g = DirectedGraph()
    for node in ["A", "B", "C"]:
        g.add_node(node)
    g.add_edge("A", "C")
    g.add_edge("B", "C")
    return g


@pytest.fixture
def complex_graph():
    """
    A more complex DAG:
        A -> C
        B -> C
        C -> D
        C -> E
        D -> F
        E -> F
    """
    g = DirectedGraph()
    for node in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(node)
    g.add_edge("A", "C")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("C", "E")
    g.add_edge("D", "F")
    g.add_edge("E", "F")
    return g


@pytest.fixture
def nx_chain():
    """NetworkX version of chain A -> B -> C."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C")])
    return G


@pytest.fixture
def nx_fork():
    """NetworkX version of fork B <- A -> C."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C")])
    return G


@pytest.fixture
def nx_collider():
    """NetworkX version of collider A -> C <- B."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "C"), ("B", "C")])
    return G


@pytest.fixture
def nx_complex():
    """NetworkX version of the complex graph."""
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "C"), ("B", "C"), ("C", "D"),
        ("C", "E"), ("D", "F"), ("E", "F"),
    ])
    return G


# ---------------------------------------------------------------------------
# Tests: DirectedGraph class
# ---------------------------------------------------------------------------

class TestDirectedGraphConstruction:
    """Test DirectedGraph class construction and basic operations."""

    def test_add_node(self):
        """Adding a node should make it appear in the graph's node set."""
        g = DirectedGraph()
        g.add_node("X")
        assert "X" in g.get_nodes()

    def test_add_multiple_nodes(self):
        """Multiple nodes can be added."""
        g = DirectedGraph()
        g.add_node("A")
        g.add_node("B")
        assert {"A", "B"} == set(g.get_nodes())

    def test_add_edge(self):
        """Adding an edge should record the relationship."""
        g = DirectedGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        assert "B" in g.get_children("A")
        assert "A" in g.get_parents("B")

    def test_add_edge_auto_adds_nodes(self):
        """Adding an edge for nodes not yet present should auto-add them."""
        g = DirectedGraph()
        g.add_edge("X", "Y")
        assert "X" in g.get_nodes()
        assert "Y" in g.get_nodes()

    def test_get_children(self, chain_graph):
        """get_children should return direct successors."""
        assert set(chain_graph.get_children("A")) == {"B"}
        assert set(chain_graph.get_children("B")) == {"C"}
        assert set(chain_graph.get_children("C")) == set()

    def test_get_parents(self, chain_graph):
        """get_parents should return direct predecessors."""
        assert set(chain_graph.get_parents("A")) == set()
        assert set(chain_graph.get_parents("B")) == {"A"}
        assert set(chain_graph.get_parents("C")) == {"B"}

    def test_get_nodes_returns_all(self, complex_graph):
        """get_nodes should return all nodes in the graph."""
        assert set(complex_graph.get_nodes()) == {"A", "B", "C", "D", "E", "F"}

    def test_duplicate_node_is_idempotent(self):
        """Adding the same node twice should not create duplicates."""
        g = DirectedGraph()
        g.add_node("A")
        g.add_node("A")
        assert list(g.get_nodes()).count("A") == 1

    def test_duplicate_edge_is_idempotent(self):
        """Adding the same edge twice should not create duplicates."""
        g = DirectedGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        children = list(g.get_children("A"))
        assert children.count("B") == 1


# ---------------------------------------------------------------------------
# Tests: d-separation on chains
# ---------------------------------------------------------------------------

class TestDSeparationChain:
    """d-separation tests on A -> B -> C."""

    def test_chain_unconditional_not_separated(self, chain_graph):
        """A and C are NOT d-separated given empty set (info flows through B)."""
        assert d_separated(chain_graph, {"A"}, {"C"}, set()) is False

    def test_chain_conditioned_on_middle(self, chain_graph):
        """A and C ARE d-separated given {B} (chain blocked)."""
        assert d_separated(chain_graph, {"A"}, {"C"}, {"B"}) is True

    def test_chain_adjacent_not_separated(self, chain_graph):
        """A and B are NOT d-separated given empty set."""
        assert d_separated(chain_graph, {"A"}, {"B"}, set()) is False


# ---------------------------------------------------------------------------
# Tests: d-separation on forks
# ---------------------------------------------------------------------------

class TestDSeparationFork:
    """d-separation tests on B <- A -> C."""

    def test_fork_unconditional_not_separated(self, fork_graph):
        """B and C are NOT d-separated given empty set (common cause A)."""
        assert d_separated(fork_graph, {"B"}, {"C"}, set()) is False

    def test_fork_conditioned_on_parent(self, fork_graph):
        """B and C ARE d-separated given {A} (fork blocked)."""
        assert d_separated(fork_graph, {"B"}, {"C"}, {"A"}) is True


# ---------------------------------------------------------------------------
# Tests: d-separation on colliders
# ---------------------------------------------------------------------------

class TestDSeparationCollider:
    """d-separation tests on A -> C <- B."""

    def test_collider_unconditional_separated(self, collider_graph):
        """A and B ARE d-separated given empty set (collider blocks)."""
        assert d_separated(collider_graph, {"A"}, {"B"}, set()) is True

    def test_collider_conditioned_on_collider(self, collider_graph):
        """A and B are NOT d-separated given {C} (conditioning on collider opens path)."""
        assert d_separated(collider_graph, {"A"}, {"B"}, {"C"}) is False


# ---------------------------------------------------------------------------
# Tests: d-separation on complex graph
# ---------------------------------------------------------------------------

class TestDSeparationComplex:
    """d-separation on the more complex six-node graph."""

    def test_complex_a_b_given_empty(self, complex_graph):
        """A and B are d-separated given empty set (collider at C)."""
        assert d_separated(complex_graph, {"A"}, {"B"}, set()) is True

    def test_complex_a_b_given_c(self, complex_graph):
        """A and B are NOT d-separated given {C} (conditioning on collider)."""
        assert d_separated(complex_graph, {"A"}, {"B"}, {"C"}) is False

    def test_complex_a_b_given_d(self, complex_graph):
        """A and B are NOT d-separated given {D} (descendant of collider)."""
        assert d_separated(complex_graph, {"A"}, {"B"}, {"D"}) is False

    def test_complex_d_e_given_c(self, complex_graph):
        """D and E are d-separated given {C}."""
        assert d_separated(complex_graph, {"D"}, {"E"}, {"C"}) is True

    def test_complex_a_f_given_empty(self, complex_graph):
        """A and F are NOT d-separated given empty set (path through C, D)."""
        assert d_separated(complex_graph, {"A"}, {"F"}, set()) is False

    def test_complex_a_f_given_c(self, complex_graph):
        """A and F are d-separated given {C} (blocks all paths)."""
        assert d_separated(complex_graph, {"A"}, {"F"}, {"C"}) is True


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestDSeparationEdgeCases:
    """Edge case tests for d_separated."""

    def test_empty_graph(self):
        """In an empty graph, any two empty sets are trivially d-separated."""
        g = DirectedGraph()
        assert d_separated(g, set(), set(), set()) is True

    def test_single_node_disjoint_sets(self):
        """Single node graph with X in one set, empty other set."""
        g = DirectedGraph()
        g.add_node("A")
        assert d_separated(g, {"A"}, set(), set()) is True

    def test_single_node_same_node_both_sets(self):
        """If x and y share a node the result should be False (not d-separated)."""
        g = DirectedGraph()
        g.add_node("A")
        assert d_separated(g, {"A"}, {"A"}, set()) is False

    def test_disconnected_nodes(self):
        """Two disconnected nodes are d-separated."""
        g = DirectedGraph()
        g.add_node("A")
        g.add_node("B")
        assert d_separated(g, {"A"}, {"B"}, set()) is True

    def test_self_loop_ignored_for_dsep(self):
        """A self-loop should not break d-separation queries."""
        g = DirectedGraph()
        g.add_edge("A", "A")
        g.add_edge("A", "B")
        # A and B are connected, so not d-separated given empty
        assert d_separated(g, {"A"}, {"B"}, set()) is False


# ---------------------------------------------------------------------------
# Tests: comparison with NetworkX d_separation
# ---------------------------------------------------------------------------

def _to_nx(graph):
    """Convert our DirectedGraph to a NetworkX DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(graph.get_nodes())
    for node in graph.get_nodes():
        for child in graph.get_children(node):
            G.add_edge(node, child)
    return G


class TestAgainstNetworkX:
    """Validate d_separated results match NetworkX's nx.is_d_separator."""

    @pytest.mark.parametrize(
        "set_x, set_y, set_z",
        [
            ({"A"}, {"C"}, set()),
            ({"A"}, {"C"}, {"B"}),
            ({"A"}, {"B"}, set()),
        ],
    )
    def test_chain_matches_nx(self, chain_graph, nx_chain, set_x, set_y, set_z):
        ours = d_separated(chain_graph, set_x, set_y, set_z)
        theirs = nx.is_d_separator(nx_chain, set_x, set_y, set_z)
        assert ours == theirs, (
            f"Mismatch for chain: X={set_x}, Y={set_y}, Z={set_z}: "
            f"ours={ours}, nx={theirs}"
        )

    @pytest.mark.parametrize(
        "set_x, set_y, set_z",
        [
            ({"B"}, {"C"}, set()),
            ({"B"}, {"C"}, {"A"}),
        ],
    )
    def test_fork_matches_nx(self, fork_graph, nx_fork, set_x, set_y, set_z):
        ours = d_separated(fork_graph, set_x, set_y, set_z)
        theirs = nx.is_d_separator(nx_fork, set_x, set_y, set_z)
        assert ours == theirs, (
            f"Mismatch for fork: X={set_x}, Y={set_y}, Z={set_z}: "
            f"ours={ours}, nx={theirs}"
        )

    @pytest.mark.parametrize(
        "set_x, set_y, set_z",
        [
            ({"A"}, {"B"}, set()),
            ({"A"}, {"B"}, {"C"}),
        ],
    )
    def test_collider_matches_nx(self, collider_graph, nx_collider, set_x, set_y, set_z):
        ours = d_separated(collider_graph, set_x, set_y, set_z)
        theirs = nx.is_d_separator(nx_collider, set_x, set_y, set_z)
        assert ours == theirs, (
            f"Mismatch for collider: X={set_x}, Y={set_y}, Z={set_z}: "
            f"ours={ours}, nx={theirs}"
        )

    @pytest.mark.parametrize(
        "set_x, set_y, set_z",
        [
            ({"A"}, {"B"}, set()),
            ({"A"}, {"B"}, {"C"}),
            ({"A"}, {"B"}, {"D"}),
            ({"D"}, {"E"}, {"C"}),
            ({"A"}, {"F"}, set()),
            ({"A"}, {"F"}, {"C"}),
            ({"A"}, {"F"}, {"D", "E"}),
            ({"B"}, {"D"}, set()),
            ({"B"}, {"D"}, {"C"}),
        ],
    )
    def test_complex_matches_nx(self, complex_graph, nx_complex, set_x, set_y, set_z):
        ours = d_separated(complex_graph, set_x, set_y, set_z)
        theirs = nx.is_d_separator(nx_complex, set_x, set_y, set_z)
        assert ours == theirs, (
            f"Mismatch for complex: X={set_x}, Y={set_y}, Z={set_z}: "
            f"ours={ours}, nx={theirs}"
        )

    def test_exhaustive_random_dag_against_nx(self):
        """Build a random DAG and test all pairs of singletons with various
        conditioning sets against NetworkX."""
        import random
        random.seed(42)

        # Build a random DAG on 8 nodes using topological ordering
        nodes = list(range(8))
        nx_g = nx.DiGraph()
        nx_g.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < 0.35:
                    nx_g.add_edge(nodes[i], nodes[j])

        # Mirror into our DirectedGraph
        g = DirectedGraph()
        for n in nx_g.nodes:
            g.add_node(n)
        for u, v in nx_g.edges:
            g.add_edge(u, v)

        # Test all singleton pairs with a few conditioning sets
        for x_node in nodes:
            for y_node in nodes:
                if x_node == y_node:
                    continue
                for z_set in [set(), {nodes[0]}, {nodes[3], nodes[5]}]:
                    # Skip if z overlaps with x or y
                    if x_node in z_set or y_node in z_set:
                        continue
                    ours = d_separated(g, {x_node}, {y_node}, z_set)
                    theirs = nx.is_d_separator(nx_g, {x_node}, {y_node}, z_set)
                    assert ours == theirs, (
                        f"Random DAG mismatch: X={{{x_node}}}, Y={{{y_node}}}, "
                        f"Z={z_set}: ours={ours}, nx={theirs}"
                    )
