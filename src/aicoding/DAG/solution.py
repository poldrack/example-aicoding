"""Directed graph class and d-separation algorithm.

Provides a ``DirectedGraph`` class for building directed acyclic graphs
and a ``d_separated`` function that decides whether two sets of vertices
are d-separated given a conditioning set.  The implementation uses the
*ancestral graph* method:

1. Collect all ancestors of X, Y, and Z.
2. Build the induced ancestral sub-graph.
3. *Moralize* it (marry co-parents, drop directions).
4. Remove the conditioning set Z.
5. X and Y are d-separated iff they are disconnected in the result.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Set


class DirectedGraph:
    """A simple directed graph (supports cycles, but d-separation assumes a DAG)."""

    def __init__(self) -> None:
        self._children: dict[object, set] = {}  # node -> set of children
        self._parents: dict[object, set] = {}   # node -> set of parents

    # ---- mutation --------------------------------------------------------

    def add_node(self, node: object) -> None:
        """Add a node to the graph (idempotent)."""
        if node not in self._children:
            self._children[node] = set()
            self._parents[node] = set()

    def add_edge(self, parent: object, child: object) -> None:
        """Add a directed edge *parent* -> *child*.

        Nodes are created automatically if they do not already exist.
        Adding the same edge twice is idempotent.
        """
        self.add_node(parent)
        self.add_node(child)
        self._children[parent].add(child)
        self._parents[child].add(parent)

    # ---- queries ---------------------------------------------------------

    def get_nodes(self) -> list:
        """Return a list of all nodes in the graph."""
        return list(self._children.keys())

    def get_children(self, node: object) -> list:
        """Return a list of direct children (successors) of *node*."""
        return list(self._children.get(node, set()))

    def get_parents(self, node: object) -> list:
        """Return a list of direct parents (predecessors) of *node*."""
        return list(self._parents.get(node, set()))

    def get_edges(self) -> list[tuple]:
        """Return all directed edges as ``(parent, child)`` tuples."""
        edges = []
        for parent, children in self._children.items():
            for child in children:
                edges.append((parent, child))
        return edges


# ---------------------------------------------------------------------------
# D-separation via the ancestral-graph method
# ---------------------------------------------------------------------------

def _ancestors(graph: DirectedGraph, nodes: Set) -> Set:
    """Return *nodes* together with all of their ancestors."""
    visited: set = set()
    queue = deque(nodes)
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for parent in graph.get_parents(node):
            if parent not in visited:
                queue.append(parent)
    return visited


def d_separated(
    graph: DirectedGraph,
    set_x: Set,
    set_y: Set,
    set_z: Set,
) -> bool:
    """Decide whether *set_x* and *set_y* are d-separated given *set_z*.

    Uses the ancestral-graph / moralization algorithm:

    1. Collect all ancestors of ``set_x | set_y | set_z``.
    2. Build the induced ancestral sub-graph.
    3. Moralize (connect co-parents sharing a child, then drop directions).
    4. Remove the nodes in ``set_z``.
    5. Return ``True`` iff no node in ``set_x`` is reachable from any
       node in ``set_y`` in the resulting undirected graph.

    Parameters
    ----------
    graph : DirectedGraph
        The directed (acyclic) graph.
    set_x, set_y : set
        Two disjoint sets of vertices to test.
    set_z : set
        The conditioning set.

    Returns
    -------
    bool
        ``True`` if *set_x* and *set_y* are d-separated given *set_z*.
    """
    # Trivial cases
    if not set_x or not set_y:
        return True

    # If X and Y overlap, they are trivially *not* d-separated
    if set_x & set_y:
        return False

    # Step 1: ancestors of X ∪ Y ∪ Z
    all_relevant = set_x | set_y | set_z
    ancestor_set = _ancestors(graph, all_relevant)

    # Step 2 + 3: build the moralized undirected adjacency on the
    # ancestral sub-graph
    undirected: dict[object, set] = {n: set() for n in ancestor_set}

    for node in ancestor_set:
        # Add undirected versions of existing directed edges
        for child in graph.get_children(node):
            if child in ancestor_set:
                undirected[node].add(child)
                undirected[child].add(node)

        # Moralize: marry co-parents that share a child in ancestor_set
        for child in graph.get_children(node):
            if child in ancestor_set:
                for other_parent in graph.get_parents(child):
                    if other_parent in ancestor_set and other_parent != node:
                        undirected[node].add(other_parent)
                        undirected[other_parent].add(node)

    # Step 4: remove conditioning set Z
    for z_node in set_z:
        if z_node in undirected:
            # Remove z_node from all neighbor lists, then delete it
            for neighbor in undirected[z_node]:
                undirected[neighbor].discard(z_node)
            del undirected[z_node]

    # Step 5: BFS from set_x; check if any node in set_y is reachable
    visited: set = set()
    queue = deque(n for n in set_x if n in undirected)
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node in set_y:
            return False  # reachable -> not d-separated
        for neighbor in undirected.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    return True


# ---------------------------------------------------------------------------
# __main__ demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    # Build a sample DAG:  A -> C <- B,  C -> D,  C -> E,  D -> F <- E
    g = DirectedGraph()
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "F"), ("E", "F")]
    for u, v in edges:
        g.add_edge(u, v)

    # Also build the NetworkX equivalent for comparison
    nx_g = nx.DiGraph(edges)

    queries = [
        ({"A"}, {"B"}, set()),
        ({"A"}, {"B"}, {"C"}),
        ({"A"}, {"B"}, {"D"}),
        ({"D"}, {"E"}, {"C"}),
        ({"A"}, {"F"}, set()),
        ({"A"}, {"F"}, {"C"}),
    ]

    print("D-separation results (ours vs NetworkX):")
    print(f"{'X':<8} {'Y':<8} {'Z':<12} {'Ours':<8} {'NX':<8} {'Match'}")
    print("-" * 56)
    for x, y, z in queries:
        ours = d_separated(g, x, y, z)
        theirs = nx.is_d_separator(nx_g, x, y, z)
        match = "OK" if ours == theirs else "MISMATCH"
        print(f"{str(x):<8} {str(y):<8} {str(z):<12} {str(ours):<8} {str(theirs):<8} {match}")
