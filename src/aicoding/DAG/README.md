# DAG -- Directed Graph and D-Separation

## Problem

Implement a Python class to represent a directed graph and a function that tests
whether two sets of vertices are d-separated given a conditioning set.  Validate
the results against the NetworkX library's `is_d_separator` function.

## Approach

### DirectedGraph class

A lightweight directed graph stored as two adjacency dictionaries (`_children`
and `_parents`).  Supports `add_node`, `add_edge`, `get_nodes`, `get_children`,
`get_parents`, and `get_edges`.

### D-separation algorithm

The `d_separated(graph, set_x, set_y, set_z)` function uses the **ancestral
graph moralization** method:

1. **Ancestors** -- collect all ancestors of X, Y, and Z.
2. **Ancestral sub-graph** -- restrict to those ancestors.
3. **Moralize** -- connect co-parents that share a child (marry parents), then
   drop edge directions to produce an undirected graph.
4. **Remove Z** -- delete the conditioning nodes.
5. **Reachability** -- BFS from X; if any node in Y is reached, the sets are
   *not* d-separated.

This is equivalent to the standard textbook definition and produces results
consistent with `networkx.is_d_separator`.

## Validation

The test suite includes:

- Unit tests on the three canonical structures (chain, fork, collider).
- Tests on a six-node complex DAG.
- Edge cases (empty graph, single node, self-loop, overlapping sets).
- Parametrized comparison tests against `networkx.is_d_separator` on four
  different graph topologies.
- An exhaustive random-DAG test that generates a random 8-node DAG and checks
  all singleton-pair queries with multiple conditioning sets against NetworkX.

## Running

```bash
# Run tests
pytest tests/DAG/ -v

# Run the demo script
python -m aicoding.DAG.solution
```
