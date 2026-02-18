"""PC causal inference algorithm implemented from scratch.

This module provides:
- ``partial_correlation`` -- partial correlation test using linear regression
  residuals and scipy.stats for p-values.
- ``PCAlgorithm`` -- the constraint-based PC algorithm for causal skeleton
  discovery.
- ``generate_synthetic_data`` -- helper to create data from a known DAG for
  testing purposes.

The PC algorithm starts with a complete undirected graph and removes edges
between pairs of variables that are conditionally independent given some
subset of their neighbours, using partial correlation as the independence
test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

def partial_correlation(
    X: np.ndarray,
    i: int,
    j: int,
    conditioning_set: list[int],
) -> tuple[float, float]:
    """Compute the partial correlation between variables *i* and *j*
    conditioned on a set of other variables.

    The partial correlation is obtained by regressing variables *i* and *j*
    on the conditioning set, computing the Pearson correlation of the
    residuals, and deriving a two-sided p-value via the Fisher z-transform.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape ``(n, p)`` where *n* is the number of
        observations and *p* is the number of variables.
    i, j : int
        Column indices of the two variables of interest.
    conditioning_set : list[int]
        Column indices of the variables to condition on.  May be empty.

    Returns
    -------
    r : float
        Partial correlation coefficient in ``[-1, 1]``.
    p : float
        Two-sided p-value for the null hypothesis that the partial
        correlation is zero.
    """
    n = X.shape[0]

    if len(conditioning_set) == 0:
        # Simple Pearson correlation
        r, p = stats.pearsonr(X[:, i], X[:, j])
        return float(r), float(p)

    # Build the conditioning matrix and add an intercept column
    Z = X[:, conditioning_set]
    Z_design = np.column_stack([np.ones(n), Z])

    # Regress X_i on Z and compute residuals
    beta_i, _, _, _ = np.linalg.lstsq(Z_design, X[:, i], rcond=None)
    residuals_i = X[:, i] - Z_design @ beta_i

    # Regress X_j on Z and compute residuals
    beta_j, _, _, _ = np.linalg.lstsq(Z_design, X[:, j], rcond=None)
    residuals_j = X[:, j] - Z_design @ beta_j

    # Pearson correlation of residuals
    r, _ = stats.pearsonr(residuals_i, residuals_j)
    r = float(r)

    # Clamp r to (-1, 1) to avoid numerical issues with arctanh
    r_clamped = max(min(r, 1.0 - 1e-15), -1.0 + 1e-15)

    # Fisher z-transform for the p-value
    k = len(conditioning_set)
    z = np.sqrt(n - k - 3) * np.arctanh(r_clamped)
    p = float(2.0 * stats.norm.sf(abs(z)))

    return r, p


# ---------------------------------------------------------------------------
# PC Algorithm
# ---------------------------------------------------------------------------

class PCAlgorithm:
    """Constraint-based PC algorithm for causal skeleton discovery.

    Parameters
    ----------
    alpha : float, optional
        Significance level for the conditional independence tests.
        Default is ``0.05``.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.skeleton: set[frozenset[int]] = set()
        self.separation_sets: dict[frozenset[int], list[int]] = {}
        self.num_nodes: int = 0

    # ---- helpers ----------------------------------------------------------

    def _neighbours(self, node: int) -> list[int]:
        """Return the list of current neighbours of *node* in the skeleton."""
        nbrs: list[int] = []
        for edge in self.skeleton:
            if node in edge:
                other = next(iter(edge - frozenset({node})))
                nbrs.append(other)
        return sorted(nbrs)

    # ---- main algorithm ---------------------------------------------------

    def fit(self, data: np.ndarray) -> "PCAlgorithm":
        """Run the PC algorithm on *data* and discover the causal skeleton.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape ``(n, p)``.

        Returns
        -------
        self
        """
        n, p = data.shape
        self.num_nodes = p

        if p < 2:
            self.skeleton = set()
            return self

        # Start with a complete undirected graph
        self.skeleton = {
            frozenset({i, j}) for i in range(p) for j in range(i + 1, p)
        }
        self.separation_sets = {}

        # Increase the conditioning-set size from 0 up to p-2
        depth = 0
        while depth <= p - 2:
            edges_to_test = list(self.skeleton)
            removed_any = False

            for edge in edges_to_test:
                if edge not in self.skeleton:
                    # Edge was already removed in this pass
                    continue

                i, j = tuple(edge)
                # Neighbours of i excluding j (and vice versa)
                adj_i = [nb for nb in self._neighbours(i) if nb != j]
                adj_j = [nb for nb in self._neighbours(j) if nb != i]

                # Only test if a node has enough neighbours for this depth
                tested = False
                for adj_set in (adj_i, adj_j):
                    if len(adj_set) >= depth:
                        for S in combinations(adj_set, depth):
                            S_list = list(S)
                            r, p_val = partial_correlation(data, i, j, S_list)
                            if p_val > self.alpha:
                                # Conditionally independent -> remove edge
                                self.skeleton.discard(edge)
                                self.separation_sets[edge] = S_list
                                removed_any = True
                                tested = True
                                break
                    if tested and edge not in self.skeleton:
                        break

            # If no neighbour set of the current depth exists for any
            # remaining edge, we can stop early.
            max_nbr = 0
            for edge in self.skeleton:
                i, j = tuple(edge)
                adj_i = [nb for nb in self._neighbours(i) if nb != j]
                adj_j = [nb for nb in self._neighbours(j) if nb != i]
                max_nbr = max(max_nbr, len(adj_i), len(adj_j))
            if depth >= max_nbr:
                break

            depth += 1

        return self


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 500, seed: int = 42) -> np.ndarray:
    """Generate synthetic data from a known causal DAG.

    The DAG structure is::

        X0 -----> X2 <----- X1
                   |
                   v
                  X3

    That is:
    - ``X0 -> X2`` (direct effect)
    - ``X1 -> X2`` (direct effect)
    - ``X2 -> X3`` (direct effect)

    ``X0`` and ``X1`` are independent root nodes.

    Parameters
    ----------
    n : int, optional
        Number of observations. Default is ``500``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.

    Returns
    -------
    data : np.ndarray
        Array of shape ``(n, 4)`` with columns ``[X0, X1, X2, X3]``.
    """
    rng = np.random.default_rng(seed)

    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = 0.8 * x0 + 0.8 * x1 + rng.standard_normal(n) * 0.5
    x3 = 1.0 * x2 + rng.standard_normal(n) * 0.5

    return np.column_stack([x0, x1, x2, x3])


# ---------------------------------------------------------------------------
# __main__ demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PC Algorithm Demo ===\n")

    # Generate synthetic data
    data = generate_synthetic_data(n=1000, seed=42)
    print(f"Generated synthetic data: {data.shape[0]} observations, "
          f"{data.shape[1]} variables\n")

    print("Known DAG structure:")
    print("  X0 -> X2")
    print("  X1 -> X2")
    print("  X2 -> X3")
    print("  (X0 and X1 are independent root nodes)\n")

    # Run PC algorithm
    pc = PCAlgorithm(alpha=0.05)
    pc.fit(data)

    print("Discovered skeleton (undirected edges):")
    for edge in sorted(pc.skeleton, key=lambda e: tuple(sorted(e))):
        nodes = sorted(edge)
        print(f"  X{nodes[0]} --- X{nodes[1]}")

    print(f"\nTotal edges in skeleton: {len(pc.skeleton)}")
    print(f"Separation sets: {pc.separation_sets}")

    # Show some partial correlations
    print("\nPartial correlations:")
    for i, j, cond in [(0, 1, []), (0, 2, []), (0, 1, [2]), (0, 3, [2])]:
        r, p = partial_correlation(data, i, j, cond)
        cond_str = f"| {cond}" if cond else ""
        print(f"  r(X{i}, X{j} {cond_str}) = {r:+.4f}  (p = {p:.4e})")
