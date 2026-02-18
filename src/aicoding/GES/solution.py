"""Greedy Equivalence Search (GES) for causal discovery.

Implements the GES algorithm that searches over equivalence classes of DAGs
using a BIC scoring criterion, with v-structure detection and Meek rules
for proper edge orientation.
"""

import numpy as np


def bic_score(data, child, parents):
    """Compute the BIC score for a node given its parents.

    Uses linear regression: child ~ parents, with Gaussian likelihood.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
    child : int
        Index of the child variable.
    parents : list of int
        Indices of parent variables.

    Returns
    -------
    float
        BIC score (higher is better).
    """
    n = data.shape[0]
    y = data[:, child]

    if len(parents) == 0:
        residuals = y - np.mean(y)
        k = 1  # just the intercept
    else:
        X = data[:, parents]
        X = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        k = len(parents) + 1

    rss = np.sum(residuals ** 2)
    sigma2 = rss / n

    if sigma2 <= 0:
        sigma2 = 1e-10

    ll = -n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return ll - (k / 2) * np.log(n)


def _would_create_cycle(adj, i, j):
    """Check if adding edge i->j to adj would create a cycle."""
    p = adj.shape[0]
    visited = set()
    queue = [j]
    while queue:
        node = queue.pop(0)
        if node == i:
            return True
        if node in visited:
            continue
        visited.add(node)
        for k in range(p):
            if adj[node, k] != 0 and k not in visited:
                queue.append(k)
    return False


def _partial_correlation(data, i, j, conditioning):
    """Compute partial correlation between variables i and j given a set."""
    if not conditioning:
        return np.corrcoef(data[:, i], data[:, j])[0, 1]

    C = data[:, conditioning]
    C_aug = np.column_stack([np.ones(data.shape[0]), C])
    ri = data[:, i] - C_aug @ np.linalg.lstsq(C_aug, data[:, i], rcond=None)[0]
    rj = data[:, j] - C_aug @ np.linalg.lstsq(C_aug, data[:, j], rcond=None)[0]
    denom = np.sqrt(np.sum(ri ** 2) * np.sum(rj ** 2))
    if denom < 1e-15:
        return 0.0
    return np.sum(ri * rj) / denom


def _orient_edges(data, skeleton):
    """Orient edges using v-structure detection and Meek rules.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
    skeleton : ndarray of shape (p, p)
        Symmetric binary adjacency matrix.

    Returns
    -------
    ndarray of shape (p, p)
        Directed adjacency matrix (DAG).
    """
    p = skeleton.shape[0]
    n = data.shape[0]
    directed = np.zeros((p, p), dtype=int)
    undirected = skeleton.copy()

    # Step 1: V-structure detection
    # For each unshielded triple i-k-j, check whether any conditioning
    # set containing k separates i and j.  If none does, k is a collider.
    threshold = 2.0 / np.sqrt(n)
    for k in range(p):
        neighbors = list(np.where(skeleton[k] != 0)[0])
        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                i, j = neighbors[a_idx], neighbors[b_idx]
                if skeleton[i, j] != 0:
                    continue  # shielded triple

                # Check if k is in the separating set of i and j
                # by testing conditioning sets that include k.
                separated = False
                # Try {k} alone
                if abs(_partial_correlation(data, i, j, [k])) < threshold:
                    separated = True
                # Try {k} + neighbors of j \ {i, k}
                if not separated:
                    nbrs_j = [x for x in range(p) if skeleton[j, x] != 0
                              and x != i and x != k]
                    if nbrs_j:
                        cond = [k] + nbrs_j
                        if abs(_partial_correlation(data, i, j, cond)) < threshold:
                            separated = True
                # Try {k} + neighbors of i \ {j, k}
                if not separated:
                    nbrs_i = [x for x in range(p) if skeleton[i, x] != 0
                              and x != j and x != k]
                    if nbrs_i:
                        cond = [k] + nbrs_i
                        if abs(_partial_correlation(data, i, j, cond)) < threshold:
                            separated = True

                if not separated:
                    # Collider: i→k←j
                    if undirected[i, k] != 0:
                        directed[i, k] = 1
                        undirected[i, k] = 0
                        undirected[k, i] = 0
                    if undirected[j, k] != 0:
                        directed[j, k] = 1
                        undirected[j, k] = 0
                        undirected[k, j] = 0

    # Step 2: Meek rules
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if undirected[i, j] == 0:
                    continue
                # Rule 1: k→i-j, k not adjacent to j → orient i→j
                for k in range(p):
                    if directed[k, i] != 0 and skeleton[k, j] == 0 and k != j:
                        directed[i, j] = 1
                        undirected[i, j] = 0
                        undirected[j, i] = 0
                        changed = True
                        break
                if undirected[i, j] == 0:
                    continue
                # Rule 2: i→k→j and i-j → orient i→j
                for k in range(p):
                    if directed[i, k] != 0 and directed[k, j] != 0:
                        directed[i, j] = 1
                        undirected[i, j] = 0
                        undirected[j, i] = 0
                        changed = True
                        break

    # Step 3: Orient remaining using BIC, maintaining acyclicity
    for i in range(p):
        for j in range(i + 1, p):
            if undirected[i, j] == 0:
                continue
            parents_j = list(np.where(directed[:, j] != 0)[0])
            parents_i = list(np.where(directed[:, i] != 0)[0])
            score_ij = bic_score(data, child=j, parents=parents_j + [i])
            score_ji = bic_score(data, child=i, parents=parents_i + [j])

            if score_ij >= score_ji:
                if not _would_create_cycle(directed, i, j):
                    directed[i, j] = 1
                else:
                    directed[j, i] = 1
            else:
                if not _would_create_cycle(directed, j, i):
                    directed[j, i] = 1
                else:
                    directed[i, j] = 1
            undirected[i, j] = 0
            undirected[j, i] = 0

    return directed


def greedy_equivalence_search(data):
    """Run the Greedy Equivalence Search algorithm.

    Implements GES with forward (edge addition) and backward (edge removal)
    phases using BIC scoring, followed by v-structure detection and Meek
    rules for proper edge orientation.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)

    Returns
    -------
    ndarray of shape (n_variables, n_variables)
        Adjacency matrix. adj[i,j]=1 means edge i->j.
    """
    n, p = data.shape
    adj = np.zeros((p, p), dtype=int)

    # Forward phase: greedily add edges
    # For each pair, evaluate both directions to avoid floating-point bias
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_edge = None

        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] != 0 or adj[j, i] != 0:
                    continue

                # Try i→j
                if not _would_create_cycle(adj, i, j):
                    old_parents_j = list(np.where(adj[:, j] != 0)[0])
                    gain_ij = (
                        bic_score(data, child=j, parents=old_parents_j + [i])
                        - bic_score(data, child=j, parents=old_parents_j)
                    )
                else:
                    gain_ij = -np.inf

                # Try j→i
                if not _would_create_cycle(adj, j, i):
                    old_parents_i = list(np.where(adj[:, i] != 0)[0])
                    gain_ji = (
                        bic_score(data, child=i, parents=old_parents_i + [j])
                        - bic_score(data, child=i, parents=old_parents_i)
                    )
                else:
                    gain_ji = -np.inf

                # Pick the better direction; when gains are effectively
                # equal (floating-point symmetric), prefer i→j (i<j).
                if gain_ij >= gain_ji - 1e-6:
                    gain, edge = gain_ij, (i, j)
                else:
                    gain, edge = gain_ji, (j, i)

                if gain > best_gain:
                    best_gain = gain
                    best_edge = edge

        if best_edge is not None:
            adj[best_edge[0], best_edge[1]] = 1
            improved = True

    # Backward phase: greedily remove edges that improve BIC
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_edge = None

        for i in range(p):
            for j in range(p):
                if adj[i, j] == 0:
                    continue
                old_parents = list(np.where(adj[:, j] != 0)[0])
                old_score = bic_score(data, child=j, parents=old_parents)
                new_parents = [pa for pa in old_parents if pa != i]
                new_score = bic_score(data, child=j, parents=new_parents)
                gain = new_score - old_score
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge is not None:
            adj[best_edge[0], best_edge[1]] = 0
            improved = True

    # Post-processing: re-orient edges using v-structure detection + Meek rules
    skeleton = ((adj + adj.T) != 0).astype(int)
    np.fill_diagonal(skeleton, 0)
    adj = _orient_edges(data, skeleton)

    return adj


def generate_test_data(n_samples=2500, n_variables=4, edge_prob=0.4, seed=None):
    """Generate synthetic data from a random DAG for testing GES.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_variables : int
        Number of variables.
    edge_prob : float
        Probability of each edge in the DAG.
    seed : int, optional
        Random seed.

    Returns
    -------
    data : ndarray of shape (n_samples, n_variables)
    adj : ndarray of shape (n_variables, n_variables)
        True adjacency matrix.
    """
    rng = np.random.RandomState(seed)

    # Random lower-triangular adjacency (ensures DAG via topological order)
    adj = np.zeros((n_variables, n_variables), dtype=int)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if rng.rand() < edge_prob:
                adj[i, j] = 1

    # Generate data
    weights = adj.astype(float) * rng.uniform(0.5, 1.5, size=(n_variables, n_variables))
    weights *= rng.choice([-1, 1], size=(n_variables, n_variables))
    weights *= adj

    data = np.zeros((n_samples, n_variables))
    for j in range(n_variables):
        parents = np.where(adj[:, j] != 0)[0]
        noise = rng.randn(n_samples) * 0.5
        if len(parents) > 0:
            data[:, j] = data[:, parents] @ weights[parents, j] + noise
        else:
            data[:, j] = noise

    return data, adj


if __name__ == "__main__":
    print("Greedy Equivalence Search (GES)")
    print("=" * 50)

    # Use seed=6: has v-structures so directions are identifiable
    data, true_adj = generate_test_data(n_samples=2000, n_variables=5, seed=6)
    print("True adjacency matrix:")
    print(true_adj)

    estimated_adj = greedy_equivalence_search(data)
    print("\nEstimated adjacency matrix:")
    print(estimated_adj)

    true_edges = set(zip(*np.where(true_adj != 0)))
    est_edges = set(zip(*np.where(estimated_adj != 0)))
    true_undirected = {frozenset(e) for e in true_edges}
    est_undirected = {frozenset(e) for e in est_edges}
    overlap = true_undirected & est_undirected
    directed_match = true_edges & est_edges
    print(f"\nTrue edges: {len(true_edges)}")
    print(f"Estimated edges: {len(est_edges)}")
    print(f"Skeleton overlap: {len(overlap)}")
    print(f"Directed match: {len(directed_match)}")
