"""Principal Component Analysis via the power iteration method.

This module implements a scikit-learn-style PCA class that uses the power
iteration algorithm to extract principal components from scratch, without
relying on eigenvalue decomposition routines from external libraries.

The power iteration method finds the dominant eigenvector of a matrix by
repeatedly multiplying a random vector by the matrix and normalizing.
To recover multiple components, we use *deflation*: after finding each
eigenvector, we subtract its contribution from the covariance matrix and
repeat.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA


class PowerIterationPCA:
    """PCA using the power iteration method (sklearn-compatible interface).

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to compute.
    max_iter : int, default=1000
        Maximum number of power iteration steps per component.
    tol : float, default=1e-6
        Convergence tolerance.  Iteration stops when the change in the
        eigenvector (measured by 1 - |dot product|) is below *tol*.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "PowerIterationPCA":
        """Compute principal components from *X* using power iteration.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix (will be deflated iteratively)
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        components = []
        eigenvalues = []

        for _ in range(self.n_components):
            eigenvector = self._power_iteration(cov, n_features, components)
            eigenvalue = float(eigenvector @ cov @ eigenvector)

            components.append(eigenvector)
            eigenvalues.append(eigenvalue)

            # Deflation: remove the contribution of this component
            cov = cov - eigenvalue * np.outer(eigenvector, eigenvector)

        self.components_ = np.array(components)  # shape (n_components, n_features)
        self.explained_variance_ = np.array(eigenvalues)
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* onto the principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    # --------------------------------------------------------- fit_transform
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and transform *X* in a single step.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    # ---------------------------------------------------- power iteration
    def _power_iteration(
        self,
        matrix: np.ndarray,
        n: int,
        prev_components: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Find the dominant eigenvector of *matrix* via power iteration.

        After each matrix-vector multiply, the candidate vector is
        re-orthogonalized against all previously found eigenvectors using
        Gram-Schmidt.  This ensures numerical orthogonality even when
        deflation introduces small errors.

        Parameters
        ----------
        matrix : ndarray of shape (n, n)
            Symmetric matrix (the possibly-deflated covariance matrix).
        n : int
            Dimensionality (number of features).
        prev_components : list of ndarray, optional
            Previously computed eigenvectors to orthogonalize against.

        Returns
        -------
        eigenvector : ndarray of shape (n,)
            Unit-norm dominant eigenvector.
        """
        if prev_components is None:
            prev_components = []

        rng = np.random.RandomState(42)
        v = rng.randn(n)
        # Remove any component along previously found eigenvectors
        v = self._orthogonalize(v, prev_components)
        v = v / np.linalg.norm(v)

        for _ in range(self.max_iter):
            v_new = matrix @ v

            # Re-orthogonalize against all previous components
            v_new = self._orthogonalize(v_new, prev_components)

            norm = np.linalg.norm(v_new)
            if norm < 1e-14:
                # Degenerate case: restart with a different random vector
                break
            v_new = v_new / norm

            # Check convergence: |v_new . v| ~ 1 means vectors are parallel
            if abs(np.dot(v_new, v)) > 1.0 - self.tol:
                v = v_new
                break
            v = v_new

        return v

    @staticmethod
    def _orthogonalize(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
        """Remove from *v* all projections onto vectors in *basis* (Gram-Schmidt)."""
        for u in basis:
            v = v - np.dot(v, u) * u
        return v


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_with_sklearn(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> dict:
    """Compare PowerIterationPCA results with scikit-learn's PCA.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_components : int
        Number of components to compute.
    max_iter : int
        Max iterations for power iteration.
    tol : float
        Convergence tolerance for power iteration.

    Returns
    -------
    result : dict
        Dictionary containing:
        - ``power_iteration_components`` : components from PowerIterationPCA
        - ``sklearn_components`` : components from sklearn PCA
        - ``component_cosine_similarity`` : list of cosine similarities per
          component (absolute value, so sign ambiguity is accounted for)
        - ``explained_variance_power_iteration`` : explained variances (ours)
        - ``explained_variance_sklearn`` : explained variances (sklearn)
    """
    X = np.asarray(X, dtype=np.float64)

    pca_pi = PowerIterationPCA(
        n_components=n_components, max_iter=max_iter, tol=tol
    )
    pca_pi.fit(X)

    pca_sk = SklearnPCA(n_components=n_components)
    pca_sk.fit(X)

    # Cosine similarity per component (absolute value to handle sign flips)
    cosine_sims = []
    for i in range(n_components):
        dot = np.dot(pca_pi.components_[i], pca_sk.components_[i])
        norm_pi = np.linalg.norm(pca_pi.components_[i])
        norm_sk = np.linalg.norm(pca_sk.components_[i])
        cosine_sims.append(float(abs(dot / (norm_pi * norm_sk))))

    return {
        "power_iteration_components": pca_pi.components_,
        "sklearn_components": pca_sk.components_,
        "component_cosine_similarity": cosine_sims,
        "explained_variance_power_iteration": pca_pi.explained_variance_,
        "explained_variance_sklearn": pca_sk.explained_variance_,
    }


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PCA via Power Iteration — Demo")
    print("=" * 60)

    # Generate synthetic data
    rng = np.random.RandomState(42)
    n_samples, n_features = 300, 5
    X = rng.randn(n_samples, n_features)
    X[:, 0] *= 10  # dominant direction
    X[:, 1] *= 5   # second direction

    n_components = 3

    # Fit our PCA
    pca = PowerIterationPCA(n_components=n_components, max_iter=2000)
    pca.fit(X)

    print(f"\nData shape: {X.shape}")
    print(f"Number of components: {n_components}")
    print(f"\nMean:\n  {pca.mean_}")
    print(f"\nComponents (rows = eigenvectors):\n{pca.components_}")
    print(f"\nExplained variance:\n  {pca.explained_variance_}")

    # Compare with sklearn
    print("\n" + "-" * 60)
    print("Comparison with sklearn PCA")
    print("-" * 60)
    result = compare_with_sklearn(X, n_components=n_components)

    print(f"\nsklearn components:\n{result['sklearn_components']}")
    print(f"\nCosine similarities (per component): "
          f"{result['component_cosine_similarity']}")
    print(f"\nExplained variance (power iteration): "
          f"{result['explained_variance_power_iteration']}")
    print(f"Explained variance (sklearn):         "
          f"{result['explained_variance_sklearn']}")

    # Transform and show first few rows
    X_t = pca.transform(X)
    print(f"\nTransformed data shape: {X_t.shape}")
    print(f"First 5 rows of transformed data:\n{X_t[:5]}")
