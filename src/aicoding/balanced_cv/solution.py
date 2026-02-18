"""Balanced cross-validation for regression.

Provides a ``BalancedKFold`` class that follows the scikit-learn
cross-validator interface.  Among many random candidate splits it
selects the one whose fold-wise target distributions are most similar,
as measured by a one-way ANOVA F-statistic (lower is more balanced).
"""

import numpy as np
from scipy.stats import f_oneway
from sklearn.model_selection import KFold


class BalancedKFold:
    """K-Fold cross-validator that minimises distributional imbalance.

    For a given dataset the class generates ``n_candidates`` random
    shuffled K-Fold splits, computes the one-way ANOVA F-statistic of
    the target variable across the test folds of each candidate, and
    returns the candidate with the smallest F-statistic.

    This ensures that the folds have maximally similar distributions
    of the target variable, which is useful when the target is skewed
    or has outliers that could make individual folds unrepresentative.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    n_candidates : int, default=100
        Number of random split candidates to evaluate.
    seed : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from aicoding.balanced_cv.solution import BalancedKFold
    >>> X = np.random.randn(100, 3)
    >>> y = np.random.randn(100)
    >>> cv = BalancedKFold(n_splits=5, n_candidates=50, seed=0)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(len(train_idx), len(test_idx))
    80 20
    80 20
    80 20
    80 20
    80 20
    """

    def __init__(self, n_splits=5, n_candidates=100, seed=42):
        self.n_splits = n_splits
        self.n_candidates = n_candidates
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : ignored
        y : ignored
        groups : ignored

        Returns
        -------
        int
            The number of folds.
        """
        return self.n_splits

    def split(self, X, y, groups=None):
        """Generate train/test index sets for balanced K-Fold splits.

        Among ``n_candidates`` random shuffled K-Fold partitions, this
        method picks the one whose test-fold target distributions have
        the lowest one-way ANOVA F-statistic, then yields (train, test)
        index arrays for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  Only its length is used.
        y : array-like of shape (n_samples,)
            Target values whose distribution should be balanced.
        groups : ignored
            Present for API compatibility with sklearn splitters.

        Yields
        ------
        train_indices : ndarray
            Indices for the training set of this fold.
        test_indices : ndarray
            Indices for the test set of this fold.
        """
        y = np.asarray(y)
        n_samples = len(y)

        rng = np.random.default_rng(self.seed)

        best_f = np.inf
        best_splits = None

        for _ in range(self.n_candidates):
            # Generate a shuffled permutation and partition into folds
            perm = rng.permutation(n_samples)
            fold_indices = np.array_split(perm, self.n_splits)

            # Compute the F-statistic across fold target distributions
            fold_values = [y[idx] for idx in fold_indices]

            # f_oneway needs at least 2 groups with >= 1 observation each
            if all(len(fv) >= 1 for fv in fold_values) and len(fold_values) >= 2:
                f_stat, _ = f_oneway(*fold_values)
                # Handle NaN (can occur if all values in a fold are identical)
                if np.isnan(f_stat):
                    f_stat = 0.0
            else:
                f_stat = np.inf

            if f_stat < best_f:
                best_f = f_stat
                best_splits = fold_indices

        # Yield (train, test) pairs from the best candidate
        all_indices = np.arange(n_samples)
        for test_idx in best_splits:
            test_idx_sorted = np.sort(test_idx)
            train_idx = np.setdiff1d(all_indices, test_idx_sorted)
            yield train_idx, test_idx_sorted


if __name__ == "__main__":
    from sklearn.model_selection import KFold as SklearnKFold, cross_val_score
    from sklearn.linear_model import Ridge
    from scipy.stats import f_oneway as f_test

    print("=" * 65)
    print("  Balanced K-Fold vs Standard K-Fold  —  Comparison Demo")
    print("=" * 65)

    # ----- Generate a synthetic regression dataset with skewed target -----
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.standard_normal((n, p))
    # Skewed target to make balancing non-trivial
    y = np.exp(X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.3)

    print(f"\nDataset: {n} samples, {p} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], "
          f"mean={y.mean():.2f}, std={y.std():.2f}\n")

    # ----- Balanced K-Fold -----
    n_splits = 5
    cv_balanced = BalancedKFold(n_splits=n_splits, n_candidates=200, seed=42)
    balanced_splits = list(cv_balanced.split(X, y))

    balanced_fold_values = [y[te] for _, te in balanced_splits]
    f_balanced, p_balanced = f_test(*balanced_fold_values)

    print("Balanced K-Fold")
    print("-" * 40)
    for i, (_, te) in enumerate(balanced_splits):
        vals = y[te]
        print(f"  Fold {i}: n={len(te):3d}  mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}")
    print(f"  F-statistic = {f_balanced:.4f}  (p = {p_balanced:.4f})")

    # ----- Standard K-Fold (shuffled) -----
    cv_standard = SklearnKFold(n_splits=n_splits, shuffle=True, random_state=0)
    standard_splits = list(cv_standard.split(X, y))

    standard_fold_values = [y[te] for _, te in standard_splits]
    f_standard, p_standard = f_test(*standard_fold_values)

    print(f"\nStandard K-Fold (shuffle=True, random_state=0)")
    print("-" * 40)
    for i, (_, te) in enumerate(standard_splits):
        vals = y[te]
        print(f"  Fold {i}: n={len(te):3d}  mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}")
    print(f"  F-statistic = {f_standard:.4f}  (p = {p_standard:.4f})")

    # ----- cross_val_score comparison -----
    model = Ridge(alpha=1.0)

    scores_balanced = cross_val_score(
        model, X, y, cv=cv_balanced, scoring="neg_mean_squared_error"
    )
    scores_standard = cross_val_score(
        model, X, y, cv=cv_standard, scoring="neg_mean_squared_error"
    )

    print(f"\ncross_val_score (neg MSE) — Ridge(alpha=1)")
    print("-" * 40)
    print(f"  Balanced:  {scores_balanced}  (mean={scores_balanced.mean():.4f})")
    print(f"  Standard:  {scores_standard}  (mean={scores_standard.mean():.4f})")
    print(f"\nF-stat reduction: "
          f"{f_standard:.4f} -> {f_balanced:.4f}  "
          f"({(1 - f_balanced / f_standard) * 100:.1f}% lower)")
