"""Randomization test for hypothesis testing via permutation.

Generates bivariate correlated data, computes a Pearson correlation,
then builds a null distribution by shuffling one variable and
computes an empirical p-value.
"""

import numpy as np
from scipy import stats


def generate_correlated_data(n=100, r=0.1, seed=42):
    """Generate bivariate normal data with a specified correlation.

    Uses a Cholesky-based approach to produce two variables with the
    desired population correlation.

    Args:
        n: Sample size (number of observations).
        r: Target Pearson correlation between the two variables.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of shape (n, 2) with correlated columns.
    """
    rng = np.random.default_rng(seed)
    # Covariance matrix with unit variances and off-diagonal = r
    cov = np.array([[1.0, r], [r, 1.0]])
    mean = [0.0, 0.0]
    data = rng.multivariate_normal(mean, cov, size=n)
    return data


def compute_correlation(x, y):
    """Compute Pearson correlation coefficient and parametric p-value.

    Args:
        x: 1-D array of observations.
        y: 1-D array of observations (same length as x).

    Returns:
        Tuple of (r, p_value) where r is the Pearson correlation
        coefficient and p_value is the two-sided parametric p-value
        against H0: r = 0.
    """
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def permutation_test(x, y, n_permutations=5000, seed=42):
    """Build a null distribution by shuffling y and computing correlations.

    For each permutation, y is randomly shuffled and the Pearson
    correlation with x is computed. This produces a null distribution
    under the hypothesis of no association.

    Args:
        x: 1-D array of observations.
        y: 1-D array of observations (same length as x).
        n_permutations: Number of random shuffles to perform.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of length n_permutations containing null correlations.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    null_distribution = np.empty(n_permutations)

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        # Use np.corrcoef for speed (no p-value needed)
        null_distribution[i] = np.corrcoef(x, y_shuffled)[0, 1]

    return null_distribution


def compute_empirical_pvalue(observed_r, null_distribution):
    """Compute a two-sided empirical p-value from a null distribution.

    The empirical p-value is the proportion of null distribution values
    whose absolute value is greater than or equal to the absolute value
    of the observed correlation.

    Args:
        observed_r: The observed Pearson correlation coefficient.
        null_distribution: Array of correlation values from the null
            distribution (e.g., from permutation_test).

    Returns:
        Empirical p-value (float between 0 and 1).
    """
    null_distribution = np.asarray(null_distribution)
    count = np.sum(np.abs(null_distribution) >= np.abs(observed_r))
    return float(count / len(null_distribution))


if __name__ == "__main__":
    # Full simulation: generate data, compute correlation, run permutation test
    print("=== Randomization Test for Hypothesis Testing ===\n")

    # Step 1: Generate bivariate data with n=100, r=0.1
    data = generate_correlated_data(n=100, r=0.1, seed=42)
    x, y = data[:, 0], data[:, 1]
    print(f"Generated bivariate dataset: n={len(x)}, target r=0.1")

    # Step 2: Compute observed correlation and parametric p-value
    observed_r, parametric_p = compute_correlation(x, y)
    print(f"Observed correlation: r = {observed_r:.4f}")
    print(f"Parametric p-value:  p = {parametric_p:.4f}")

    # Step 3: Permutation test — shuffle y 5000 times
    null_dist = permutation_test(x, y, n_permutations=5000, seed=42)
    print(f"\nPermutation test: {len(null_dist)} shuffles")
    print(f"Null distribution: mean = {np.mean(null_dist):.4f}, "
          f"std = {np.std(null_dist):.4f}")

    # Step 4: Empirical p-value
    empirical_p = compute_empirical_pvalue(observed_r, null_dist)
    print(f"\nEmpirical p-value: p = {empirical_p:.4f}")
    print(f"Parametric p-value: p = {parametric_p:.4f}")
    print(f"\nConclusion: The empirical and parametric p-values are "
          f"{'consistent' if abs(empirical_p - parametric_p) < 0.1 else 'somewhat different'}, "
          f"demonstrating that the permutation approach approximates the "
          f"parametric test.")
