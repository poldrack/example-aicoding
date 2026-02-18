"""Generate synthetic classification data for cross-validation experiments.

Creates a dataset with many features (mostly noise) and a binary target,
suitable for demonstrating the pitfalls of performing feature selection
outside the cross-validation loop (problem #9, bad_cv).

When feature selection is applied to the full dataset before CV, noise
features that are spuriously correlated with the target leak information,
inflating accuracy estimates.
"""

import numpy as np
from sklearn.datasets import make_classification as _sklearn_make_classification


def make_classification_data(
    n_samples=200,
    n_features=1000,
    n_informative=10,
    random_state=None,
):
    """Generate a synthetic binary classification dataset.

    Produces a feature matrix X and binary target vector y with a small
    number of informative features and many noise features.  This setup
    is ideal for illustrating how feature selection outside a
    cross-validation loop leads to overfitting.

    Args:
        n_samples: Number of samples (rows) to generate.
        n_features: Total number of features (columns).
        n_informative: Number of features that are truly predictive
            of the target.  Must be <= n_features.
        random_state: Seed for the random number generator.  Pass an
            integer for reproducible output.

    Returns:
        Tuple of (X, y) where X is an ndarray of shape
        (n_samples, n_features) and y is an ndarray of shape
        (n_samples,) with values in {0, 1}.
    """
    # Ensure n_informative does not exceed n_features
    n_informative = min(n_informative, n_features)

    # sklearn requires: 2 * n_clusters_per_class <= 2**n_informative.
    # With default n_clusters_per_class=2 this needs n_informative >= 2.
    # Reduce clusters_per_class when n_informative is very small.
    n_clusters_per_class = min(2, 2 ** n_informative // 2) or 1

    # sklearn's make_classification requires n_informative + n_redundant +
    # n_repeated <= n_features.  We set redundant and repeated to 0 so that
    # the remaining features are pure noise, which is the clearest setup
    # for demonstrating feature-selection leakage.
    X, y = _sklearn_make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=n_clusters_per_class,
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state,
    )

    return X, y


if __name__ == "__main__":
    X, y = make_classification_data(random_state=42)
    print(f"Generated classification dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  n_informative: 10 (of {X.shape[1]} total features)")
    print()

    # Show that most features have low correlation with target
    correlations = np.array(
        [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
    )
    print(f"  Mean |correlation| with y: {correlations.mean():.4f}")
    print(f"  Max  |correlation| with y: {correlations.max():.4f}")
    print(f"  Features with |r| > 0.2: {np.sum(correlations > 0.2)}")
    print(f"  Features with |r| < 0.1: {np.sum(correlations < 0.1)}")
