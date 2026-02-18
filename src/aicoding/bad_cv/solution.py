"""Demonstrate incorrect vs correct cross-validation with feature selection.

This module shows how performing feature selection (SelectKBest) on the
FULL dataset before cross-validation leaks information, inflating accuracy
estimates.  The correct approach embeds feature selection inside each CV
fold using a Pipeline, producing honest accuracy estimates.

Problem #9 (bad_cv) -- depends on mk_classification_data (#10).
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


def bad_crossvalidation(X, y, k=10, n_features_to_select=10, random_state=42):
    """Perform cross-validation with feature selection OUTSIDE the CV loop.

    This is the WRONG way: feature selection uses the full dataset (including
    the test fold), so noise features that are spuriously correlated with
    the target leak information into the model.  The resulting accuracy
    estimate is optimistically biased.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Binary target vector, shape (n_samples,).
        k: Number of cross-validation folds.
        n_features_to_select: Number of top features to keep.
        random_state: Random seed for KFold and the classifier.

    Returns:
        dict with keys:
            - 'mean_accuracy': float, mean accuracy across folds.
            - 'fold_accuracies': ndarray of per-fold accuracy scores.
    """
    # BAD: Select features on the FULL dataset before splitting into folds.
    selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
    X_selected = selector.fit_transform(X, y)

    # Now cross-validate on the already-filtered features.
    clf = LogisticRegression(
        solver="liblinear", random_state=random_state, max_iter=1000
    )
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_accuracies = cross_val_score(clf, X_selected, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "fold_accuracies": fold_accuracies,
    }


def good_crossvalidation(X, y, k=10, n_features_to_select=10, random_state=42):
    """Perform cross-validation with feature selection INSIDE the CV loop.

    This is the CORRECT way: a Pipeline ensures that feature selection is
    fit only on the training fold, preventing information leakage.  The
    resulting accuracy estimate is unbiased.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Binary target vector, shape (n_samples,).
        k: Number of cross-validation folds.
        n_features_to_select: Number of top features to keep.
        random_state: Random seed for KFold and the classifier.

    Returns:
        dict with keys:
            - 'mean_accuracy': float, mean accuracy across folds.
            - 'fold_accuracies': ndarray of per-fold accuracy scores.
    """
    # GOOD: Feature selection is embedded in a Pipeline, so it is fit
    # only on the training data within each CV fold.
    pipeline = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=n_features_to_select)),
        ("classifier", LogisticRegression(
            solver="liblinear", random_state=random_state, max_iter=1000
        )),
    ])
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_accuracies = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "fold_accuracies": fold_accuracies,
    }


if __name__ == "__main__":
    from aicoding.mk_classification_data.solution import make_classification_data

    print("=" * 65)
    print("Demonstration: Feature Selection Inside vs Outside CV Loop")
    print("=" * 65)
    print()

    X, y = make_classification_data(
        n_samples=200, n_features=1000, n_informative=10, random_state=42
    )
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features "
          f"(10 informative)")
    print()

    bad_result = bad_crossvalidation(X, y)
    good_result = good_crossvalidation(X, y)

    print("BAD CV  (feature selection OUTSIDE loop — data leakage):")
    print(f"  Mean accuracy: {bad_result['mean_accuracy']:.3f}")
    print(f"  Fold accuracies: {[f'{a:.3f}' for a in bad_result['fold_accuracies']]}")
    print()

    print("GOOD CV (feature selection INSIDE loop — no leakage):")
    print(f"  Mean accuracy: {good_result['mean_accuracy']:.3f}")
    print(f"  Fold accuracies: {[f'{a:.3f}' for a in good_result['fold_accuracies']]}")
    print()

    diff = bad_result["mean_accuracy"] - good_result["mean_accuracy"]
    print(f"Difference (bad - good): {diff:+.3f}")
    print()
    if diff > 0:
        print(">>> The bad CV inflates accuracy by "
              f"{diff * 100:.1f} percentage points.")
        print("    This illustrates the danger of performing feature selection")
        print("    on the full dataset before cross-validation.")
    else:
        print(">>> Unexpectedly, good CV was higher. This can happen with")
        print("    certain random seeds but is atypical for high-dimensional")
        print("    noise-dominated datasets.")
