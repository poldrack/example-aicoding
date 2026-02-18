# Balanced Cross-Validation for Regression

## Problem

Standard K-Fold cross-validation partitions data sequentially or with a single random shuffle. When the target variable is skewed or has outliers, this can produce folds with very different distributions, leading to unstable or misleading performance estimates.

## Approach

`BalancedKFold` is a scikit-learn-compatible cross-validator that searches over many random candidate splits and selects the one where the target distributions across folds are most similar.

The balancing criterion is the one-way ANOVA F-statistic (`scipy.stats.f_oneway`) computed on the target values of the test folds. A lower F-statistic means the fold means are more similar relative to within-fold variance. The algorithm:

1. Generate `n_candidates` random shuffled K-Fold partitions.
2. For each candidate, compute the F-statistic of the target across the K test folds.
3. Select the candidate with the smallest F-statistic.
4. Yield (train, test) index pairs for the winning split.

## Interface

```python
from aicoding.balanced_cv.solution import BalancedKFold

cv = BalancedKFold(n_splits=5, n_candidates=100, seed=42)

# Use directly
for train_idx, test_idx in cv.split(X, y):
    ...

# Or with sklearn utilities
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv)
```

## Key Design Decisions

- **Random search over candidates** rather than a deterministic sorting/stratification approach. This is simple, general, and effective for continuous targets where binning would be arbitrary.
- **F-statistic as the objective** because it directly measures whether fold means differ significantly, which is the distributional property most relevant to regression performance estimates.
- **Sorted test indices** within each fold for consistency and easier debugging.

## Running

```bash
python -m aicoding.balanced_cv.solution
```

This prints a comparison between `BalancedKFold` and standard `KFold`, showing fold-level statistics and the F-statistic reduction.
