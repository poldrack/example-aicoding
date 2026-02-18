# hurdle (#18)

Hurdle regression model with a scikit-learn style interface for zero-inflated count data.

## Approach

The hurdle model is a two-part model:

1. **Binary classifier** (logistic regression): predicts P(y > 0 | X) — whether an observation crosses the "hurdle".
2. **Count regressor** (Poisson regression): predicts E[y | y > 0, X] — the expected count given that it is positive, fit only on positive observations.

The final prediction combines both parts: `P(y > 0) * E[y | y > 0]`.

## Key design decisions

- Uses `BaseEstimator` and `RegressorMixin` for sklearn compatibility (`clone`, `get_params`, `score`).
- Handles the edge case where all observations are positive (no zeros) by creating synthetic data for the classifier.
- The Poisson regressor is fit only on the positive subset of the data.
- Regularization defaults are kept light (`alpha=1e-4`) to avoid excessive shrinkage.
