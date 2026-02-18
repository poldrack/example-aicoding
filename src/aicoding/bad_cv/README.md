# bad_cv -- Incorrect vs Correct Cross-Validation with Feature Selection

## Problem

When performing cross-validation on a high-dimensional dataset, a common
mistake is to apply feature selection on the **full dataset** before
splitting into folds.  This leaks information from the test fold into the
training process because the feature selector has already "seen" the test
data.  The result is an inflated (optimistically biased) accuracy estimate.

## Approach

This module provides two functions that contrast the wrong and right ways
to combine feature selection with cross-validation:

- **`bad_crossvalidation(X, y, k=10)`** -- Applies `SelectKBest(f_classif)`
  to the entire dataset first, then runs k-fold CV on the reduced feature
  set.  Because the selector was fit on all samples (including future test
  folds), noise features that happen to correlate with the target by chance
  are retained, inflating accuracy.

- **`good_crossvalidation(X, y, k=10)`** -- Uses a scikit-learn `Pipeline`
  that embeds `SelectKBest` inside the CV loop.  Each fold fits the
  selector only on its training split, so no information leaks from the
  test split.  This gives an honest accuracy estimate.

Both functions use `LogisticRegression` (liblinear solver) as the
classifier and return a dictionary with `mean_accuracy` and per-fold
`fold_accuracies`.

## Dataset

The module is designed to work with data from `mk_classification_data`:
200 samples, 1000 features, only 10 informative.  The high ratio of noise
features to samples makes the leakage effect clearly visible.

## Key Result

On the default dataset (random_state=42), bad CV reports ~81.5% accuracy
while good CV reports ~75.0% -- a ~6.5 percentage point inflation caused
entirely by performing feature selection outside the CV loop.
