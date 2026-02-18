# mk_classification_data

## Problem

Generate synthetic classification test data suitable for demonstrating incorrect cross-validation with feature selection performed outside the CV loop (problem #9, `bad_cv`).

## Approach

The module exposes a single function, `make_classification_data`, that wraps `sklearn.datasets.make_classification` with defaults tailored for the feature-selection leakage demonstration:

- **200 samples** (small enough that spurious correlations are likely)
- **1000 features** (high-dimensional, so many noise features can appear correlated with the target by chance)
- **10 informative features** (only a small fraction of features are truly predictive)
- **Binary classification** (two balanced classes)

This configuration creates a dataset where performing feature selection on the full dataset before cross-validation will select noise features that happen to correlate with the target, leading to inflated accuracy estimates. When feature selection is correctly placed inside the CV loop, accuracy drops to a more realistic level.

## Parameters

| Parameter       | Default | Description                                  |
|-----------------|---------|----------------------------------------------|
| `n_samples`     | 200     | Number of samples                            |
| `n_features`    | 1000    | Total number of features                     |
| `n_informative` | 10      | Number of truly predictive features          |
| `random_state`  | None    | Random seed for reproducibility              |

## Usage

```python
from aicoding.mk_classification_data.solution import make_classification_data

X, y = make_classification_data(random_state=42)
```

## Running

```bash
python -m aicoding.mk_classification_data.solution
```

## Testing

```bash
pytest tests/mk_classification_data/ -v
```
