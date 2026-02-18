# PCA via Power Iteration

## Problem

Implement a scikit-learn-style class that performs Principal Component Analysis (PCA) using the power iteration method, coded from scratch without relying on eigenvalue decomposition routines. Provide a function to compare the results against scikit-learn's built-in PCA.

## Approach

### Power Iteration Algorithm

The power iteration method finds the dominant eigenvector of a matrix by repeatedly multiplying a random vector by the matrix and normalizing the result. After enough iterations the vector converges to the eigenvector associated with the largest eigenvalue.

To extract multiple principal components the algorithm uses **deflation**: after recovering each eigenvector, its contribution is subtracted from the covariance matrix and the process is repeated on the residual matrix.

A key implementation detail is **Gram-Schmidt re-orthogonalization**. Because deflation introduces small floating-point errors, each candidate vector is explicitly orthogonalized against all previously found eigenvectors at every iteration step. This guarantees that the resulting components are numerically orthogonal.

### Class Interface

`PowerIterationPCA` follows the scikit-learn estimator API:

- `__init__(n_components=2, max_iter=1000, tol=1e-6)` -- configure the model.
- `fit(X)` -- center the data, compute the covariance matrix, and extract components via power iteration with deflation.
- `transform(X)` -- project data onto the learned components.
- `fit_transform(X)` -- convenience method combining fit and transform.
- Attributes after fitting: `components_`, `explained_variance_`, `mean_`.

### Comparison Function

`compare_with_sklearn(X, n_components)` fits both `PowerIterationPCA` and `sklearn.decomposition.PCA`, then reports cosine similarities between corresponding components and the explained variances from each method.

## Key Implementation Details

- Only `numpy` is used for the core algorithm (no `np.linalg.eig` or `scipy.linalg`).
- The covariance matrix is computed with Bessel's correction (dividing by n-1).
- Convergence is checked by measuring how close `|v_new . v|` is to 1.
- Cosine similarities between power-iteration and sklearn components are consistently above 0.999 on test data.

## Running

```bash
# Run the demo
python -m aicoding.PCA_poweriteration.solution

# Run the tests
pytest tests/PCA_poweriteration/ -v
```
