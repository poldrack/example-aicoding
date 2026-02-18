# randomization

Demonstrates randomization (permutation) testing to obtain a null distribution for hypothesis testing.

## Approach

1. **Generate** -- Creates bivariate normal data with a specified correlation using a Cholesky-based multivariate normal draw (`numpy.random.Generator.multivariate_normal`).
2. **Correlate** -- Computes the Pearson correlation coefficient and parametric p-value via `scipy.stats.pearsonr`.
3. **Permute** -- Shuffles one variable 5000 times, computing the correlation each time to build a null distribution under H0: r = 0.
4. **Empirical p-value** -- Computes the proportion of null distribution values with |r| >= |observed r|.

## Usage

```bash
python -m aicoding.randomization.solution
```

## Testing

```bash
pytest tests/randomization/test_solution.py -v
```
