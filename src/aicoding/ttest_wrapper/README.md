# ttest_wrapper

A Python wrapper around `statsmodels.stats.weightstats.ttest_ind` that returns
results formatted like R's `t.test` function.

## Approach

The core t-statistic, p-value, and degrees of freedom are computed by
statsmodels.  On top of that, this module:

1. Computes the 95% confidence interval for the difference in means using the
   appropriate standard error (pooled or Welch) and the t-distribution critical
   value from scipy.
2. Stores all results in a `TTestResult` dataclass-like object with a `__str__`
   method that formats output to match R's console report.

## API

```python
from aicoding.ttest_wrapper.solution import ttest_ind

result = ttest_ind(x1, x2, alternative="two-sided", usevar="pooled")
print(result)  # R-style formatted output
```

### Parameters

| Parameter     | Description                                              | Default        |
|---------------|----------------------------------------------------------|----------------|
| `x1`          | First sample (list, tuple, or numpy array)               | required       |
| `x2`          | Second sample                                            | required       |
| `alternative` | `"two-sided"`, `"larger"`, or `"smaller"`                | `"two-sided"`  |
| `usevar`      | `"pooled"` (equal variance) or `"unequal"` (Welch)      | `"pooled"`     |

### Return value

A `TTestResult` object with attributes:

- `t_statistic` -- the t test statistic
- `p_value` -- the p-value
- `df` -- degrees of freedom
- `conf_int` -- 95% confidence interval `(lower, upper)` for the mean difference
- `mean_x1`, `mean_x2` -- sample means

## Example output

```
	Welch Two Sample t-test

data:  x and y
t = -1.8974, df = 5.882, p-value = 0.1075
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -6.887742 0.887742
sample estimates:
mean of x mean of y
        3         6
```

## Dependencies

- `statsmodels` (core t-test computation)
- `scipy` (critical values for the confidence interval)
- `numpy` (array handling)
