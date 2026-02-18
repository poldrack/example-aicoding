# lmm

Combines CSV data files and fits a linear mixed model using statsmodels.

## Approach

1. **Combine** — Reads CSV files (rt, correct, compatible columns) from a directory, adds subject identifiers from filenames.
2. **Fit** — Uses `statsmodels.formula.api.mixedlm` to fit `rt ~ compatible` with random intercept and slope grouped by subject.
3. **Generate** — Includes synthetic data generator for testing with known effect sizes.

## Usage

```bash
python -m aicoding.lmm.solution
```
