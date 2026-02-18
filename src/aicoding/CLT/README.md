# CLT -- Central Limit Theorem Simulation

## Problem

Simulate the Central Limit Theorem by repeatedly drawing samples from 6 different distributions, computing sample means, and visualizing how the sampling distribution of the mean becomes approximately normal regardless of the underlying distribution.

## Distributions

| Distribution | numpy call | Parameters |
|---|---|---|
| Normal | `standard_normal` | mean=0, std=1 |
| Uniform | `uniform(0, 1)` | a=0, b=1 |
| Chi-squared | `chisquare(df=2)` | df=2 |
| Poisson | `poisson(lam=3)` | lambda=3 |
| Exponential | `exponential(scale=1)` | scale=1 |
| Beta | `beta(a=2, b=5)` | a=2, b=5 |

## Approach

1. `simulate_sample_means(distribution, ...)` draws `n_samples` independent samples of size `sample_size` from a given distribution and returns the array of sample means.
2. `run_clt_simulation(...)` orchestrates the simulation across all 6 distributions, using deterministic sub-seeds from a base seed so each distribution gets an independent random stream.
3. `plot_clt(results)` creates a 2x3 grid of histograms (one per distribution) showing the density of sample means, with a vertical line at the grand mean.

The `__main__` block runs the full simulation, prints summary statistics, saves the figure to `clt_simulation.png`, and displays it.

## Usage

```bash
python -m aicoding.CLT.solution
```

## Tests

```bash
pytest tests/CLT/ -v
```

28 tests covering return types, array shapes, reproducibility, statistical sanity checks (mean values, positivity, range), edge cases (invalid distribution name), and plot structure (figure type, number of axes, subplot titles).
