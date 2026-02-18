# PC Causal Inference Algorithm

## Overview

This module implements the PC (Peter-Clark) algorithm for causal skeleton
discovery from observational data.  The algorithm is coded from scratch
without relying on any existing causal inference library.

## Components

| Function / Class | Description |
|---|---|
| `partial_correlation(X, i, j, conditioning_set)` | Computes the partial correlation between variables `i` and `j` given a conditioning set, using OLS residuals and the Fisher z-transform for p-values (via `scipy.stats`). |
| `PCAlgorithm(alpha=0.05)` | The main class.  Call `.fit(data)` to discover the causal skeleton. |
| `generate_synthetic_data(n, seed)` | Generates data from the DAG: X0 -> X2 <- X1, X2 -> X3. |

## Algorithm

1. Start with a complete undirected graph over all variables.
2. For conditioning-set sizes d = 0, 1, 2, ...:
   - For each remaining edge (i, j), consider all subsets S of size d drawn
     from the neighbours of i (or j).
   - Compute the partial correlation of i and j given S.
   - If the p-value exceeds `alpha`, remove the edge and record S as the
     separation set.
3. Stop when the maximum neighbourhood size is smaller than the current
   conditioning-set size.

## Synthetic Data

The DAG used for testing:

```
X0 -----> X2 <----- X1
            |
            v
           X3
```

X0 and X1 are independent Gaussian root nodes.  X2 is a linear combination
of X0 and X1 plus noise.  X3 depends linearly on X2 plus noise.

## Usage

```bash
python -m aicoding.PC.solution
```

This runs a demonstration that generates synthetic data, runs the PC
algorithm, and prints the discovered skeleton along with selected partial
correlations.
