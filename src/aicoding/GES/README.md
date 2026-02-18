# GES (#14)

Greedy equivalence search for causal discovery.

## Approach

- Implements a simplified GES with two phases:
  1. **Forward phase**: Greedily add edges that improve the total BIC score.
  2. **Backward phase**: Greedily remove edges that improve the score.
- BIC scoring uses linear regression log-likelihood with a complexity penalty.
- Includes a `generate_test_data` function that creates random DAGs and generates samples from linear structural equation models.
