# corridor (#19)

Schönbrodt "corridor of stability" simulation.

## Approach

- Generates bivariate normal data with a known population correlation.
- For each simulation, computes sample correlations at increasing sample sizes (n=10 to n=max_n).
- The corridor plot shows individual simulation trajectories, plus a 95% CI envelope and median line.
- Demonstrates that correlation estimates stabilize as sample size increases.
