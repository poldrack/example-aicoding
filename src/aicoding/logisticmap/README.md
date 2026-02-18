# logisticmap (#13)

Logistic map simulation and bifurcation diagram.

## Approach

- Implements the logistic map iteration `x_{n+1} = r * x_n * (1 - x_n)`.
- For the bifurcation diagram, iterates the map for many r values (2.5–4.0), discards transient iterations, and plots the remaining x values vs r.
- The result is the classic bifurcation diagram showing period doubling and chaos.
