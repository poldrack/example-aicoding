# code_complexity (#8)

Compute cyclomatic complexity, maintainability index, and Halstead metrics for Python files using the radon package.

## Approach

- `compute_cyclomatic_complexity()`: Uses `radon.complexity.cc_visit` to analyze control flow.
- `compute_maintainability_index()`: Uses `radon.metrics.mi_visit` for the MI score.
- `compute_halstead_metrics()`: Uses `radon.metrics.h_visit` for operator/operand-based metrics.
- `analyze_directory()`: Scans a directory for `.py` files and computes all three metrics for each.
