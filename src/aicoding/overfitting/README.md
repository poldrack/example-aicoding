# Overfitting Simulation

## Problem

Simulate the effect of overfitting in polynomial regression by fitting models of increasing complexity to small-sample bivariate normal data and comparing training vs. test error.

## Approach

1. **Data generation** (`generate_data`): Draws observations from a bivariate normal distribution with a specified Pearson correlation using `numpy.random.Generator.multivariate_normal`. The first variable serves as the predictor (X) and the second as the response (y).

2. **Model fitting** (`fit_models`): Fits three scikit-learn Pipelines (PolynomialFeatures + LinearRegression) of degree 1 (linear), 2, and 9 to the training data.

3. **Error computation** (`compute_errors`): Computes mean squared error (MSE) on both the training set and an independently generated test set drawn from the same distribution.

4. **Plotting** (`plot_fits`): Scatter-plots the training data and overlays the fitted curves for all three models, with a legend and clamped y-axis to keep the high-degree polynomial visible.

## Key Result

With only 32 observations, the 9th-order polynomial achieves the lowest training error but dramatically higher test error than the simpler models, clearly demonstrating overfitting.

## Running

```bash
python -m aicoding.overfitting.solution
pytest tests/overfitting/ -v
```
