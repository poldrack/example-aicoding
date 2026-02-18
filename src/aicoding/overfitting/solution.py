"""Simulate the effect of overfitting in polynomial regression.

Generates bivariate normal data, fits linear, 2nd-order, and 9th-order
polynomial regression models, computes training and test errors, and
plots the fitted curves overlaid on the training data.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def generate_data(n=32, correlation=0.5, seed=42):
    """Generate synthetic bivariate normal data.

    Draws *n* observations from a bivariate normal distribution with
    mean [0, 0] and a covariance matrix that yields the requested
    Pearson correlation between the two variables.

    Args:
        n: Number of observations.
        correlation: Population correlation between X and y.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (X, y) where both are 1-D numpy arrays of length *n*.
    """
    rng = np.random.default_rng(seed)
    cov = [[1.0, correlation], [correlation, 1.0]]
    data = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def fit_models(X_train, y_train):
    """Fit linear, 2nd-order, and 9th-order polynomial regression models.

    Each model is represented as a scikit-learn Pipeline consisting of a
    PolynomialFeatures transformer and a LinearRegression estimator.

    Args:
        X_train: 1-D array of predictor values.
        y_train: 1-D array of response values.

    Returns:
        Dict mapping model name to fitted Pipeline:
        ``{"linear": ..., "poly2": ..., "poly9": ...}``
    """
    X = X_train.reshape(-1, 1)
    models = {}
    for name, degree in [("linear", 1), ("poly2", 2), ("poly9", 9)]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regression", LinearRegression()),
        ])
        pipe.fit(X, y_train)
        models[name] = pipe
    return models


def compute_errors(models, X_train, y_train, X_test, y_test):
    """Compute mean squared error for each model on train and test sets.

    Args:
        models: Dict of fitted model pipelines (from :func:`fit_models`).
        X_train: 1-D array of training predictor values.
        y_train: 1-D array of training response values.
        X_test: 1-D array of test predictor values.
        y_test: 1-D array of test response values.

    Returns:
        Dict with keys ``"train"`` and ``"test"``, each mapping to a dict
        of ``{model_name: mse_value}``.
    """
    X_tr = X_train.reshape(-1, 1)
    X_te = X_test.reshape(-1, 1)
    errors = {"train": {}, "test": {}}
    for name, model in models.items():
        errors["train"][name] = mean_squared_error(y_train, model.predict(X_tr))
        errors["test"][name] = mean_squared_error(y_test, model.predict(X_te))
    return errors


def plot_fits(X_train, y_train, models):
    """Plot the training data with fitted regression curves for each model.

    Creates a scatter plot of the training data and overlays the
    predicted curves for the linear, 2nd-order, and 9th-order polynomial
    models.

    Args:
        X_train: 1-D array of training predictor values.
        y_train: 1-D array of training response values.
        models: Dict of fitted model pipelines (from :func:`fit_models`).

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter the training data
    ax.scatter(X_train, y_train, color="black", alpha=0.6, label="Training data", zorder=5)

    # Dense grid for smooth fitted curves
    x_min, x_max = X_train.min(), X_train.max()
    margin = (x_max - x_min) * 0.05
    X_grid = np.linspace(x_min - margin, x_max + margin, 200).reshape(-1, 1)

    colors = {"linear": "blue", "poly2": "green", "poly9": "red"}
    labels = {"linear": "Linear (degree 1)", "poly2": "Polynomial (degree 2)", "poly9": "Polynomial (degree 9)"}

    for name, model in models.items():
        y_pred = model.predict(X_grid)
        ax.plot(X_grid, y_pred, color=colors[name], linewidth=2, label=labels[name])

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Overfitting in Polynomial Regression")
    ax.legend()

    # Clamp y-axis to keep 9th-order curve from blowing out the display
    y_range = y_train.max() - y_train.min()
    ax.set_ylim(y_train.min() - y_range, y_train.max() + y_range)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # ---- Generate data ----
    X_train, y_train = generate_data(n=32, correlation=0.5, seed=42)
    X_test, y_test = generate_data(n=32, correlation=0.5, seed=99)

    # ---- Fit models ----
    models = fit_models(X_train, y_train)

    # ---- Compute errors ----
    errors = compute_errors(models, X_train, y_train, X_test, y_test)

    print("Training errors (MSE):")
    for name, err in errors["train"].items():
        print(f"  {name:8s}: {err:.6f}")

    print("\nTest errors (MSE):")
    for name, err in errors["test"].items():
        print(f"  {name:8s}: {err:.6f}")

    # ---- Plot ----
    os.makedirs("outputs", exist_ok=True)
    fig = plot_fits(X_train, y_train, models)
    fig.savefig("outputs/overfitting_plot.png", dpi=150)
    print("\nPlot saved to outputs/overfitting_plot.png")
    plt.close(fig)
