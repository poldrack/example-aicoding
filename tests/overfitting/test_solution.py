"""Tests for the overfitting simulation module."""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt

from aicoding.overfitting.solution import (
    generate_data,
    fit_models,
    compute_errors,
    plot_fits,
)

matplotlib.use("Agg")


@pytest.fixture
def training_data():
    """Generate reproducible training data."""
    X_train, y_train = generate_data(n=32, correlation=0.5, seed=42)
    return X_train, y_train


@pytest.fixture
def test_data():
    """Generate reproducible test data from the same distribution."""
    X_test, y_test = generate_data(n=32, correlation=0.5, seed=99)
    return X_test, y_test


@pytest.fixture
def fitted_models(training_data):
    """Fit all three models on training data."""
    X_train, y_train = training_data
    return fit_models(X_train, y_train)


# ---------------------------------------------------------------------------
# Tests for generate_data
# ---------------------------------------------------------------------------

class TestGenerateData:
    """Tests for the generate_data function."""

    def test_returns_two_arrays(self):
        """generate_data should return a tuple of two arrays (X, y)."""
        result = generate_data(n=32, correlation=0.5, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_correct_number_of_observations(self):
        """Output arrays should have the requested number of observations."""
        X, y = generate_data(n=32, correlation=0.5, seed=42)
        assert X.shape[0] == 32
        assert y.shape[0] == 32

    def test_x_is_one_dimensional(self):
        """X should be a 1-D array (single predictor)."""
        X, y = generate_data(n=32, correlation=0.5, seed=42)
        assert X.ndim == 1

    def test_approximate_correlation(self):
        """The sample correlation between X and y should be close to the
        population correlation of 0.5 (within a tolerance for n=32)."""
        X, y = generate_data(n=32, correlation=0.5, seed=42)
        r = np.corrcoef(X, y)[0, 1]
        assert abs(r - 0.5) < 0.25, f"Sample correlation {r:.3f} too far from 0.5"

    def test_seed_reproducibility(self):
        """Same seed should produce identical data."""
        X1, y1 = generate_data(n=32, correlation=0.5, seed=42)
        X2, y2 = generate_data(n=32, correlation=0.5, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        X1, y1 = generate_data(n=32, correlation=0.5, seed=42)
        X2, y2 = generate_data(n=32, correlation=0.5, seed=99)
        assert not np.array_equal(X1, X2)

    def test_custom_n(self):
        """Should support arbitrary sample sizes."""
        X, y = generate_data(n=100, correlation=0.5, seed=42)
        assert X.shape[0] == 100
        assert y.shape[0] == 100

    def test_edge_case_zero_correlation(self):
        """Correlation=0 should produce approximately uncorrelated data."""
        X, y = generate_data(n=1000, correlation=0.0, seed=42)
        r = np.corrcoef(X, y)[0, 1]
        assert abs(r) < 0.15, f"Expected near-zero correlation, got {r:.3f}"


# ---------------------------------------------------------------------------
# Tests for fit_models
# ---------------------------------------------------------------------------

class TestFitModels:
    """Tests for the fit_models function."""

    def test_returns_dict_of_three_models(self, training_data):
        """fit_models should return a dict with three model entries."""
        X_train, y_train = training_data
        models = fit_models(X_train, y_train)
        assert isinstance(models, dict)
        assert len(models) == 3

    def test_model_keys(self, training_data):
        """Dict keys should identify the model orders."""
        X_train, y_train = training_data
        models = fit_models(X_train, y_train)
        assert "linear" in models
        assert "poly2" in models
        assert "poly9" in models

    def test_models_can_predict(self, training_data):
        """Each model should be callable for prediction on the training X."""
        X_train, y_train = training_data
        models = fit_models(X_train, y_train)
        for name, model in models.items():
            pred = model.predict(np.sort(X_train).reshape(-1, 1) if hasattr(model, 'predict') else X_train)
            assert pred is not None, f"Model {name} returned None for predict"
            assert len(pred) == len(X_train)

    def test_linear_model_has_two_coefficients(self, training_data):
        """Linear model should have intercept + 1 coefficient."""
        X_train, y_train = training_data
        models = fit_models(X_train, y_train)
        # Pipeline's last step is the linear model
        linear_model = models["linear"]
        # Accept sklearn Pipeline or a plain model
        if hasattr(linear_model, "named_steps"):
            reg = linear_model.named_steps.get("regression") or linear_model.named_steps.get("model") or linear_model[-1]
        else:
            reg = linear_model
        assert len(reg.coef_[0]) if reg.coef_.ndim > 1 else len(reg.coef_) == 1


# ---------------------------------------------------------------------------
# Tests for compute_errors
# ---------------------------------------------------------------------------

class TestComputeErrors:
    """Tests for the compute_errors function."""

    def test_returns_dict_with_train_and_test(self, fitted_models, training_data, test_data):
        """compute_errors should return a dict with 'train' and 'test' sub-dicts."""
        X_train, y_train = training_data
        X_test, y_test = test_data
        errors = compute_errors(fitted_models, X_train, y_train, X_test, y_test)
        assert "train" in errors
        assert "test" in errors

    def test_errors_contain_all_model_names(self, fitted_models, training_data, test_data):
        """Both train and test error dicts should contain keys for each model."""
        X_train, y_train = training_data
        X_test, y_test = test_data
        errors = compute_errors(fitted_models, X_train, y_train, X_test, y_test)
        for split in ("train", "test"):
            for model_name in ("linear", "poly2", "poly9"):
                assert model_name in errors[split], (
                    f"Missing key '{model_name}' in errors['{split}']"
                )

    def test_errors_are_non_negative(self, fitted_models, training_data, test_data):
        """All MSE values should be non-negative."""
        X_train, y_train = training_data
        X_test, y_test = test_data
        errors = compute_errors(fitted_models, X_train, y_train, X_test, y_test)
        for split in ("train", "test"):
            for name, val in errors[split].items():
                assert val >= 0, f"errors['{split}']['{name}'] = {val} is negative"

    def test_training_error_decreases_with_complexity(self, fitted_models, training_data, test_data):
        """Training error should decrease (or stay equal) as model complexity
        increases: linear >= poly2 >= poly9."""
        X_train, y_train = training_data
        X_test, y_test = test_data
        errors = compute_errors(fitted_models, X_train, y_train, X_test, y_test)
        train = errors["train"]
        assert train["linear"] >= train["poly2"] - 1e-8, (
            f"Linear train error ({train['linear']:.6f}) should be >= poly2 ({train['poly2']:.6f})"
        )
        assert train["poly2"] >= train["poly9"] - 1e-8, (
            f"Poly2 train error ({train['poly2']:.6f}) should be >= poly9 ({train['poly9']:.6f})"
        )

    def test_overfitting_pattern_test_error(self, fitted_models, training_data, test_data):
        """Test error for 9th-order polynomial should be higher than for the
        linear model, demonstrating overfitting."""
        X_train, y_train = training_data
        X_test, y_test = test_data
        errors = compute_errors(fitted_models, X_train, y_train, X_test, y_test)
        test_err = errors["test"]
        assert test_err["poly9"] > test_err["linear"], (
            f"Expected poly9 test error ({test_err['poly9']:.4f}) > linear "
            f"({test_err['linear']:.4f}) to demonstrate overfitting"
        )


# ---------------------------------------------------------------------------
# Tests for plot_fits
# ---------------------------------------------------------------------------

class TestPlotFits:
    """Tests for the plot_fits function."""

    def test_returns_figure(self, training_data, fitted_models):
        """plot_fits should return a matplotlib Figure."""
        X_train, y_train = training_data
        fig = plot_fits(X_train, y_train, fitted_models)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self, training_data, fitted_models):
        """The returned figure should contain at least one Axes."""
        X_train, y_train = training_data
        fig = plot_fits(X_train, y_train, fitted_models)
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_axes_has_lines(self, training_data, fitted_models):
        """The axes should contain plotted lines (one per model)."""
        X_train, y_train = training_data
        fig = plot_fits(X_train, y_train, fitted_models)
        ax = fig.get_axes()[0]
        # Should have at least 3 lines (one per fitted model) plus scatter
        assert len(ax.get_lines()) >= 3, (
            f"Expected at least 3 lines on the axes, found {len(ax.get_lines())}"
        )
        plt.close(fig)

    def test_axes_has_scatter_data(self, training_data, fitted_models):
        """The axes should contain the training scatter points."""
        X_train, y_train = training_data
        fig = plot_fits(X_train, y_train, fitted_models)
        ax = fig.get_axes()[0]
        # Scatter points show up as collections or as a line with no connecting line
        has_data = len(ax.collections) > 0 or len(ax.get_lines()) > 0
        assert has_data, "Expected scatter data on the axes"
        plt.close(fig)

    def test_axes_has_legend(self, training_data, fitted_models):
        """The axes should include a legend identifying the models."""
        X_train, y_train = training_data
        fig = plot_fits(X_train, y_train, fitted_models)
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        assert legend is not None, "Expected a legend on the axes"
        plt.close(fig)
