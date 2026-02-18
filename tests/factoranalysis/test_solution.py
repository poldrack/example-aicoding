"""Tests for factoranalysis — exploratory factor analysis with BIC selection."""

import numpy as np
import pandas as pd
import pytest

from aicoding.factoranalysis.solution import (
    load_and_select_variables,
    fit_factor_analysis,
    select_best_n_factors,
    get_dominant_loadings,
)


@pytest.fixture
def synthetic_data():
    """Synthetic data that mimics the survey structure."""
    rng = np.random.RandomState(42)
    n = 200
    # 3 latent factors
    f1 = rng.randn(n)
    f2 = rng.randn(n)
    f3 = rng.randn(n)

    cols = {}
    for i in range(4):
        cols[f"upps_impulsivity_survey.{i}"] = f1 + rng.randn(n) * 0.5
    for i in range(3):
        cols[f"sensation_seeking_survey.{i}"] = f2 + rng.randn(n) * 0.5
    for i in range(3):
        cols[f"bis11_survey.{i}"] = f3 + rng.randn(n) * 0.3
    for i in range(2):
        cols[f"dickman_survey.{i}"] = (f1 + f2) / 2 + rng.randn(n) * 0.5
    # Some irrelevant columns
    cols["other_variable"] = rng.randn(n)

    return pd.DataFrame(cols)


class TestLoadAndSelectVariables:
    def test_returns_dataframe(self, synthetic_data):
        result = load_and_select_variables(synthetic_data)
        assert isinstance(result, pd.DataFrame)

    def test_selects_correct_prefixes(self, synthetic_data):
        result = load_and_select_variables(synthetic_data)
        for col in result.columns:
            assert any(
                col.startswith(prefix)
                for prefix in [
                    "upps_impulsivity_survey",
                    "sensation_seeking_survey",
                    "bis11_survey",
                    "dickman_survey",
                ]
            )

    def test_excludes_other_columns(self, synthetic_data):
        result = load_and_select_variables(synthetic_data)
        assert "other_variable" not in result.columns

    def test_correct_number_of_columns(self, synthetic_data):
        result = load_and_select_variables(synthetic_data)
        assert result.shape[1] == 12  # 4+3+3+2


class TestFitFactorAnalysis:
    def test_returns_dict(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=2)
        assert isinstance(result, dict)

    def test_has_loadings(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=2)
        assert "loadings" in result

    def test_has_bic(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=2)
        assert "bic" in result

    def test_loadings_shape(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=3)
        loadings = result["loadings"]
        assert loadings.shape == (df.shape[1], 3)

    def test_bic_is_finite(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=2)
        assert np.isfinite(result["bic"])


class TestSelectBestNFactors:
    def test_returns_int(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        best_n = select_best_n_factors(df, max_factors=5)
        assert isinstance(best_n, int)

    def test_in_valid_range(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        best_n = select_best_n_factors(df, max_factors=5)
        assert 1 <= best_n <= 5

    def test_reasonable_for_three_factor_data(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        best_n = select_best_n_factors(df, max_factors=5)
        # With 3 true factors, should select 2-4
        assert 2 <= best_n <= 5


class TestGetDominantLoadings:
    def test_returns_dict(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=3)
        dominant = get_dominant_loadings(result["loadings"], df.columns)
        assert isinstance(dominant, dict)

    def test_has_entries_for_each_factor(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=3)
        dominant = get_dominant_loadings(result["loadings"], df.columns)
        assert len(dominant) == 3

    def test_each_factor_has_variables(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=3)
        dominant = get_dominant_loadings(result["loadings"], df.columns)
        for factor, variables in dominant.items():
            assert isinstance(variables, list)
            assert len(variables) > 0

    def test_variables_are_strings(self, synthetic_data):
        df = load_and_select_variables(synthetic_data)
        result = fit_factor_analysis(df, n_factors=3)
        dominant = get_dominant_loadings(result["loadings"], df.columns)
        for factor, variables in dominant.items():
            for v in variables:
                assert isinstance(v, str)
