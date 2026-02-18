"""Tests for the Daw two-step reinforcement learning task analysis.

Tests cover:
- Synthetic data generation (shape, columns, value ranges, reproducibility)
- Analysis function (return type, required keys, regression properties)
- Behavioral signatures (model-based vs model-free indices)
- Edge cases (minimal data, all-same rewards)
"""

import numpy as np
import pandas as pd
import pytest

from aicoding.twostep.solution import analyze_twostep, generate_synthetic_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Standard synthetic dataset with balanced model-based/model-free."""
    return generate_synthetic_data(n_trials=200, model_based_weight=0.5, seed=42)


@pytest.fixture
def strongly_model_based_df():
    """Dataset generated with high model-based weight."""
    return generate_synthetic_data(n_trials=500, model_based_weight=1.0, seed=99)


@pytest.fixture
def strongly_model_free_df():
    """Dataset generated with zero model-based weight (pure model-free)."""
    return generate_synthetic_data(n_trials=500, model_based_weight=0.0, seed=99)


# ---------------------------------------------------------------------------
# Test 1: generate_synthetic_data returns a DataFrame
# ---------------------------------------------------------------------------

def test_generate_returns_dataframe(synthetic_df):
    assert isinstance(synthetic_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 2: DataFrame has the correct number of rows
# ---------------------------------------------------------------------------

def test_generate_correct_row_count():
    df = generate_synthetic_data(n_trials=150, seed=0)
    # The first trial has no previous_reward, so there should be
    # n_trials rows; the previous_reward column for trial 0 may be NaN
    # but the row should still exist.
    assert len(df) == 150


# ---------------------------------------------------------------------------
# Test 3: DataFrame has all required columns
# ---------------------------------------------------------------------------

def test_generate_has_required_columns(synthetic_df):
    required_cols = {
        "trial",
        "step1_choice",
        "step1_state",
        "step2_state",
        "step2_choice",
        "reward",
        "previous_reward",
    }
    assert required_cols.issubset(set(synthetic_df.columns))


# ---------------------------------------------------------------------------
# Test 4: Value ranges for binary / categorical columns
# ---------------------------------------------------------------------------

def test_generate_value_ranges(synthetic_df):
    # reward and previous_reward should be 0 or 1 (or NaN for first trial)
    assert synthetic_df["reward"].dropna().isin([0, 1]).all()
    assert synthetic_df["previous_reward"].dropna().isin([0, 1]).all()
    # step1_choice should be binary (0 or 1)
    assert synthetic_df["step1_choice"].isin([0, 1]).all()
    # step2_choice should be binary (0 or 1)
    assert synthetic_df["step2_choice"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Test 5: Transition types — step2_state contains "common" and "rare"
# ---------------------------------------------------------------------------

def test_generate_transition_types(synthetic_df):
    unique_transitions = set(synthetic_df["step2_state"].unique())
    assert "common" in unique_transitions
    assert "rare" in unique_transitions


# ---------------------------------------------------------------------------
# Test 6: Reproducibility — same seed produces identical data
# ---------------------------------------------------------------------------

def test_generate_reproducibility():
    df1 = generate_synthetic_data(n_trials=100, model_based_weight=0.5, seed=123)
    df2 = generate_synthetic_data(n_trials=100, model_based_weight=0.5, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Test 7: analyze_twostep returns a dict with required keys
# ---------------------------------------------------------------------------

def test_analyze_returns_dict_with_keys(synthetic_df):
    result = analyze_twostep(synthetic_df)
    assert isinstance(result, dict)
    assert "model_free_index" in result
    assert "model_based_index" in result
    assert "coefficients" in result


# ---------------------------------------------------------------------------
# Test 8: model_free_index and model_based_index are numeric scalars
# ---------------------------------------------------------------------------

def test_analyze_indices_are_numeric(synthetic_df):
    result = analyze_twostep(synthetic_df)
    assert isinstance(result["model_free_index"], (int, float, np.floating))
    assert isinstance(result["model_based_index"], (int, float, np.floating))


# ---------------------------------------------------------------------------
# Test 9: coefficients is a dict-like object with regression terms
# ---------------------------------------------------------------------------

def test_analyze_coefficients_structure(synthetic_df):
    result = analyze_twostep(synthetic_df)
    coefs = result["coefficients"]
    # Should contain at least intercept, reward, transition, interaction
    assert isinstance(coefs, dict)
    assert len(coefs) >= 3  # at least reward, transition, interaction


# ---------------------------------------------------------------------------
# Test 10: Strongly model-based agent shows larger model_based_index
#           than a purely model-free agent
# ---------------------------------------------------------------------------

def test_model_based_agent_has_higher_mb_index(
    strongly_model_based_df, strongly_model_free_df
):
    result_mb = analyze_twostep(strongly_model_based_df)
    result_mf = analyze_twostep(strongly_model_free_df)
    # A strongly model-based agent should have a larger interaction
    # effect (model_based_index) than a purely model-free agent.
    assert result_mb["model_based_index"] > result_mf["model_based_index"]


# ---------------------------------------------------------------------------
# Test 11: Purely model-free agent should have a positive model_free_index
# ---------------------------------------------------------------------------

def test_model_free_agent_positive_mf_index(strongly_model_free_df):
    result = analyze_twostep(strongly_model_free_df)
    # A model-free agent repeats rewarded actions regardless of transition,
    # so the main effect of reward on stay probability should be positive.
    assert result["model_free_index"] > 0


# ---------------------------------------------------------------------------
# Test 12: Common transitions are more frequent than rare transitions
# ---------------------------------------------------------------------------

def test_common_more_frequent_than_rare(synthetic_df):
    counts = synthetic_df["step2_state"].value_counts()
    assert counts["common"] > counts["rare"]


# ---------------------------------------------------------------------------
# Test 13: Edge case — small dataset still produces results without error
# ---------------------------------------------------------------------------

def test_analyze_small_dataset():
    df = generate_synthetic_data(n_trials=30, model_based_weight=0.5, seed=7)
    result = analyze_twostep(df)
    assert "model_free_index" in result
    assert "model_based_index" in result


# ---------------------------------------------------------------------------
# Test 14: Different seeds produce different data
# ---------------------------------------------------------------------------

def test_different_seeds_different_data():
    df1 = generate_synthetic_data(n_trials=100, seed=1)
    df2 = generate_synthetic_data(n_trials=100, seed=2)
    # The reward sequences should differ
    assert not (df1["reward"].values == df2["reward"].values).all()
