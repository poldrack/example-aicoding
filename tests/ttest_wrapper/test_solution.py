"""Tests for ttest_wrapper — R-style wrapper around statsmodels ttest_ind."""

import numpy as np
import pytest

from aicoding.ttest_wrapper.solution import ttest_ind, TTestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_groups():
    """Two simple groups with known properties."""
    x1 = np.array([1, 2, 3, 4, 5], dtype=float)
    x2 = np.array([2, 4, 6, 8, 10], dtype=float)
    return x1, x2


@pytest.fixture
def identical_groups():
    """Two identical groups — edge case where t=0, p=1."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    return x, x.copy()


@pytest.fixture
def very_different_groups():
    """Two groups with huge separation — guaranteed significant."""
    x1 = np.array([100, 101, 102, 103, 104], dtype=float)
    x2 = np.array([1, 2, 3, 4, 5], dtype=float)
    return x1, x2


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestTTestResultInterface:
    """Verify that ttest_ind exists and returns a TTestResult."""

    def test_returns_ttest_result(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert isinstance(result, TTestResult)

    def test_result_has_t_statistic(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert hasattr(result, "t_statistic")
        assert isinstance(result.t_statistic, float)

    def test_result_has_p_value(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert hasattr(result, "p_value")
        assert isinstance(result.p_value, float)

    def test_result_has_df(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert hasattr(result, "df")
        assert isinstance(result.df, float)

    def test_result_has_conf_int(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert hasattr(result, "conf_int")
        assert len(result.conf_int) == 2
        # Lower bound should be less than upper bound
        assert result.conf_int[0] < result.conf_int[1]

    def test_result_has_means(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        assert hasattr(result, "mean_x1")
        assert hasattr(result, "mean_x2")
        assert isinstance(result.mean_x1, float)
        assert isinstance(result.mean_x2, float)


# ---------------------------------------------------------------------------
# Correctness tests with known data (pooled variance)
# ---------------------------------------------------------------------------

class TestPooledVariance:
    """Test with usevar='pooled' (equal variance assumption)."""

    def test_t_statistic_known(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.t_statistic == pytest.approx(-1.8973665961010275, rel=1e-6)

    def test_p_value_known(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.p_value == pytest.approx(0.09434977284243763, rel=1e-6)

    def test_df_pooled(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.df == pytest.approx(8.0)

    def test_means(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.mean_x1 == pytest.approx(3.0)
        assert result.mean_x2 == pytest.approx(6.0)

    def test_confidence_interval_pooled(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        # 95% CI for difference in means (x1 - x2)
        assert result.conf_int[0] == pytest.approx(-6.646112680506018, rel=1e-4)
        assert result.conf_int[1] == pytest.approx(0.6461126805060187, rel=1e-4)


# ---------------------------------------------------------------------------
# Correctness tests with unequal variance (Welch)
# ---------------------------------------------------------------------------

class TestUnequalVariance:
    """Test with usevar='unequal' (Welch t-test)."""

    def test_t_statistic_welch(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="unequal")
        # t-stat is the same as pooled when group sizes are equal
        assert result.t_statistic == pytest.approx(-1.8973665961010275, rel=1e-6)

    def test_p_value_welch(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="unequal")
        # p-value differs from pooled due to different df
        assert result.p_value == pytest.approx(0.10753119493062728, rel=1e-6)

    def test_df_welch(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="unequal")
        # Welch-Satterthwaite df
        assert result.df == pytest.approx(5.882352941176469, rel=1e-6)

    def test_confidence_interval_welch(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="unequal")
        assert result.conf_int[0] == pytest.approx(-6.887741643736974, rel=1e-4)
        assert result.conf_int[1] == pytest.approx(0.8877416437369741, rel=1e-4)


# ---------------------------------------------------------------------------
# Edge case: identical groups
# ---------------------------------------------------------------------------

class TestIdenticalGroups:
    """When both groups are identical, t should be 0 and p should be 1."""

    def test_t_statistic_zero(self, identical_groups):
        x1, x2 = identical_groups
        result = ttest_ind(x1, x2)
        assert result.t_statistic == pytest.approx(0.0, abs=1e-10)

    def test_p_value_one(self, identical_groups):
        x1, x2 = identical_groups
        result = ttest_ind(x1, x2)
        assert result.p_value == pytest.approx(1.0, abs=1e-10)

    def test_means_equal(self, identical_groups):
        x1, x2 = identical_groups
        result = ttest_ind(x1, x2)
        assert result.mean_x1 == pytest.approx(result.mean_x2)

    def test_ci_contains_zero(self, identical_groups):
        x1, x2 = identical_groups
        result = ttest_ind(x1, x2)
        assert result.conf_int[0] <= 0.0 <= result.conf_int[1]


# ---------------------------------------------------------------------------
# Edge case: very different groups (significant result)
# ---------------------------------------------------------------------------

class TestVeryDifferentGroups:
    """When groups are very far apart the result must be highly significant."""

    def test_large_t_statistic(self, very_different_groups):
        x1, x2 = very_different_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.t_statistic == pytest.approx(99.0, rel=1e-6)

    def test_tiny_p_value(self, very_different_groups):
        x1, x2 = very_different_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        assert result.p_value < 1e-10

    def test_ci_does_not_contain_zero(self, very_different_groups):
        x1, x2 = very_different_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        # Both bounds should be positive (x1 >> x2)
        assert result.conf_int[0] > 0
        assert result.conf_int[1] > 0


# ---------------------------------------------------------------------------
# Alternative hypothesis options
# ---------------------------------------------------------------------------

class TestAlternativeHypothesis:
    """Test the 'alternative' parameter."""

    def test_two_sided_default(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2)
        # Two-sided is the default
        assert result.p_value == pytest.approx(0.09434977284243763, rel=1e-6)

    def test_larger_alternative(self, simple_groups):
        x1, x2 = simple_groups
        # x1 < x2, so "larger" (x1 - x2 > 0) should have a high p-value
        result = ttest_ind(x1, x2, alternative="larger")
        assert result.p_value > 0.5

    def test_smaller_alternative(self, simple_groups):
        x1, x2 = simple_groups
        # x1 < x2, so "smaller" (x1 - x2 < 0) should have low p-value
        result = ttest_ind(x1, x2, alternative="smaller")
        assert result.p_value < 0.1


# ---------------------------------------------------------------------------
# String representation (R-style output)
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    """The __str__ method should produce output similar to R's t.test."""

    def test_str_contains_title(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "Two Sample t-test" in output

    def test_str_contains_t_statistic(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "t = " in output

    def test_str_contains_df(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "df = " in output

    def test_str_contains_p_value(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "p-value" in output

    def test_str_contains_alternative(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "alternative hypothesis" in output

    def test_str_contains_confidence_interval(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "95 percent confidence interval" in output

    def test_str_contains_sample_estimates(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "sample estimates" in output
        assert "mean of x" in output
        assert "mean of y" in output

    def test_welch_title_for_unequal(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="unequal")
        output = str(result)
        assert "Welch" in output

    def test_str_lists_input(self, simple_groups):
        x1, x2 = simple_groups
        result = ttest_ind(x1, x2, usevar="pooled")
        output = str(result)
        assert "data:" in output.lower() or "data:" in output


# ---------------------------------------------------------------------------
# Input type flexibility
# ---------------------------------------------------------------------------

class TestInputTypes:
    """The wrapper should accept lists, tuples, and numpy arrays."""

    def test_accepts_lists(self):
        result = ttest_ind([1, 2, 3], [4, 5, 6])
        assert isinstance(result, TTestResult)

    def test_accepts_tuples(self):
        result = ttest_ind((1, 2, 3), (4, 5, 6))
        assert isinstance(result, TTestResult)

    def test_accepts_numpy_arrays(self):
        result = ttest_ind(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert isinstance(result, TTestResult)
