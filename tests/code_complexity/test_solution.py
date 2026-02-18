"""Tests for code_complexity — cyclomatic complexity, maintainability index, Halstead metrics."""

import os
import tempfile

import pytest

from aicoding.code_complexity.solution import (
    compute_cyclomatic_complexity,
    compute_maintainability_index,
    compute_halstead_metrics,
    analyze_directory,
)


@pytest.fixture
def sample_python_dir(tmp_path):
    """Create a temp directory with sample Python files."""
    # Simple function
    (tmp_path / "simple.py").write_text(
        "def hello():\n    print('hello')\n"
    )
    # More complex function with branches
    (tmp_path / "complex.py").write_text(
        "def decide(x):\n"
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            return 'big'\n"
        "        return 'small'\n"
        "    elif x == 0:\n"
        "        return 'zero'\n"
        "    else:\n"
        "        return 'negative'\n"
    )
    # Non-python file (should be ignored)
    (tmp_path / "readme.txt").write_text("not python")
    return tmp_path


@pytest.fixture
def simple_code():
    return "def foo():\n    return 1\n"


@pytest.fixture
def complex_code():
    return (
        "def bar(x):\n"
        "    if x > 0:\n"
        "        for i in range(x):\n"
        "            if i % 2 == 0:\n"
        "                print(i)\n"
        "    return x\n"
    )


class TestCyclomaticComplexity:
    def test_returns_list(self, simple_code):
        result = compute_cyclomatic_complexity(simple_code)
        assert isinstance(result, list)

    def test_simple_function_low_complexity(self, simple_code):
        result = compute_cyclomatic_complexity(simple_code)
        assert len(result) >= 1
        assert result[0]["complexity"] <= 2

    def test_complex_function_higher_complexity(self, complex_code):
        result = compute_cyclomatic_complexity(complex_code)
        assert len(result) >= 1
        assert result[0]["complexity"] >= 3

    def test_result_has_expected_keys(self, simple_code):
        result = compute_cyclomatic_complexity(simple_code)
        assert len(result) >= 1
        entry = result[0]
        assert "name" in entry
        assert "complexity" in entry

    def test_empty_code(self):
        result = compute_cyclomatic_complexity("")
        assert isinstance(result, list)


class TestMaintainabilityIndex:
    def test_returns_numeric(self, simple_code):
        result = compute_maintainability_index(simple_code)
        assert isinstance(result, (int, float))

    def test_simple_code_high_mi(self, simple_code):
        result = compute_maintainability_index(simple_code)
        assert result > 50  # simple code should be maintainable

    def test_value_in_range(self, simple_code):
        result = compute_maintainability_index(simple_code)
        assert 0 <= result <= 100


class TestHalsteadMetrics:
    def test_returns_dict_or_object(self, simple_code):
        result = compute_halstead_metrics(simple_code)
        # Should return something with Halstead metrics
        assert result is not None

    def test_has_volume(self, simple_code):
        result = compute_halstead_metrics(simple_code)
        assert "volume" in result or hasattr(result, "volume")

    def test_has_difficulty(self, simple_code):
        result = compute_halstead_metrics(simple_code)
        assert "difficulty" in result or hasattr(result, "difficulty")

    def test_has_effort(self, simple_code):
        result = compute_halstead_metrics(simple_code)
        assert "effort" in result or hasattr(result, "effort")

    def test_empty_code(self):
        result = compute_halstead_metrics("")
        assert result is not None


class TestAnalyzeDirectory:
    def test_returns_dict(self, sample_python_dir):
        result = analyze_directory(str(sample_python_dir))
        assert isinstance(result, dict)

    def test_has_entries_for_python_files(self, sample_python_dir):
        result = analyze_directory(str(sample_python_dir))
        assert len(result) >= 2
        filenames = list(result.keys())
        assert any("simple" in f for f in filenames)
        assert any("complex" in f for f in filenames)

    def test_no_non_python_files(self, sample_python_dir):
        result = analyze_directory(str(sample_python_dir))
        for filename in result.keys():
            assert filename.endswith(".py")

    def test_each_entry_has_metrics(self, sample_python_dir):
        result = analyze_directory(str(sample_python_dir))
        for filename, metrics in result.items():
            assert "cyclomatic_complexity" in metrics
            assert "maintainability_index" in metrics
            assert "halstead" in metrics

    def test_nonexistent_directory(self):
        result = analyze_directory("/nonexistent/path/xyz")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_empty_directory(self, tmp_path):
        result = analyze_directory(str(tmp_path))
        assert isinstance(result, dict)
        assert len(result) == 0
