"""Tests for the lmm (linear mixed model) module."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from aicoding.lmm.solution import combine_csv_files, fit_lmm, generate_test_data


@pytest.fixture
def test_data_dir():
    """Generate test CSV files with known properties and return the temp dir."""
    tmpdir = tempfile.mkdtemp()
    rng = np.random.default_rng(42)

    n_subjects = 5
    n_trials = 100

    for i in range(n_subjects):
        compatible = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
        rng.shuffle(compatible)
        # True effect: compatible trials are ~50ms faster
        base_rt = 500 + i * 20  # random intercept per subject
        effect = -50 + i * 5  # random slope per subject
        rt = base_rt + effect * compatible + rng.normal(0, 50, n_trials)
        correct = rng.binomial(1, 0.85, n_trials)
        df = pd.DataFrame({"rt": rt, "correct": correct, "compatible": compatible})
        df.to_csv(os.path.join(tmpdir, f"subject_{i}.csv"), index=False)

    yield tmpdir

    # Cleanup
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


class TestCombineCsvFiles:
    """Test the CSV file combination function."""

    def test_returns_dataframe(self, test_data_dir):
        """combine_csv_files should return a pandas DataFrame."""
        df = combine_csv_files(test_data_dir)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, test_data_dir):
        """Combined dataframe should have rt, correct, compatible, and subject columns."""
        df = combine_csv_files(test_data_dir)
        for col in ["rt", "correct", "compatible"]:
            assert col in df.columns

    def test_has_subject_index(self, test_data_dir):
        """Combined dataframe should have a subject/filename index variable."""
        df = combine_csv_files(test_data_dir)
        # Should have a column identifying the source file
        assert "subject" in df.columns or "filename" in df.columns

    def test_correct_number_of_rows(self, test_data_dir):
        """Should have rows from all files combined."""
        df = combine_csv_files(test_data_dir)
        assert len(df) == 5 * 100  # 5 subjects x 100 trials

    def test_compatible_is_binary(self, test_data_dir):
        """Compatible column should only contain 0 and 1."""
        df = combine_csv_files(test_data_dir)
        assert set(df["compatible"].unique()).issubset({0, 1})


class TestFitLMM:
    """Test the linear mixed model fitting."""

    def test_returns_result(self, test_data_dir):
        """fit_lmm should return a model result object."""
        df = combine_csv_files(test_data_dir)
        result = fit_lmm(df)
        assert result is not None

    def test_has_fixed_effect(self, test_data_dir):
        """Result should include a fixed effect for compatible."""
        df = combine_csv_files(test_data_dir)
        result = fit_lmm(df)
        assert "compatible" in result.summary().as_text().lower()

    def test_has_random_effects(self, test_data_dir):
        """Model should include random effects."""
        df = combine_csv_files(test_data_dir)
        result = fit_lmm(df)
        # Random effects should be present
        assert result.random_effects is not None


class TestGenerateTestData:
    """Test the synthetic data generator."""

    def test_generates_files(self):
        """generate_test_data should create CSV files in a directory."""
        tmpdir = generate_test_data()
        files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        assert len(files) > 0
        # Cleanup
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    def test_generated_data_has_correct_columns(self):
        """Generated CSV files should have rt, correct, compatible columns."""
        tmpdir = generate_test_data()
        files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        df = pd.read_csv(os.path.join(tmpdir, files[0]))
        assert "rt" in df.columns
        assert "correct" in df.columns
        assert "compatible" in df.columns
        # Cleanup
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
