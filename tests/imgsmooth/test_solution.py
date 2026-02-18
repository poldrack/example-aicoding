"""Tests for the imgsmooth (spatial smoothing on NIfTI images) module."""

import os
import tempfile

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from aicoding.imgsmooth.solution import (
    ImageSmoother,
    create_test_image,
    plot_input_vs_output,
)


@pytest.fixture
def synthetic_nifti(tmp_path):
    """Create a synthetic NIfTI image with structured data for testing.

    The image has a bright cube embedded in a dark background so that
    smoothing effects are clearly detectable at the edges.
    """
    shape = (32, 32, 32)
    data = np.zeros(shape, dtype=np.float64)
    # Place a bright cube in the center
    data[10:22, 10:22, 10:22] = 100.0
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    filepath = str(tmp_path / "test_image.nii.gz")
    nib.save(img, filepath)
    return filepath


@pytest.fixture
def uniform_nifti(tmp_path):
    """Create a uniform NIfTI image (edge case: smoothing should not change data)."""
    shape = (16, 16, 16)
    data = np.full(shape, 42.0, dtype=np.float64)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    filepath = str(tmp_path / "uniform_image.nii.gz")
    nib.save(img, filepath)
    return filepath


class TestImageSmootherInstantiation:
    """Test that ImageSmoother can be instantiated."""

    def test_default_instantiation(self):
        """ImageSmoother should be instantiable with default parameters."""
        smoother = ImageSmoother()
        assert smoother is not None

    def test_custom_kernel_size(self):
        """ImageSmoother should accept a custom kernel size."""
        smoother = ImageSmoother(size=5)
        assert smoother.size == 5

    def test_default_kernel_size(self):
        """ImageSmoother should have a sensible default kernel size."""
        smoother = ImageSmoother()
        assert smoother.size == 3


class TestSmoothing:
    """Test the smoothing operation on NIfTI images."""

    def test_smooth_returns_nifti_image(self, synthetic_nifti):
        """smooth() should return a nibabel Nifti1Image."""
        smoother = ImageSmoother()
        result = smoother.smooth(synthetic_nifti)
        assert isinstance(result, nib.Nifti1Image)

    def test_output_shape_matches_input(self, synthetic_nifti):
        """Smoothed image should have the same shape as the input."""
        smoother = ImageSmoother()
        result = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        assert result.shape == input_img.shape

    def test_smoothing_changes_data(self, synthetic_nifti):
        """Smoothing a non-uniform image should produce different data."""
        smoother = ImageSmoother()
        result = smoother.smooth(synthetic_nifti)
        input_data = nib.load(synthetic_nifti).get_fdata()
        output_data = result.get_fdata()
        assert not np.array_equal(input_data, output_data)

    def test_smoothing_preserves_affine(self, synthetic_nifti):
        """Smoothed image should retain the same affine as the input."""
        smoother = ImageSmoother()
        result = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        np.testing.assert_array_equal(result.affine, input_img.affine)

    def test_smooth_with_filepath_string(self, synthetic_nifti):
        """smooth() should accept a file path string."""
        smoother = ImageSmoother()
        result = smoother.smooth(synthetic_nifti)
        assert isinstance(result, nib.Nifti1Image)

    def test_smooth_with_nifti_object(self, synthetic_nifti):
        """smooth() should also accept a nibabel image object directly."""
        smoother = ImageSmoother()
        img = nib.load(synthetic_nifti)
        result = smoother.smooth(img)
        assert isinstance(result, nib.Nifti1Image)
        assert result.shape == img.shape

    def test_larger_kernel_produces_more_smoothing(self, synthetic_nifti):
        """A larger kernel should produce a more smoothed result."""
        smoother_small = ImageSmoother(size=3)
        smoother_large = ImageSmoother(size=7)
        input_data = nib.load(synthetic_nifti).get_fdata()
        result_small = smoother_small.smooth(synthetic_nifti).get_fdata()
        result_large = smoother_large.smooth(synthetic_nifti).get_fdata()
        # Larger kernel should differ more from the original
        diff_small = np.sum(np.abs(input_data - result_small))
        diff_large = np.sum(np.abs(input_data - result_large))
        assert diff_large > diff_small


class TestUniformEdgeCase:
    """Test smoothing on a uniform image (edge case)."""

    def test_uniform_image_unchanged(self, uniform_nifti):
        """Smoothing a uniform image should produce identical data."""
        smoother = ImageSmoother()
        result = smoother.smooth(uniform_nifti)
        input_data = nib.load(uniform_nifti).get_fdata()
        output_data = result.get_fdata()
        np.testing.assert_array_almost_equal(input_data, output_data)


class TestCreateTestImage:
    """Test the test image creation function."""

    def test_creates_file(self, tmp_path):
        """create_test_image should create a NIfTI file on disk."""
        filepath = str(tmp_path / "created_test.nii.gz")
        create_test_image(filepath)
        assert os.path.exists(filepath)

    def test_created_image_is_valid_nifti(self, tmp_path):
        """The created file should be a loadable NIfTI image."""
        filepath = str(tmp_path / "created_test.nii.gz")
        create_test_image(filepath)
        img = nib.load(filepath)
        assert img.get_fdata().ndim == 3

    def test_created_image_is_not_empty(self, tmp_path):
        """The created test image should contain non-zero data."""
        filepath = str(tmp_path / "created_test.nii.gz")
        create_test_image(filepath)
        data = nib.load(filepath).get_fdata()
        assert np.any(data != 0)


class TestPlotFunction:
    """Test the plotting function for input vs output comparison."""

    def test_returns_figure(self, synthetic_nifti):
        """plot_input_vs_output should return a matplotlib Figure."""
        matplotlib.use("Agg")
        smoother = ImageSmoother()
        smoothed = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        fig = plot_input_vs_output(input_img, smoothed)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_figure_has_two_axes(self, synthetic_nifti):
        """The figure should have exactly two subplots (input and output)."""
        matplotlib.use("Agg")
        smoother = ImageSmoother()
        smoothed = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        fig = plot_input_vs_output(input_img, smoothed)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_axes_have_titles(self, synthetic_nifti):
        """Each subplot should have a descriptive title."""
        matplotlib.use("Agg")
        smoother = ImageSmoother()
        smoothed = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        fig = plot_input_vs_output(input_img, smoothed)
        axes = fig.get_axes()
        for ax in axes:
            assert ax.get_title() != ""
        plt.close(fig)

    def test_custom_slice_index(self, synthetic_nifti):
        """plot_input_vs_output should accept a custom slice index."""
        matplotlib.use("Agg")
        smoother = ImageSmoother()
        smoothed = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        fig = plot_input_vs_output(input_img, smoothed, slice_index=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_saves_to_file(self, synthetic_nifti, tmp_path):
        """plot_input_vs_output should optionally save to a file."""
        matplotlib.use("Agg")
        smoother = ImageSmoother()
        smoothed = smoother.smooth(synthetic_nifti)
        input_img = nib.load(synthetic_nifti)
        outpath = str(tmp_path / "comparison.png")
        fig = plot_input_vs_output(input_img, smoothed, save_path=outpath)
        assert os.path.exists(outpath)
        plt.close(fig)
