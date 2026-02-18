"""Spatial smoothing on NIfTI images using a median filter.

Provides an ImageSmoother class that applies scipy.ndimage.median_filter
to 3D NIfTI brain images, a helper to create synthetic test images,
and a plotting function to compare input vs. smoothed output side by side.
"""

import os
import tempfile

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter


class ImageSmoother:
    """Apply spatial smoothing to NIfTI images using a median filter.

    Attributes:
        size: The kernel size for the median filter (applied uniformly
              across all spatial dimensions).
    """

    def __init__(self, size=3):
        """Initialize the ImageSmoother.

        Args:
            size: Kernel size for the median filter. Must be a positive
                  odd integer. Defaults to 3.
        """
        self.size = size

    def smooth(self, image):
        """Apply median filter smoothing to a NIfTI image.

        Args:
            image: Either a file path (str) to a NIfTI image or a
                   nibabel Nifti1Image object.

        Returns:
            A new nibabel Nifti1Image containing the smoothed data,
            with the same affine and header as the input.
        """
        if isinstance(image, str):
            img = nib.load(image)
        else:
            img = image

        data = img.get_fdata()
        smoothed_data = median_filter(data, size=self.size)
        smoothed_img = nib.Nifti1Image(smoothed_data, img.affine, img.header)
        return smoothed_img


def create_test_image(filepath, shape=(64, 64, 64), seed=42):
    """Create a synthetic NIfTI image for testing and demonstration.

    Generates a 3D image with a bright sphere embedded in a noisy
    background, providing clear structure for smoothing demonstrations.

    Args:
        filepath: Path where the NIfTI file will be saved.
        shape: Tuple of (x, y, z) dimensions. Defaults to (64, 64, 64).
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        The file path of the created image.
    """
    rng = np.random.default_rng(seed)

    # Create background noise
    data = rng.normal(loc=10, scale=5, size=shape).astype(np.float64)

    # Add a bright sphere in the center
    center = np.array(shape) / 2.0
    coords = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    distance = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    radius = min(shape) / 4.0
    sphere_mask = distance <= radius
    data[sphere_mask] = 100.0

    # Add salt-and-pepper noise for the median filter to clean
    noise_mask = rng.random(shape) < 0.02
    data[noise_mask] = rng.choice([0.0, 200.0], size=noise_mask.sum())

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)
    return filepath


def plot_input_vs_output(input_img, output_img, slice_index=None,
                         save_path=None):
    """Plot one axial slice from the input image next to the smoothed output.

    Args:
        input_img: nibabel Nifti1Image of the original (unsmoothed) image.
        output_img: nibabel Nifti1Image of the smoothed image.
        slice_index: Index of the axial slice to display. If None, uses
                     the middle slice. Defaults to None.
        save_path: If provided, saves the figure to this file path.
                   Defaults to None (no save).

    Returns:
        The matplotlib Figure object.
    """
    input_data = input_img.get_fdata()
    output_data = output_img.get_fdata()

    if slice_index is None:
        slice_index = input_data.shape[2] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(input_data[:, :, slice_index].T, cmap="gray", origin="lower")
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(output_data[:, :, slice_index].T, cmap="gray", origin="lower")
    ax2.set_title("Smoothed Output")
    ax2.axis("off")

    fig.suptitle(f"Median Filter Smoothing (axial slice {slice_index})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, "test_image.nii.gz")
    output_path = os.path.join(tmpdir, "smoothed_image.nii.gz")
    plot_path = "outputs/imgsmooth_comparison.png"

    print("Creating test NIfTI image...")
    create_test_image(input_path)

    print("Loading and smoothing the image...")
    smoother = ImageSmoother(size=3)
    input_img = nib.load(input_path)
    smoothed_img = smoother.smooth(input_img)

    # Save the smoothed image
    nib.save(smoothed_img, output_path)
    print(f"Smoothed image saved to: {output_path}")

    # Plot comparison
    print("Generating comparison plot...")
    fig = plot_input_vs_output(input_img, smoothed_img, save_path=plot_path)
    print(f"Comparison plot saved to: {plot_path}")

    plt.close(fig)
