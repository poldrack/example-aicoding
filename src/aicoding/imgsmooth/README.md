# imgsmooth -- Spatial Smoothing on NIfTI Images

## Problem

Apply spatial smoothing to a NIfTI brain image using a median filter, create a synthetic test image for demonstration, and plot one slice from the input next to the corresponding smoothed output slice.

## Approach

- **ImageSmoother class** wraps `scipy.ndimage.median_filter` to apply 3D spatial median smoothing to NIfTI images loaded via `nibabel`. It accepts either a file path or a `Nifti1Image` object and returns a new `Nifti1Image` with the smoothed data while preserving the original affine and header.
- **`create_test_image`** generates a synthetic 3D NIfTI volume containing a bright sphere on a noisy background with salt-and-pepper noise, providing clear structure for demonstrating median filter behavior.
- **`plot_input_vs_output`** displays a side-by-side comparison of one axial slice from the original and smoothed images using matplotlib, with optional file saving.

## Key Design Decisions

- The median filter (rather than Gaussian) is used as specified in the problem prompt. Median filters are effective at removing salt-and-pepper noise while preserving edges.
- The `smooth()` method accepts both file paths and nibabel image objects for flexibility.
- The kernel size defaults to 3 but is configurable at instantiation.

## Dependencies

- `nibabel` -- NIfTI image I/O
- `scipy.ndimage` -- median filter implementation
- `matplotlib` -- visualization
- `numpy` -- array operations

## Usage

```python
from aicoding.imgsmooth.solution import ImageSmoother, create_test_image, plot_input_vs_output
import nibabel as nib

# Create a test image
create_test_image("test.nii.gz")

# Smooth it
smoother = ImageSmoother(size=3)
smoothed = smoother.smooth("test.nii.gz")

# Plot comparison
input_img = nib.load("test.nii.gz")
fig = plot_input_vs_output(input_img, smoothed, save_path="comparison.png")
```

Run as a standalone script:

```bash
python -m aicoding.imgsmooth.solution
```
