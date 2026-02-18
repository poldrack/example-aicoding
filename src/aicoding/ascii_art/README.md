# ASCII Art

Convert images to ASCII art using Python and Pillow.

## Overview

This module reads an image file and renders it as ASCII art by:

1. Converting the image to grayscale
2. Resizing to a target width while maintaining aspect ratio (adjusted for character height)
3. Mapping each pixel's brightness to an ASCII character from a dark-to-light gradient

The character set used (darkest to lightest):

```
@%#*+=-:.
```

## Usage

### As a module

```python
from aicoding.ascii_art.solution import image_to_ascii

# Convert an image to ASCII art with default width (100 chars)
ascii_art = image_to_ascii("photo.png")
print(ascii_art)

# Specify a custom width
ascii_art = image_to_ascii("photo.png", width=60)
print(ascii_art)
```

### From the command line

Running the module directly creates a synthetic radial gradient image and renders it:

```bash
python -m aicoding.ascii_art.solution
```

## API

### `image_to_ascii(image_path, width=100)`

- **image_path**: Path to any image file supported by Pillow (PNG, JPEG, BMP, etc.)
- **width**: Number of characters per line in the output (default: 100)
- **Returns**: A string containing the ASCII art, with lines separated by newlines

## Dependencies

- [Pillow](https://pillow.readthedocs.io/) (PIL) for image loading and manipulation

## Testing

```bash
pytest tests/ascii_art/ -v
```

Tests cover:
- Return type and non-empty output
- Valid ASCII characters only
- Consistent line widths matching the requested width
- Gradient images producing varied characters
- Edge cases: 1x1 images, very small widths, uniform brightness images
- Different brightness levels mapping to different characters
