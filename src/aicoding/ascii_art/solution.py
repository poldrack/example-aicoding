"""Convert images to ASCII art.

Opens an image file, converts it to grayscale, resizes it to a target
width (maintaining aspect ratio adjusted for character height), and
maps pixel brightness values to ASCII characters.
"""

from PIL import Image

# ASCII characters ordered from darkest to lightest.
# Dense characters represent dark pixels; sparse characters represent light pixels.
ASCII_CHARS = "@%#*+=-:. "


def image_to_ascii(image_path, width=100):
    """Convert an image file to an ASCII art string.

    Args:
        image_path: Path to the image file (any format supported by Pillow).
        width: Number of characters per line in the output. Defaults to 100.

    Returns:
        A string containing the ASCII art representation, with lines
        separated by newline characters.
    """
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert("L")

    # Calculate new height preserving aspect ratio.
    # Characters are roughly twice as tall as they are wide,
    # so we halve the height to compensate.
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    height = max(1, int(width * aspect_ratio * 0.5))

    # Resize image to target dimensions
    img = img.resize((width, height))

    # Map each pixel to an ASCII character
    pixels = list(img.getdata())
    num_chars = len(ASCII_CHARS)

    lines = []
    for row in range(height):
        row_start = row * width
        row_pixels = pixels[row_start : row_start + width]
        line = ""
        for pixel in row_pixels:
            # Map pixel value (0-255) to index in ASCII_CHARS
            # 0 (black) -> first char (densest), 255 (white) -> last char (lightest)
            index = pixel * (num_chars - 1) // 255
            line += ASCII_CHARS[index]
        lines.append(line)

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)

    # Create a synthetic test image with a radial gradient
    test_width, test_height = 200, 150
    img = Image.new("RGB", (test_width, test_height))
    for x in range(test_width):
        for y in range(test_height):
            cx, cy = test_width / 2, test_height / 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            max_dist = (cx**2 + cy**2) ** 0.5
            brightness = int(255 * min(dist / max_dist, 1.0))
            img.putpixel((x, y), (brightness, brightness, brightness))

    # Save original image to outputs
    original_path = "outputs/ascii_art_original.png"
    img.save(original_path)
    print(f"Saved original image to {original_path}")
    print(f"Image size: {test_width}x{test_height}")

    # Convert to ASCII art
    ascii_art = image_to_ascii(original_path, width=80)
    print()
    print(ascii_art)

    # Plot original image alongside ASCII rendering
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.02, 0.98, ascii_art,
        transform=ax2.transAxes,
        fontsize=3, fontfamily="monospace",
        verticalalignment="top", color="black",
    )
    ax2.set_title("ASCII Art")
    ax2.axis("off")

    comparison_path = "outputs/ascii_art_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {comparison_path}")
