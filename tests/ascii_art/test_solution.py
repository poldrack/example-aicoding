"""Tests for the ascii_art module."""

import os
import tempfile

import pytest
from PIL import Image

from aicoding.ascii_art.solution import image_to_ascii


@pytest.fixture
def synthetic_image():
    """Create a synthetic test image (gradient from black to white)."""
    width, height = 80, 60
    img = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            # Horizontal gradient: black on left, white on right
            gray = int(255 * x / (width - 1))
            img.putpixel((x, y), (gray, gray, gray))

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmpfile.name)
    yield tmpfile.name
    os.unlink(tmpfile.name)


@pytest.fixture
def tiny_image():
    """Create a 1x1 pixel image (edge case)."""
    img = Image.new("RGB", (1, 1), color=(128, 128, 128))
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmpfile.name)
    yield tmpfile.name
    os.unlink(tmpfile.name)


@pytest.fixture
def black_image():
    """Create an all-black image."""
    img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmpfile.name)
    yield tmpfile.name
    os.unlink(tmpfile.name)


@pytest.fixture
def white_image():
    """Create an all-white image."""
    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmpfile.name)
    yield tmpfile.name
    os.unlink(tmpfile.name)


class TestImageToAsciiExists:
    """Test that image_to_ascii function exists and returns correct type."""

    def test_returns_string(self, synthetic_image):
        """image_to_ascii should return a string."""
        result = image_to_ascii(synthetic_image)
        assert isinstance(result, str)

    def test_returns_nonempty_string(self, synthetic_image):
        """image_to_ascii should return a non-empty string."""
        result = image_to_ascii(synthetic_image)
        assert len(result) > 0


class TestAsciiOutput:
    """Test properties of the ASCII art output."""

    def test_contains_only_valid_ascii(self, synthetic_image):
        """Output should contain only printable ASCII characters and newlines."""
        result = image_to_ascii(synthetic_image)
        for char in result:
            assert char == "\n" or (32 <= ord(char) <= 126), (
                f"Invalid character found: {repr(char)} (ord={ord(char)})"
            )

    def test_output_has_expected_line_count(self, synthetic_image):
        """Output should have a number of lines based on image height."""
        result = image_to_ascii(synthetic_image, width=40)
        lines = result.rstrip("\n").split("\n")
        # There should be at least 1 line and lines should be > 0
        assert len(lines) >= 1
        # Each line should have the same width
        line_lengths = {len(line) for line in lines}
        assert len(line_lengths) == 1, (
            f"All lines should have the same width, got: {line_lengths}"
        )

    def test_line_width_matches_requested(self, synthetic_image):
        """Each line should have the requested width."""
        target_width = 60
        result = image_to_ascii(synthetic_image, width=target_width)
        lines = result.rstrip("\n").split("\n")
        for line in lines:
            assert len(line) == target_width


class TestWithSyntheticImage:
    """Test with synthetic images of known properties."""

    def test_gradient_produces_varied_characters(self, synthetic_image):
        """A gradient image should produce multiple distinct ASCII characters."""
        result = image_to_ascii(synthetic_image, width=80)
        # Remove newlines and count unique characters
        chars = set(result.replace("\n", ""))
        # A gradient should map to multiple different characters
        assert len(chars) >= 3, (
            f"Expected at least 3 distinct characters, got {len(chars)}: {chars}"
        )

    def test_default_width(self, synthetic_image):
        """Default width should be 100."""
        result = image_to_ascii(synthetic_image)
        lines = result.rstrip("\n").split("\n")
        assert len(lines[0]) == 100


class TestEdgeCases:
    """Test edge cases."""

    def test_tiny_image_1x1(self, tiny_image):
        """A 1x1 image should still produce valid ASCII output."""
        result = image_to_ascii(tiny_image, width=1)
        assert isinstance(result, str)
        assert len(result.strip()) >= 1
        # Should be a single character (possibly with newline)
        lines = result.rstrip("\n").split("\n")
        assert len(lines) >= 1

    def test_tiny_image_with_default_width(self, tiny_image):
        """A 1x1 image should work even with the default width."""
        result = image_to_ascii(tiny_image)
        assert isinstance(result, str)
        assert len(result.strip()) >= 1

    def test_different_brightness_different_chars(self, black_image, white_image):
        """Black and white images should map to different ASCII characters."""
        black_result = image_to_ascii(black_image, width=10)
        white_result = image_to_ascii(white_image, width=10)

        # Get the unique non-newline characters from each
        black_chars = set(black_result.replace("\n", ""))
        white_chars = set(white_result.replace("\n", ""))

        # Each should use exactly one character (uniform image)
        assert len(black_chars) == 1, (
            f"Black image should use 1 char, got {black_chars}"
        )
        assert len(white_chars) == 1, (
            f"White image should use 1 char, got {white_chars}"
        )

        # And they should be different from each other
        assert black_chars != white_chars, (
            "Black and white images should map to different ASCII characters"
        )

    def test_small_width(self, synthetic_image):
        """Should handle very small target widths."""
        result = image_to_ascii(synthetic_image, width=5)
        lines = result.rstrip("\n").split("\n")
        assert len(lines[0]) == 5
