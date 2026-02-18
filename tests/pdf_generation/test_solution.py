"""Tests for the pdf_generation module (using fpdf2)."""

import os
import subprocess
import sys
import tempfile

import pytest
from PIL import Image

from aicoding.pdf_generation.solution import (
    generate_pdf,
    markdown_to_html,
    render_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_markdown():
    """Return a simple Markdown string."""
    return "# Hello World\n\nThis is a **bold** paragraph.\n"


@pytest.fixture
def complex_markdown():
    """Return a Markdown string with multiple element types."""
    return (
        "# Title\n\n"
        "## Subtitle\n\n"
        "A paragraph with *italic* and **bold** text.\n\n"
        "- item one\n"
        "- item two\n"
        "- item three\n\n"
        "```python\nprint('hello')\n```\n"
    )


@pytest.fixture
def markdown_file(simple_markdown):
    """Write simple markdown to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(simple_markdown)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def complex_markdown_file(complex_markdown):
    """Write complex markdown to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write(complex_markdown)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def header_image():
    """Create a small synthetic PNG image for use as a header."""
    img = Image.new("RGB", (200, 50), color=(30, 100, 200))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def output_pdf():
    """Provide a temporary path for PDF output and clean up afterwards."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    if os.path.exists(path):
        os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests for markdown_to_html
# ---------------------------------------------------------------------------

class TestMarkdownToHtml:
    def test_returns_string(self, simple_markdown):
        result = markdown_to_html(simple_markdown)
        assert isinstance(result, str)

    def test_heading_converted(self, simple_markdown):
        result = markdown_to_html(simple_markdown)
        assert "<h1>" in result or "<h1 " in result

    def test_bold_converted(self, simple_markdown):
        result = markdown_to_html(simple_markdown)
        assert "<strong>" in result

    def test_empty_markdown_returns_empty_or_whitespace(self):
        result = markdown_to_html("")
        assert result.strip() == ""

    def test_complex_markdown_has_list(self, complex_markdown):
        result = markdown_to_html(complex_markdown)
        assert "<li>" in result

    def test_complex_markdown_has_code(self, complex_markdown):
        result = markdown_to_html(complex_markdown)
        assert "<code>" in result or "<pre>" in result

    def test_italic_converted(self):
        result = markdown_to_html("This is *italic* text.")
        assert "<em>" in result

    def test_paragraph_wrapped(self):
        result = markdown_to_html("Just a paragraph.")
        assert "<p>" in result


# ---------------------------------------------------------------------------
# Tests for render_pdf
# ---------------------------------------------------------------------------

class TestRenderPdf:
    def test_creates_pdf_file(self, output_pdf):
        html = "<h1>Test</h1><p>Hello world.</p>"
        render_pdf(html, output_pdf)
        assert os.path.exists(output_pdf)

    def test_pdf_has_valid_magic_bytes(self, output_pdf):
        html = "<p>Check magic bytes</p>"
        render_pdf(html, output_pdf)
        with open(output_pdf, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_pdf_file_not_empty(self, output_pdf):
        html = "<p>Not empty</p>"
        render_pdf(html, output_pdf)
        assert os.path.getsize(output_pdf) > 0

    def test_render_with_header_image(self, output_pdf, header_image):
        html = "<p>With header</p>"
        render_pdf(html, output_pdf, header_image=header_image)
        assert os.path.exists(output_pdf)
        with open(output_pdf, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_pdf_with_header_image_is_larger(self, output_pdf, header_image):
        html = "<p>Content here</p>"
        render_pdf(html, output_pdf)
        size_without = os.path.getsize(output_pdf)

        render_pdf(html, output_pdf, header_image=header_image)
        size_with = os.path.getsize(output_pdf)

        assert size_with > size_without


# ---------------------------------------------------------------------------
# Tests for generate_pdf (end-to-end)
# ---------------------------------------------------------------------------

class TestGeneratePdf:
    def test_generates_pdf_from_markdown_file(self, markdown_file, output_pdf):
        generate_pdf(markdown_file, output_pdf)
        assert os.path.exists(output_pdf)
        with open(output_pdf, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_generates_pdf_with_header_image(
        self, markdown_file, output_pdf, header_image
    ):
        generate_pdf(markdown_file, output_pdf, header_image=header_image)
        assert os.path.exists(output_pdf)
        with open(output_pdf, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_raises_on_missing_markdown_file(self, output_pdf):
        with pytest.raises(FileNotFoundError):
            generate_pdf("/nonexistent/path/file.md", output_pdf)

    def test_complex_markdown_generates_pdf(
        self, complex_markdown_file, output_pdf
    ):
        generate_pdf(complex_markdown_file, output_pdf)
        assert os.path.exists(output_pdf)
        assert os.path.getsize(output_pdf) > 0


class TestGeneratePdfEdgeCases:
    def test_raises_on_missing_markdown_file(self, output_pdf):
        with pytest.raises(FileNotFoundError):
            generate_pdf("/nonexistent/path/file.md", output_pdf)


# ---------------------------------------------------------------------------
# Tests for the argparse CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_produces_pdf(self, markdown_file, output_pdf):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aicoding.pdf_generation.solution",
                "--input",
                markdown_file,
                "--output",
                output_pdf,
            ],
            capture_output=True,
            text=True,
            cwd="/Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-aicoding",
        )
        assert result.returncode == 0, (
            f"CLI failed with stderr: {result.stderr}"
        )
        assert os.path.exists(output_pdf)
        with open(output_pdf, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_cli_with_header_image(
        self, markdown_file, output_pdf, header_image
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aicoding.pdf_generation.solution",
                "--input",
                markdown_file,
                "--output",
                output_pdf,
                "--header-image",
                header_image,
            ],
            capture_output=True,
            text=True,
            cwd="/Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-aicoding",
        )
        assert result.returncode == 0, (
            f"CLI failed with stderr: {result.stderr}"
        )
        assert os.path.exists(output_pdf)


class TestCliEdgeCases:
    def test_cli_no_args_runs_demo(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aicoding.pdf_generation.solution",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-aicoding",
        )
        assert result.returncode == 0
        assert "PDF" in result.stdout
