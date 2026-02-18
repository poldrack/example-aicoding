"""Markdown-to-PDF renderer with optional header image.

Converts a Markdown document to a PDF file using the ``markdown`` library
for Markdown-to-HTML conversion and ``fpdf2`` for HTML-to-PDF rendering.
An optional header image can be prepended to the document.

Functions
---------
markdown_to_html(markdown_text)
    Convert a Markdown string to an HTML string.
render_pdf(html_content, output_path, header_image=None)
    Render an HTML string to a PDF, optionally prepending a header image.
generate_pdf(markdown_path, output_path, header_image=None)
    Read a Markdown file, convert it to HTML, and render it as a PDF.
"""

import argparse
import os

import markdown
from fpdf import FPDF


def markdown_to_html(markdown_text: str) -> str:
    """Convert a Markdown string to HTML.

    Parameters
    ----------
    markdown_text : str
        The Markdown source text.

    Returns
    -------
    str
        The resulting HTML fragment.
    """
    return markdown.markdown(
        markdown_text,
        extensions=["fenced_code", "tables"],
    )


def render_pdf(
    html_content: str,
    output_path: str,
    header_image: str | None = None,
) -> None:
    """Render an HTML string to a PDF file.

    If *header_image* is supplied, the image is placed at the top of the
    first page before the HTML content.

    Parameters
    ----------
    html_content : str
        The HTML content to render.
    output_path : str
        Destination path for the generated PDF.
    header_image : str or None, optional
        Path to an image file to use as a header.  When ``None`` (the
        default), no header image is added.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if header_image is not None and os.path.exists(header_image):
        pdf.image(header_image, x=10, y=10, w=pdf.epw)
        pdf.ln(10)

    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html_content)
    pdf.output(output_path)


def generate_pdf(
    markdown_path: str,
    output_path: str,
    header_image: str | None = None,
) -> None:
    """Read a Markdown file, convert it to HTML, and render it as a PDF.

    Parameters
    ----------
    markdown_path : str
        Path to the input Markdown file.
    output_path : str
        Destination path for the generated PDF.
    header_image : str or None, optional
        Path to an image file to use as a document header.

    Raises
    ------
    FileNotFoundError
        If *markdown_path* does not exist.
    """
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(
            f"Markdown file not found: {markdown_path}"
        )

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    html_content = markdown_to_html(markdown_text)
    render_pdf(html_content, output_path, header_image=header_image)


if __name__ == "__main__":
    import sys
    import tempfile

    # When run without arguments, demonstrate the module with a sample
    if len(sys.argv) == 1:
        sample_md = "# Sample Document\n\nThis is a **demo** of the PDF generator.\n\n- Item 1\n- Item 2\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(sample_md)
            md_path = f.name

        os.makedirs("outputs", exist_ok=True)
        pdf_path = os.path.join("outputs", "demo_output.pdf")
        print("PDF Generation Demo")
        print("=" * 60)
        print(f"Input (temp): {md_path}")
        print(f"Output: {pdf_path}")
        generate_pdf(md_path, pdf_path)
        print(f"PDF generated successfully ({os.path.getsize(pdf_path)} bytes)")
        os.unlink(md_path)
    else:
        parser = argparse.ArgumentParser(
            description="Convert a Markdown document to PDF."
        )
        parser.add_argument(
            "--input",
            required=True,
            help="Path to the input Markdown file.",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Path for the output PDF file.",
        )
        parser.add_argument(
            "--header-image",
            default=None,
            help="Optional path to an image to use as a document header.",
        )

        args = parser.parse_args()
        generate_pdf(args.input, args.output, header_image=args.header_image)
