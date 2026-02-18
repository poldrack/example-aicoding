# pdf_generation

Convert a Markdown document to a PDF file, with an optional header image.

## Approach

1. **Markdown to HTML** -- The `markdown` library (with `fenced_code` and `tables` extensions) converts the input `.md` file into an HTML fragment.
2. **HTML to PDF** -- `fpdf2` renders the HTML content into a PDF using its built-in `write_html()` method. No system-level dependencies required.
3. **Header image** -- When a `--header-image` path is provided, the image is placed at the top of the first page before the HTML content.

## Usage

```bash
python -m aicoding.pdf_generation.solution \
    --input document.md \
    --output document.pdf \
    --header-image logo.png   # optional
```

### CLI arguments

| Argument         | Required | Description                          |
|------------------|----------|--------------------------------------|
| `--input`        | yes      | Path to the input Markdown file      |
| `--output`       | yes      | Path for the output PDF file         |
| `--header-image` | no       | Path to an image for the PDF header  |

## Dependencies

- `markdown` -- Markdown-to-HTML conversion.
- `fpdf2` -- HTML-to-PDF rendering (pure Python, no system dependencies).

## Functions

- `markdown_to_html(markdown_text)` -- Convert a Markdown string to an HTML string.
- `render_pdf(html_content, output_path, header_image=None)` -- Render HTML to PDF, optionally prepending a header image.
- `generate_pdf(markdown_path, output_path, header_image=None)` -- End-to-end: read a Markdown file, convert, and write a PDF.
