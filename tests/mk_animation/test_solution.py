"""Tests for mk_animation — text scrolling animation saved as AVI."""

import os
import tempfile

import pytest

from aicoding.mk_animation.solution import create_text_animation


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for animation."""
    text = "\n".join([f"Line {i}: The quick brown fox jumps over the lazy dog." for i in range(50)])
    fpath = tmp_path / "sample.txt"
    fpath.write_text(text)
    return str(fpath)


class TestCreateTextAnimation:
    def test_function_exists(self):
        assert callable(create_text_animation)

    def test_creates_avi_file(self, sample_text_file, tmp_path):
        output_path = str(tmp_path / "output.avi")
        create_text_animation(
            sample_text_file,
            output_path=output_path,
            lines_per_frame=10,
            chars_per_line=40,
            scroll_lines=2,
            duration_seconds=2,
            fps=10,
            display=False,
        )
        assert os.path.exists(output_path)

    def test_output_file_not_empty(self, sample_text_file, tmp_path):
        output_path = str(tmp_path / "output.avi")
        create_text_animation(
            sample_text_file,
            output_path=output_path,
            lines_per_frame=10,
            chars_per_line=40,
            scroll_lines=2,
            duration_seconds=2,
            fps=10,
            display=False,
        )
        assert os.path.getsize(output_path) > 0

    def test_short_file(self, tmp_path):
        """A file shorter than lines_per_frame should still work."""
        fpath = tmp_path / "short.txt"
        fpath.write_text("Only one line.")
        output_path = str(tmp_path / "short.avi")
        create_text_animation(
            str(fpath),
            output_path=output_path,
            lines_per_frame=10,
            chars_per_line=40,
            scroll_lines=1,
            duration_seconds=1,
            fps=5,
            display=False,
        )
        assert os.path.exists(output_path)

    def test_long_lines_wrapped(self, tmp_path):
        """Lines longer than chars_per_line should be wrapped."""
        fpath = tmp_path / "long.txt"
        fpath.write_text("A" * 200 + "\n" + "B" * 200)
        output_path = str(tmp_path / "long.avi")
        create_text_animation(
            str(fpath),
            output_path=output_path,
            lines_per_frame=5,
            chars_per_line=20,
            scroll_lines=1,
            duration_seconds=1,
            fps=5,
            display=False,
        )
        assert os.path.exists(output_path)

    def test_custom_fps(self, sample_text_file, tmp_path):
        output_path = str(tmp_path / "custom_fps.avi")
        create_text_animation(
            sample_text_file,
            output_path=output_path,
            lines_per_frame=5,
            chars_per_line=30,
            scroll_lines=1,
            duration_seconds=1,
            fps=15,
            display=False,
        )
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_text_animation(
                "/nonexistent/file.txt",
                output_path=str(tmp_path / "out.avi"),
                lines_per_frame=10,
                chars_per_line=40,
                scroll_lines=1,
                duration_seconds=1,
                fps=10,
                display=False,
            )
