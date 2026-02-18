"""Tests for the download_github_code module."""

import pytest
from unittest.mock import patch, MagicMock
from aicoding.download_github_code.solution import download_recent_python_files


class TestDownloadRecentPythonFiles:
    """Test the GitHub code download function."""

    def test_returns_list_with_real_api(self):
        """Function should return a list with real API (may be empty if rate-limited)."""
        result = download_recent_python_files(max_files=3)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "name" in item
            assert "url" in item
            assert "content" in item
            assert item["name"].endswith(".py")

    @patch("aicoding.download_github_code.solution.requests.get")
    def test_returns_dicts_with_required_keys(self, mock_get):
        """Each result should be a dict with name, url, and content keys."""
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "items": [
                {
                    "full_name": "user/repo",
                    "default_branch": "main",
                }
            ]
        }

        tree_resp = MagicMock()
        tree_resp.status_code = 200
        tree_resp.json.return_value = {
            "tree": [
                {"path": "src/test.py", "type": "blob"},
                {"path": "README.md", "type": "blob"},
            ]
        }

        raw_resp = MagicMock()
        raw_resp.status_code = 200
        raw_resp.text = 'print("hello")'

        mock_get.side_effect = [search_resp, tree_resp, raw_resp]
        result = download_recent_python_files(max_files=1)
        assert len(result) == 1
        assert result[0]["name"] == "test.py"
        assert result[0]["content"] == 'print("hello")'

    @patch("aicoding.download_github_code.solution.requests.get")
    def test_files_are_python(self, mock_get):
        """All returned files should be Python files (non-.py files filtered)."""
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "items": [
                {"full_name": "user/repo", "default_branch": "main"}
            ]
        }

        tree_resp = MagicMock()
        tree_resp.status_code = 200
        tree_resp.json.return_value = {
            "tree": [
                {"path": "app.py", "type": "blob"},
                {"path": "README.md", "type": "blob"},
                {"path": "data.json", "type": "blob"},
            ]
        }

        raw_resp = MagicMock()
        raw_resp.status_code = 200
        raw_resp.text = "pass"

        mock_get.side_effect = [search_resp, tree_resp, raw_resp]

        result = download_recent_python_files(max_files=5)
        assert len(result) == 1
        for item in result:
            assert item["name"].endswith(".py")

    @patch("aicoding.download_github_code.solution.requests.get")
    def test_respects_max_files(self, mock_get):
        """Should not return more files than max_files."""
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "items": [
                {"full_name": "user/repo", "default_branch": "main"}
            ]
        }

        tree_resp = MagicMock()
        tree_resp.status_code = 200
        tree_resp.json.return_value = {
            "tree": [
                {"path": f"file{i}.py", "type": "blob"}
                for i in range(10)
            ]
        }

        raw_resp = MagicMock()
        raw_resp.status_code = 200
        raw_resp.text = "pass"

        mock_get.side_effect = [search_resp, tree_resp, raw_resp, raw_resp]
        result = download_recent_python_files(max_files=2)
        assert len(result) <= 2

    @patch("aicoding.download_github_code.solution.requests.get")
    def test_handles_api_error(self, mock_get):
        """Should return empty list on API error."""
        error_resp = MagicMock()
        error_resp.status_code = 403
        mock_get.return_value = error_resp

        result = download_recent_python_files(max_files=5)
        assert result == []
