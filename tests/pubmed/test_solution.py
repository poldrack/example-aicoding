"""Tests for the pubmed module."""

import pytest
from abc import ABC
from unittest.mock import patch, MagicMock
from aicoding.pubmed.solution import AbstractPublication, Article


# --- Unit tests (mocked) ---

class TestAbstractPublication:
    """Test the AbstractPublication abstract base class."""

    def test_is_abstract(self):
        """AbstractPublication should be an abstract base class."""
        assert issubclass(AbstractPublication, ABC)

    def test_cannot_instantiate(self):
        """Cannot instantiate AbstractPublication directly."""
        with pytest.raises(TypeError):
            AbstractPublication()

    def test_has_from_pubmed_method(self):
        """AbstractPublication declares a from_pubmed method."""
        assert hasattr(AbstractPublication, "from_pubmed")


class TestArticle:
    """Test the Article class."""

    def test_inherits_abstract_publication(self):
        """Article should inherit from AbstractPublication."""
        assert issubclass(Article, AbstractPublication)

    def test_from_pubmed_sets_authors(self):
        """from_pubmed should populate the authors attribute."""
        record = {
            "AuthorList": [
                {"LastName": "Smith", "ForeName": "John"},
                {"LastName": "Doe", "ForeName": "Jane"},
            ],
            "ArticleTitle": "Test Title",
        }
        article = Article()
        article.from_pubmed(record)
        assert hasattr(article, "authors")
        assert len(article.authors) == 2

    def test_from_pubmed_sets_title(self):
        """from_pubmed should populate the title attribute."""
        record = {
            "AuthorList": [{"LastName": "Smith", "ForeName": "John"}],
            "ArticleTitle": "A Study on Cognitive Control",
        }
        article = Article()
        article.from_pubmed(record)
        assert article.title == "A Study on Cognitive Control"

    def test_from_pubmed_empty_authors(self):
        """from_pubmed should handle an empty author list."""
        record = {
            "AuthorList": [],
            "ArticleTitle": "No Authors Paper",
        }
        article = Article()
        article.from_pubmed(record)
        assert article.authors == []

    def test_from_pubmed_returns_self(self):
        """from_pubmed should return the Article instance for chaining."""
        record = {
            "AuthorList": [{"LastName": "A", "ForeName": "B"}],
            "ArticleTitle": "Title",
        }
        article = Article()
        result = article.from_pubmed(record)
        assert result is article


class TestMainSearch:
    """Test the main PubMed search functionality."""

    @patch("aicoding.pubmed.solution.Entrez")
    def test_search_returns_dict(self, mock_entrez):
        """search_pubmed should return a dict keyed by PubMed ID."""
        from aicoding.pubmed.solution import search_pubmed

        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": ["123", "456"]}
        mock_entrez.efetch.return_value = MagicMock()
        mock_entrez.read.side_effect = [
            {"IdList": ["123", "456"]},
            {
                "PubmedArticle": [
                    {
                        "MedlineCitation": {
                            "PMID": "123",
                            "Article": {
                                "AuthorList": [{"LastName": "A", "ForeName": "B"}],
                                "ArticleTitle": "Title 1",
                            },
                        }
                    },
                    {
                        "MedlineCitation": {
                            "PMID": "456",
                            "Article": {
                                "AuthorList": [{"LastName": "C", "ForeName": "D"}],
                                "ArticleTitle": "Title 2",
                            },
                        }
                    },
                ]
            },
        ]

        result = search_pubmed("cognitive control", max_results=2)
        assert isinstance(result, dict)

    def test_search_returns_dict_with_real_api(self):
        """search_pubmed should work with the real PubMed API."""
        from aicoding.pubmed.solution import search_pubmed

        result = search_pubmed("cognitive control", max_results=3)
        assert isinstance(result, dict)
        assert len(result) > 0
        for pmid, article in result.items():
            assert isinstance(pmid, str)
            assert isinstance(article, Article)
            assert hasattr(article, "title")
            assert hasattr(article, "authors")
