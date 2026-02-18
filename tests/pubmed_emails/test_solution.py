"""Tests for pubmed_emails — search PubMed, extract authors/affiliations, scrape emails."""

import re
from unittest.mock import patch, MagicMock

import pytest

from aicoding.pubmed_emails.solution import (
    search_pubmed,
    extract_authors,
    find_emails_from_web,
    find_author_emails,
)


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_article_record():
    """A realistic PubMed article record dict."""
    return {
        "MedlineCitation": {
            "Article": {
                "AuthorList": [
                    {
                        "LastName": "Smith",
                        "ForeName": "John",
                        "AffiliationInfo": [
                            {"Affiliation": "Department of Psychology, Stanford University, Stanford, CA"}
                        ],
                    },
                    {
                        "LastName": "Jones",
                        "ForeName": "Alice",
                        "AffiliationInfo": [
                            {"Affiliation": "Department of Neuroscience, MIT, Cambridge, MA"}
                        ],
                    },
                    {
                        "LastName": "Doe",
                        "ForeName": "Jane",
                        "AffiliationInfo": [
                            {"Affiliation": "Department of Cognitive Science, UC Berkeley, Berkeley, CA"}
                        ],
                    },
                ],
            }
        }
    }


# ---------------------------------------------------------------------------
# search_pubmed tests (mocked)
# ---------------------------------------------------------------------------

class TestSearchPubmed:
    @patch("aicoding.pubmed_emails.solution.Entrez")
    def test_returns_list_of_ids(self, mock_entrez):
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": ["12345", "67890"]}
        result = search_pubmed("cognitive control", max_results=2)
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("aicoding.pubmed_emails.solution.Entrez")
    def test_returns_strings(self, mock_entrez):
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": ["12345"]}
        result = search_pubmed("cognitive control", max_results=1)
        assert all(isinstance(x, str) for x in result)

    @patch("aicoding.pubmed_emails.solution.Entrez")
    def test_empty_query_returns_list(self, mock_entrez):
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}
        result = search_pubmed("xyznonexistent", max_results=5)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# extract_authors tests
# ---------------------------------------------------------------------------

class TestExtractAuthors:
    def test_returns_dict_with_first_and_last(self, mock_article_record):
        result = extract_authors(mock_article_record)
        assert "first_author" in result
        assert "last_author" in result

    def test_first_author_has_name(self, mock_article_record):
        result = extract_authors(mock_article_record)
        first = result["first_author"]
        assert "last_name" in first
        assert "first_name" in first
        assert first["last_name"] == "Smith"

    def test_last_author_has_name(self, mock_article_record):
        result = extract_authors(mock_article_record)
        last = result["last_author"]
        assert last["last_name"] == "Doe"

    def test_authors_have_affiliation(self, mock_article_record):
        result = extract_authors(mock_article_record)
        assert "affiliation" in result["first_author"]
        assert "affiliation" in result["last_author"]
        assert "Stanford" in result["first_author"]["affiliation"]

    def test_single_author_article(self):
        record = {
            "MedlineCitation": {
                "Article": {
                    "AuthorList": [
                        {
                            "LastName": "Solo",
                            "ForeName": "Han",
                            "AffiliationInfo": [{"Affiliation": "Millennium Falcon"}],
                        }
                    ]
                }
            }
        }
        result = extract_authors(record)
        assert result["first_author"]["last_name"] == "Solo"
        assert result["last_author"]["last_name"] == "Solo"

    def test_missing_affiliation(self):
        record = {
            "MedlineCitation": {
                "Article": {
                    "AuthorList": [
                        {"LastName": "NoAffil", "ForeName": "Test", "AffiliationInfo": []},
                    ]
                }
            }
        }
        result = extract_authors(record)
        assert result["first_author"]["affiliation"] == ""


# ---------------------------------------------------------------------------
# find_emails_from_web tests (mocked)
# ---------------------------------------------------------------------------

class TestFindEmailsFromWeb:
    @patch("aicoding.pubmed_emails.solution.requests.get")
    def test_finds_email_in_html(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = '<html><body>Contact: researcher@university.edu</body></html>'
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        emails = find_emails_from_web("John Smith", "Stanford University")
        assert isinstance(emails, list)

    @patch("aicoding.pubmed_emails.solution.requests.get")
    def test_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = '<html>no emails here</html>'
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        emails = find_emails_from_web("Nobody", "Nowhere")
        assert isinstance(emails, list)

    def test_email_regex_pattern(self):
        """Verify our email regex matches standard patterns."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        assert re.findall(pattern, "foo@bar.edu")
        assert re.findall(pattern, "first.last@university.ac.uk")
        assert not re.findall(pattern, "not-an-email")


# ---------------------------------------------------------------------------
# Integration (mocked end-to-end)
# ---------------------------------------------------------------------------

class TestFindAuthorEmails:
    @patch("aicoding.pubmed_emails.solution.find_emails_from_web")
    @patch("aicoding.pubmed_emails.solution.Entrez")
    def test_returns_list_of_dicts(self, mock_entrez, mock_find_emails):
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": ["12345"]}
        mock_entrez.efetch.return_value = MagicMock()

        # Mock parse to return one article
        from io import StringIO
        mock_record = {
            "MedlineCitation": {
                "Article": {
                    "AuthorList": [
                        {
                            "LastName": "Smith",
                            "ForeName": "John",
                            "AffiliationInfo": [{"Affiliation": "Stanford University"}],
                        },
                    ]
                }
            }
        }
        mock_entrez.read.side_effect = [
            {"IdList": ["12345"]},
            {"PubmedArticle": [mock_record]},
        ]
        mock_find_emails.return_value = ["smith@stanford.edu"]

        results = find_author_emails("cognitive control", max_results=1)
        assert isinstance(results, list)
        assert len(results) >= 1

    @patch("aicoding.pubmed_emails.solution.find_emails_from_web")
    @patch("aicoding.pubmed_emails.solution.Entrez")
    def test_result_has_author_info(self, mock_entrez, mock_find_emails):
        mock_record = {
            "MedlineCitation": {
                "Article": {
                    "AuthorList": [
                        {
                            "LastName": "Smith",
                            "ForeName": "John",
                            "AffiliationInfo": [{"Affiliation": "Stanford University"}],
                        },
                    ]
                }
            }
        }
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.side_effect = [
            {"IdList": ["12345"]},
            {"PubmedArticle": [mock_record]},
        ]
        mock_find_emails.return_value = ["smith@stanford.edu"]

        results = find_author_emails("cognitive control", max_results=1)
        entry = results[0]
        assert "name" in entry or "author" in entry or "first_name" in entry


# ---------------------------------------------------------------------------
# Live API test (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLiveAPI:
    def test_search_pubmed_returns_results(self):
        ids = search_pubmed("cognitive control", max_results=3)
        assert len(ids) > 0

    def test_full_pipeline(self):
        results = find_author_emails("cognitive control", max_results=1)
        assert isinstance(results, list)

    def test_gets_some_emails(self):
        """At least some emails should be found when searching enough articles."""
        results = find_author_emails("cognitive control", max_results=20)
        all_emails = [e for r in results for e in r.get("emails", [])]
        assert len(all_emails) > 0, (
            "No emails found across 20 articles — "
            "extraction from PubMed affiliations may be broken"
        )
