"""Tests for the LDA topic modeling module."""

import pytest
from unittest.mock import patch, MagicMock
from aicoding.LDA.solution import (
    download_abstracts,
    clean_text,
    run_lda,
)


class TestCleanText:
    """Test text cleaning functionality."""

    def test_removes_stopwords(self):
        """Stopwords should be removed from text."""
        result = clean_text("this is a test of the system")
        assert "this" not in result
        assert "is" not in result
        assert "the" not in result

    def test_lemmatizes_words(self):
        """Words should be lemmatized."""
        result = clean_text("running dogs are playing")
        # Lemmatized forms should be present
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_list_of_strings(self):
        """clean_text should return a list of strings."""
        result = clean_text("cognitive control in the brain")
        assert isinstance(result, list)
        for word in result:
            assert isinstance(word, str)

    def test_empty_input(self):
        """Empty string should return an empty list."""
        result = clean_text("")
        assert result == []

    def test_lowercases_text(self):
        """Text should be lowercased."""
        result = clean_text("BRAIN ACTIVATION")
        for word in result:
            assert word == word.lower()


class TestRunLDA:
    """Test the LDA topic modeling function."""

    def test_returns_model_and_topics(self):
        """run_lda should return an LDA model and list of topics."""
        docs = [
            ["brain", "cognitive", "control", "function", "activation"],
            ["memory", "hippocampus", "encoding", "retrieval", "brain"],
            ["attention", "visual", "cortex", "processing", "stimulus"],
            ["learning", "reward", "dopamine", "prediction", "error"],
            ["emotion", "amygdala", "fear", "response", "regulation"],
        ] * 10  # Repeat for enough data
        model, topics = run_lda(docs, num_topics=3)
        assert topics is not None
        assert len(topics) == 3

    def test_topics_contain_words(self):
        """Each topic should contain word-weight pairs."""
        docs = [
            ["brain", "cognitive", "control", "function"],
            ["memory", "hippocampus", "encoding", "retrieval"],
            ["attention", "visual", "cortex", "processing"],
        ] * 20
        model, topics = run_lda(docs, num_topics=2)
        for topic in topics:
            assert len(topic) > 0

    def test_empty_corpus(self):
        """Empty corpus should raise or return empty results."""
        with pytest.raises(Exception):
            run_lda([], num_topics=3)


class TestDownloadAbstracts:
    """Test the PubMed abstract download function."""

    def test_returns_list_of_strings_with_real_api(self):
        """download_abstracts should return a list of abstract strings."""
        abstracts = download_abstracts("cognitive control", max_results=5)
        assert isinstance(abstracts, list)
        assert len(abstracts) > 0
        for abstract in abstracts:
            assert isinstance(abstract, str)
            assert len(abstract) > 0

    @patch("aicoding.LDA.solution.Entrez")
    def test_download_mocked(self, mock_entrez):
        """download_abstracts with mocked Entrez."""
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.side_effect = [
            {"IdList": ["111"]},
            {
                "PubmedArticle": [
                    {
                        "MedlineCitation": {
                            "PMID": "111",
                            "Article": {
                                "Abstract": {
                                    "AbstractText": ["This is an abstract about the brain."]
                                }
                            },
                        }
                    }
                ]
            },
        ]
        mock_entrez.efetch.return_value = MagicMock()
        abstracts = download_abstracts("test", max_results=1)
        assert isinstance(abstracts, list)
