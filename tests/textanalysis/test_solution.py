"""Tests for textanalysis — sentiment analysis and linguistic complexity measures."""

import pytest

from aicoding.textanalysis.solution import analyze_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_text():
    """A short, positive, simple piece of text."""
    return "The cat sat on the mat. It was a good day."


@pytest.fixture
def complex_text():
    """A longer, more linguistically complex piece of text."""
    return (
        "Although the experiment was carefully designed, the results were "
        "inconclusive because several confounding variables had not been "
        "adequately controlled. Nevertheless, the researchers published "
        "their findings, which subsequently generated considerable debate "
        "within the scientific community. Furthermore, the replication "
        "attempts that followed produced contradictory outcomes, suggesting "
        "that the original methodology was fundamentally flawed."
    )


@pytest.fixture
def negative_text():
    """A clearly negative piece of text."""
    return (
        "This is terrible and awful. I hate everything about this horrible "
        "situation. The worst outcome is unavoidable and deeply disappointing."
    )


@pytest.fixture
def positive_text():
    """A clearly positive piece of text."""
    return (
        "This is wonderful and amazing! I love everything about this beautiful "
        "experience. The best outcome has been achieved and I am truly happy."
    )


@pytest.fixture
def single_sentence():
    """A single sentence for edge-case testing."""
    return "The quick brown fox jumps over the lazy dog."


# ---------------------------------------------------------------------------
# Test 1: Return type and required keys
# ---------------------------------------------------------------------------

class TestReturnStructure:
    """analyze_text must return a dict with all required keys."""

    REQUIRED_KEYS = [
        "sentiment",
        "word_count",
        "sentence_count",
        "mean_sentence_length",
        "type_token_ratio",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "mean_t_unit_length",
        "clauses_per_t_unit",
        "content_word_frequency",
        "connective_count",
    ]

    def test_returns_dict(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, simple_text):
        result = analyze_text(simple_text)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_no_none_values(self, simple_text):
        result = analyze_text(simple_text)
        for key in self.REQUIRED_KEYS:
            assert result[key] is not None, f"Key '{key}' is None"


# ---------------------------------------------------------------------------
# Test 2: Sentiment analysis
# ---------------------------------------------------------------------------

class TestSentiment:
    """Sentiment should be a dict with polarity and subjectivity."""

    def test_sentiment_is_dict(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["sentiment"], dict)

    def test_sentiment_has_polarity(self, simple_text):
        result = analyze_text(simple_text)
        assert "polarity" in result["sentiment"]

    def test_sentiment_has_subjectivity(self, simple_text):
        result = analyze_text(simple_text)
        assert "subjectivity" in result["sentiment"]

    def test_polarity_range(self, simple_text):
        result = analyze_text(simple_text)
        polarity = result["sentiment"]["polarity"]
        assert -1.0 <= polarity <= 1.0

    def test_subjectivity_range(self, simple_text):
        result = analyze_text(simple_text)
        subjectivity = result["sentiment"]["subjectivity"]
        assert 0.0 <= subjectivity <= 1.0

    def test_negative_text_has_negative_polarity(self, negative_text):
        result = analyze_text(negative_text)
        assert result["sentiment"]["polarity"] < 0

    def test_positive_text_has_positive_polarity(self, positive_text):
        result = analyze_text(positive_text)
        assert result["sentiment"]["polarity"] > 0


# ---------------------------------------------------------------------------
# Test 3: Word count
# ---------------------------------------------------------------------------

class TestWordCount:
    """word_count should reflect the number of words in the text."""

    def test_word_count_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["word_count"], int)

    def test_word_count_simple_text(self, simple_text):
        result = analyze_text(simple_text)
        # "The cat sat on the mat. It was a good day." = 11 words
        assert result["word_count"] == 11

    def test_word_count_positive(self, complex_text):
        result = analyze_text(complex_text)
        assert result["word_count"] > 0


# ---------------------------------------------------------------------------
# Test 4: Sentence count
# ---------------------------------------------------------------------------

class TestSentenceCount:
    """sentence_count should reflect the number of sentences."""

    def test_sentence_count_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["sentence_count"], int)

    def test_sentence_count_simple_text(self, simple_text):
        result = analyze_text(simple_text)
        # Two sentences
        assert result["sentence_count"] == 2

    def test_single_sentence_count(self, single_sentence):
        result = analyze_text(single_sentence)
        assert result["sentence_count"] == 1


# ---------------------------------------------------------------------------
# Test 5: Mean sentence length
# ---------------------------------------------------------------------------

class TestMeanSentenceLength:
    """mean_sentence_length = word_count / sentence_count."""

    def test_mean_sentence_length_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["mean_sentence_length"], float)

    def test_mean_sentence_length_value(self, simple_text):
        result = analyze_text(simple_text)
        expected = result["word_count"] / result["sentence_count"]
        assert result["mean_sentence_length"] == pytest.approx(expected, rel=1e-6)

    def test_complex_text_longer_sentences(self, simple_text, complex_text):
        simple_result = analyze_text(simple_text)
        complex_result = analyze_text(complex_text)
        assert complex_result["mean_sentence_length"] > simple_result["mean_sentence_length"]


# ---------------------------------------------------------------------------
# Test 6: Type-token ratio
# ---------------------------------------------------------------------------

class TestTypeTokenRatio:
    """type_token_ratio = unique_words / total_words (case-insensitive)."""

    def test_ttr_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["type_token_ratio"], float)

    def test_ttr_range(self, simple_text):
        result = analyze_text(simple_text)
        assert 0.0 < result["type_token_ratio"] <= 1.0

    def test_ttr_all_unique(self):
        text = "apple banana cherry."
        result = analyze_text(text)
        assert result["type_token_ratio"] == pytest.approx(1.0)

    def test_ttr_with_repeats(self):
        text = "the the the the the."
        result = analyze_text(text)
        assert result["type_token_ratio"] == pytest.approx(1.0 / 5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Flesch readability measures
# ---------------------------------------------------------------------------

class TestReadability:
    """Flesch Reading Ease and Flesch-Kincaid Grade Level."""

    def test_flesch_reading_ease_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["flesch_reading_ease"], float)

    def test_flesch_kincaid_grade_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["flesch_kincaid_grade"], float)

    def test_simple_text_easier_than_complex(self, simple_text, complex_text):
        simple_result = analyze_text(simple_text)
        complex_result = analyze_text(complex_text)
        # Higher Flesch Reading Ease = easier to read
        assert simple_result["flesch_reading_ease"] > complex_result["flesch_reading_ease"]

    def test_simple_text_lower_grade_than_complex(self, simple_text, complex_text):
        simple_result = analyze_text(simple_text)
        complex_result = analyze_text(complex_text)
        # Higher Flesch-Kincaid Grade = harder to read
        assert simple_result["flesch_kincaid_grade"] < complex_result["flesch_kincaid_grade"]


# ---------------------------------------------------------------------------
# Test 8: L2SCA measures
# ---------------------------------------------------------------------------

class TestL2SCA:
    """L2SCA-style measures: mean T-unit length and clauses per T-unit."""

    def test_mean_t_unit_length_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["mean_t_unit_length"], float)

    def test_clauses_per_t_unit_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["clauses_per_t_unit"], float)

    def test_mean_t_unit_length_positive(self, simple_text):
        result = analyze_text(simple_text)
        assert result["mean_t_unit_length"] > 0

    def test_clauses_per_t_unit_at_least_one(self, simple_text):
        result = analyze_text(simple_text)
        # Each T-unit has at least one clause
        assert result["clauses_per_t_unit"] >= 1.0

    def test_complex_text_more_clauses_per_t_unit(self, simple_text, complex_text):
        simple_result = analyze_text(simple_text)
        complex_result = analyze_text(complex_text)
        # Complex text with subordinate clauses should have higher clauses per T-unit
        assert complex_result["clauses_per_t_unit"] > simple_result["clauses_per_t_unit"]


# ---------------------------------------------------------------------------
# Test 9: Coh-Metrix style measures
# ---------------------------------------------------------------------------

class TestCohMetrix:
    """Coh-Metrix style: content word frequency and connective count."""

    def test_content_word_frequency_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["content_word_frequency"], float)

    def test_content_word_frequency_positive(self, simple_text):
        result = analyze_text(simple_text)
        assert result["content_word_frequency"] > 0

    def test_connective_count_type(self, simple_text):
        result = analyze_text(simple_text)
        assert isinstance(result["connective_count"], int)

    def test_connective_count_non_negative(self, simple_text):
        result = analyze_text(simple_text)
        assert result["connective_count"] >= 0

    def test_complex_text_has_connectives(self, complex_text):
        result = analyze_text(complex_text)
        # The complex text contains connectives like "although", "because",
        # "nevertheless", "furthermore"
        assert result["connective_count"] >= 3


# ---------------------------------------------------------------------------
# Test 10: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for analyze_text."""

    def test_single_word(self):
        result = analyze_text("Hello.")
        assert result["word_count"] == 1
        assert result["sentence_count"] == 1
        assert result["mean_sentence_length"] == pytest.approx(1.0)
        assert result["type_token_ratio"] == pytest.approx(1.0)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            analyze_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            analyze_text("   ")

    def test_numeric_value_raises(self):
        with pytest.raises(TypeError):
            analyze_text(12345)

    def test_none_raises(self):
        with pytest.raises(TypeError):
            analyze_text(None)


# ---------------------------------------------------------------------------
# Test 11: Consistency between measures
# ---------------------------------------------------------------------------

class TestConsistency:
    """Cross-checks between related measures."""

    def test_mean_sentence_length_equals_ratio(self, complex_text):
        result = analyze_text(complex_text)
        expected = result["word_count"] / result["sentence_count"]
        assert result["mean_sentence_length"] == pytest.approx(expected, rel=1e-6)

    def test_ttr_bounded_by_word_count(self, complex_text):
        result = analyze_text(complex_text)
        # TTR * word_count = number of unique words, which must be <= word_count
        unique_count = result["type_token_ratio"] * result["word_count"]
        assert unique_count <= result["word_count"] + 1e-6
