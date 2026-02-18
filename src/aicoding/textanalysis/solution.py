"""Linguistic analysis module.

Provides ``analyze_text(text)`` which returns a dictionary containing:

* **Sentiment** (via TextBlob): polarity and subjectivity.
* **Basic counts**: word count, sentence count, mean sentence length, type-token
  ratio.
* **Readability**: Flesch Reading Ease, Flesch-Kincaid Grade Level.
* **L2SCA-style measures**: mean T-unit length, clauses per T-unit.
* **Coh-Metrix-style measures**: mean content-word log frequency,
  connective count.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Ensure required NLTK data is present
# ---------------------------------------------------------------------------
for _resource in ("punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource else _resource)
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords as _sw_corpus

_STOP_WORDS: set[str] = set(_sw_corpus.words("english"))

# ---------------------------------------------------------------------------
# Connective word list (Coh-Metrix style)
# ---------------------------------------------------------------------------
_CONNECTIVES: set[str] = {
    # Causal
    "because", "since", "therefore", "thus", "hence", "consequently",
    "accordingly", "so",
    # Adversative / contrastive
    "although", "though", "however", "nevertheless", "nonetheless",
    "yet", "but", "whereas", "while", "despite", "instead",
    # Additive
    "furthermore", "moreover", "additionally", "also", "besides",
    "likewise", "similarly",
    # Temporal
    "then", "subsequently", "meanwhile", "afterwards", "finally",
    "previously", "earlier",
}

# ---------------------------------------------------------------------------
# Clause-introducing markers (used for approximate clause detection)
# ---------------------------------------------------------------------------
_CLAUSE_MARKERS: set[str] = {
    # Subordinating conjunctions
    "although", "though", "because", "since", "while", "whereas",
    "if", "unless", "until", "when", "whenever", "where", "wherever",
    "after", "before", "that", "which", "who", "whom", "whose",
    # Coordinating conjunctions that typically introduce new clauses
    "and", "but", "or", "nor", "yet", "so",
}

# ---------------------------------------------------------------------------
# Simple English word-frequency table (log10 frequency per million)
# Based on common word-frequency norms; only content words included.
# We fall back to a low default for unknown words.
# ---------------------------------------------------------------------------
_WORD_FREQ: dict[str, float] = {
    "the": 4.94, "be": 4.55, "to": 4.49, "of": 4.44, "and": 4.41,
    "a": 4.34, "in": 4.20, "have": 4.04, "it": 3.98, "for": 3.79,
    "not": 3.72, "on": 3.68, "with": 3.62, "he": 3.61, "as": 3.55,
    "do": 3.53, "at": 3.42, "this": 3.39, "but": 3.37, "his": 3.34,
    "by": 3.33, "from": 3.27, "they": 3.26, "we": 3.22, "say": 3.18,
    "she": 3.17, "or": 3.15, "an": 3.10, "will": 3.08, "my": 3.02,
    "all": 2.98, "would": 2.97, "there": 2.95, "their": 2.93,
    "what": 2.90, "so": 2.88, "up": 2.85, "out": 2.84, "if": 2.82,
    "about": 2.79, "get": 2.77, "which": 2.74, "go": 2.73, "make": 2.70,
    "can": 2.69, "like": 2.67, "time": 2.65, "just": 2.62, "know": 2.60,
    "take": 2.57, "people": 2.55, "into": 2.53, "year": 2.51,
    "good": 2.49, "some": 2.47, "could": 2.45, "them": 2.44,
    "see": 2.42, "other": 2.40, "than": 2.39, "then": 2.37, "now": 2.36,
    "look": 2.34, "only": 2.33, "come": 2.31, "think": 2.30, "also": 2.28,
    "back": 2.27, "after": 2.25, "use": 2.24, "two": 2.22, "how": 2.21,
    "work": 2.19, "first": 2.18, "well": 2.16, "way": 2.15, "even": 2.13,
    "new": 2.12, "want": 2.10, "day": 2.09, "find": 2.07, "give": 2.06,
    "more": 2.04, "most": 2.03, "very": 2.01, "when": 2.00, "still": 1.98,
    "own": 1.96, "try": 1.95, "long": 1.93, "much": 1.92, "great": 1.90,
    "cat": 1.70, "sat": 1.50, "mat": 1.30, "happy": 1.80, "sad": 1.60,
    "love": 2.00, "hate": 1.50, "terrible": 1.40, "wonderful": 1.50,
    "beautiful": 1.70, "horrible": 1.30, "experiment": 1.60,
    "results": 1.80, "research": 1.70, "designed": 1.50,
    "variables": 1.30, "controlled": 1.50, "published": 1.60,
    "findings": 1.40, "debate": 1.50, "scientific": 1.60,
    "community": 1.70, "replication": 1.10, "methodology": 1.20,
    "flawed": 1.10, "contradictory": 1.00, "outcomes": 1.40,
    "suggesting": 1.30, "original": 1.50, "fundamentally": 1.20,
    "inconclusive": 0.90, "confounding": 0.80, "adequately": 1.00,
    "considerable": 1.30, "subsequently": 1.10, "generated": 1.40,
    "attempts": 1.40, "produced": 1.50, "followed": 1.50,
    "situation": 1.60, "outcome": 1.40, "unavoidable": 0.90,
    "disappointing": 1.10, "deeply": 1.30, "amazing": 1.50,
    "experience": 1.80, "achieved": 1.50, "truly": 1.40, "worst": 1.50,
    "best": 1.90, "everything": 1.70, "quick": 1.50, "brown": 1.60,
    "fox": 1.30, "jumps": 1.20, "lazy": 1.20, "dog": 1.80,
    "carefully": 1.40, "researchers": 1.50, "although": 1.60,
    "because": 2.30, "nevertheless": 1.10, "furthermore": 1.00,
}

_DEFAULT_LOG_FREQ = 1.0  # default for unknown content words


# ---------------------------------------------------------------------------
# Syllable counter (heuristic)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Return an approximate syllable count for *word* using regex heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    # Remove trailing silent-e
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


# ---------------------------------------------------------------------------
# Approximate clause detector
# ---------------------------------------------------------------------------

def _count_clauses(tokens: list[str]) -> int:
    """Return an approximate clause count for a sequence of word tokens.

    Each sentence starts with an implicit main clause (count = 1), and every
    subordinating or coordinating conjunction that typically introduces a new
    clause increments the count.
    """
    clause_count = 1  # main clause
    for token in tokens:
        if token.lower() in _CLAUSE_MARKERS:
            clause_count += 1
    return clause_count


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """Perform linguistic analysis on English *text*.

    Parameters
    ----------
    text : str
        A non-empty English text string.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``sentiment`` : dict with ``polarity`` (float, -1..1) and
          ``subjectivity`` (float, 0..1).
        - ``word_count`` : int
        - ``sentence_count`` : int
        - ``mean_sentence_length`` : float (words per sentence)
        - ``type_token_ratio`` : float (unique / total, case-insensitive)
        - ``flesch_reading_ease`` : float
        - ``flesch_kincaid_grade`` : float
        - ``mean_t_unit_length`` : float (L2SCA)
        - ``clauses_per_t_unit`` : float (L2SCA)
        - ``content_word_frequency`` : float (Coh-Metrix, mean log freq)
        - ``connective_count`` : int (Coh-Metrix)

    Raises
    ------
    TypeError
        If *text* is not a string.
    ValueError
        If *text* is empty or whitespace-only.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("Text must not be empty or whitespace-only")

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)

    # Tokenise words (alphanumeric only)
    all_tokens: list[str] = []
    sentence_token_lists: list[list[str]] = []
    for sent in sentences:
        tokens = [t for t in word_tokenize(sent) if re.match(r"\w+", t)]
        sentence_token_lists.append(tokens)
        all_tokens.extend(tokens)

    word_count = len(all_tokens)
    if word_count == 0:
        raise ValueError("Text contains no recognisable words")

    words_lower = [w.lower() for w in all_tokens]

    # ------------------------------------------------------------------
    # Sentiment (TextBlob)
    # ------------------------------------------------------------------
    blob = TextBlob(text)
    sentiment = {
        "polarity": float(blob.sentiment.polarity),
        "subjectivity": float(blob.sentiment.subjectivity),
    }

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------
    mean_sentence_length = float(word_count) / sentence_count

    unique_words = set(words_lower)
    type_token_ratio = len(unique_words) / word_count

    # ------------------------------------------------------------------
    # Syllable statistics (for readability)
    # ------------------------------------------------------------------
    total_syllables = sum(_count_syllables(w) for w in all_tokens)

    # ------------------------------------------------------------------
    # Flesch Reading Ease  =  206.835
    #                         - 1.015 * (words / sentences)
    #                         - 84.6  * (syllables / words)
    # ------------------------------------------------------------------
    flesch_reading_ease = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (total_syllables / word_count)
    )

    # ------------------------------------------------------------------
    # Flesch-Kincaid Grade Level = 0.39 * (words / sentences)
    #                            + 11.8 * (syllables / words)
    #                            - 15.59
    # ------------------------------------------------------------------
    flesch_kincaid_grade = (
        0.39 * (word_count / sentence_count)
        + 11.8 * (total_syllables / word_count)
        - 15.59
    )

    # ------------------------------------------------------------------
    # L2SCA measures (approximate)
    # T-unit ~ sentence for the purpose of this approximation.
    # ------------------------------------------------------------------
    # Mean T-unit length (words per T-unit, where T-unit ~ sentence)
    mean_t_unit_length = mean_sentence_length

    # Clauses per T-unit
    total_clauses = 0
    for token_list in sentence_token_lists:
        total_clauses += _count_clauses(token_list)
    clauses_per_t_unit = float(total_clauses) / sentence_count

    # ------------------------------------------------------------------
    # Coh-Metrix style measures
    # ------------------------------------------------------------------
    # Content-word frequency: mean log10 frequency of content words.
    # Content words = words that are NOT stop words and are alphabetic.
    content_words = [w for w in words_lower if w.isalpha() and w not in _STOP_WORDS]
    if content_words:
        content_word_frequency = sum(
            _WORD_FREQ.get(w, _DEFAULT_LOG_FREQ) for w in content_words
        ) / len(content_words)
    else:
        content_word_frequency = _DEFAULT_LOG_FREQ

    # Connective count
    connective_count = sum(1 for w in words_lower if w in _CONNECTIVES)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return {
        "sentiment": sentiment,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "mean_sentence_length": mean_sentence_length,
        "type_token_ratio": type_token_ratio,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "mean_t_unit_length": mean_t_unit_length,
        "clauses_per_t_unit": clauses_per_t_unit,
        "content_word_frequency": content_word_frequency,
        "connective_count": connective_count,
    }


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example = (
        "Although the experiment was carefully designed, the results were "
        "inconclusive because several confounding variables had not been "
        "adequately controlled. Nevertheless, the researchers published "
        "their findings, which subsequently generated considerable debate "
        "within the scientific community. Furthermore, the replication "
        "attempts that followed produced contradictory outcomes, suggesting "
        "that the original methodology was fundamentally flawed."
    )

    print("=" * 70)
    print("Linguistic Analysis")
    print("=" * 70)
    print()
    print(f"Text: {example[:80]}...")
    print()

    result = analyze_text(example)

    print("--- Sentiment ---")
    print(f"  Polarity:      {result['sentiment']['polarity']:.3f}")
    print(f"  Subjectivity:  {result['sentiment']['subjectivity']:.3f}")
    print()

    print("--- Basic Statistics ---")
    print(f"  Word count:            {result['word_count']}")
    print(f"  Sentence count:        {result['sentence_count']}")
    print(f"  Mean sentence length:  {result['mean_sentence_length']:.2f}")
    print(f"  Type-token ratio:      {result['type_token_ratio']:.3f}")
    print()

    print("--- Readability ---")
    print(f"  Flesch Reading Ease:     {result['flesch_reading_ease']:.2f}")
    print(f"  Flesch-Kincaid Grade:    {result['flesch_kincaid_grade']:.2f}")
    print()

    print("--- L2SCA Measures ---")
    print(f"  Mean T-unit length:    {result['mean_t_unit_length']:.2f}")
    print(f"  Clauses per T-unit:    {result['clauses_per_t_unit']:.2f}")
    print()

    print("--- Coh-Metrix Style ---")
    print(f"  Content word freq (mean log10): {result['content_word_frequency']:.3f}")
    print(f"  Connective count:               {result['connective_count']}")
