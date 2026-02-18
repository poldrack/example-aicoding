# textanalysis

Linguistic analysis module that takes English text and returns sentiment, readability, and complexity metrics.

## Function

`analyze_text(text: str) -> dict` -- accepts a non-empty English string and returns a dictionary with the following keys:

### Sentiment (TextBlob)
- `sentiment` -- dict with `polarity` (-1 to 1) and `subjectivity` (0 to 1)

### Basic Statistics
- `word_count` -- total word count
- `sentence_count` -- number of sentences
- `mean_sentence_length` -- words per sentence
- `type_token_ratio` -- unique words / total words (case-insensitive)

### Readability (Flesch formulas)
- `flesch_reading_ease` -- higher = easier to read; computed from word/sentence/syllable counts
- `flesch_kincaid_grade` -- US school grade level needed to comprehend the text

### L2SCA Measures (approximate)
- `mean_t_unit_length` -- words per T-unit (T-unit approximated as sentence)
- `clauses_per_t_unit` -- clause count per T-unit, where clauses are detected via subordinating/coordinating conjunctions

### Coh-Metrix Style Measures
- `content_word_frequency` -- mean log10 frequency of content words (non-stop, alphabetic), using a built-in frequency table with a default of 1.0 for unknown words
- `connective_count` -- total count of discourse connective words (causal, adversative, additive, temporal)

## Approach

- **Tokenization**: NLTK `sent_tokenize` and `word_tokenize` for sentence splitting and word tokenization.
- **Sentiment**: TextBlob's built-in sentiment analyzer.
- **Syllable counting**: Regex heuristic (vowel-group counting with silent-e removal).
- **Readability**: Standard Flesch formulas applied to word/sentence/syllable counts.
- **L2SCA**: T-units are approximated as sentences. Clauses are detected by counting clause-introducing markers (subordinating conjunctions, relative pronouns, and select coordinating conjunctions) plus one implicit main clause per sentence.
- **Coh-Metrix**: Content words are identified by excluding NLTK English stop words. Word frequency comes from a built-in lookup table of common English words (log10 frequency per million). Connectives are matched against a curated list of discourse connectives.

## Dependencies

- `textblob` -- sentiment analysis
- `nltk` -- tokenization, POS tagging, stopword lists

## Running

```bash
python -m aicoding.textanalysis.solution
```

## Testing

```bash
pytest tests/textanalysis/ -v
```
