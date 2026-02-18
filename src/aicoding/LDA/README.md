# LDA

Downloads PubMed abstracts and performs Latent Dirichlet Allocation topic modeling.

## Approach

1. **Download** — Uses Biopython Entrez to search PubMed and fetch abstract text.
2. **Clean** — Removes stopwords (NLTK), lemmatizes tokens (WordNet), filters short words.
3. **Model** — Builds a Gensim LDA model on the cleaned corpus and extracts top words per topic.

## Usage

```bash
python -m aicoding.LDA.solution
```
