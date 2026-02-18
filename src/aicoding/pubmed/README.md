# pubmed

Searches PubMed using Biopython's Entrez API and wraps results in an OOP hierarchy.

## Approach

- `AbstractPublication` — ABC with an abstract `from_pubmed()` method.
- `Article` — Concrete subclass that extracts `authors` and `title` from a PubMed record.
- `search_pubmed()` — Performs an Entrez esearch + efetch and returns a dict of `Article` instances keyed by PMID.

## Usage

```bash
python -m aicoding.pubmed.solution
```
