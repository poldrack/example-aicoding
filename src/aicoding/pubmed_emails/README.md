# pubmed_emails (#6)

Search PubMed for articles, extract author info, and scrape email addresses from the web.

## Approach

1. **PubMed search**: Uses Biopython's `Entrez.esearch` to find article PMIDs.
2. **Author extraction**: Parses PubMed XML records for first/last author names and affiliations.
3. **Email scraping**: Performs Google searches using author name + affiliation, then extracts email addresses from the results using regex.

## Known issues

- Web scraping is inherently fragile — Google may block requests or change HTML structure.
- Email addresses are not always publicly available on institutional pages.
- Rate limiting (1s delay) is applied between web requests.
