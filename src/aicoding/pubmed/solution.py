"""PubMed publication search using Biopython/Entrez.

Provides an abstract base class for publications and an Article class
that populates itself from PubMed records.
"""

from abc import ABC, abstractmethod
from Bio import Entrez

# Required by NCBI
Entrez.email = "user@example.com"


class AbstractPublication(ABC):
    """Abstract base class representing a publication."""

    @abstractmethod
    def from_pubmed(self, record):
        """Populate the publication from a PubMed record."""
        pass


class Article(AbstractPublication):
    """A journal article populated from PubMed data."""

    def from_pubmed(self, record):
        """Convert a PubMed record into Article attributes.

        Args:
            record: A dict from Entrez containing AuthorList and ArticleTitle.

        Returns:
            self, for method chaining.
        """
        self.authors = list(record.get("AuthorList", []))
        self.title = str(record.get("ArticleTitle", ""))
        return self


def search_pubmed(query, max_results=20):
    """Search PubMed and return a dict of Article instances keyed by PMID.

    Args:
        query: Search query string.
        max_results: Maximum number of results to retrieve.

    Returns:
        dict mapping PMID strings to Article instances.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    search_results = Entrez.read(handle)
    handle.close()

    id_list = search_results["IdList"]
    if not id_list:
        return {}

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = {}
    for rec in records["PubmedArticle"]:
        citation = rec["MedlineCitation"]
        pmid = str(citation["PMID"])
        article_data = citation["Article"]
        article = Article()
        article.from_pubmed(article_data)
        articles[pmid] = article

    return articles


if __name__ == "__main__":
    results = search_pubmed("cognitive control")
    for pmid, article in results.items():
        print(f"PMID {pmid}: {article.title} ({len(article.authors)} authors)")
