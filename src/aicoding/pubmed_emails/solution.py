"""PubMed author email scraper.

Searches PubMed for articles matching a query, extracts first and last author
names and affiliations, then searches the web for their institutional pages
and scrapes email addresses.
"""

import re
import time

import requests
from Bio import Entrez

# Required by NCBI
Entrez.email = "user@example.com"

EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')


def search_pubmed(query, max_results=10):
    """Search PubMed and return a list of article PMIDs.

    Parameters
    ----------
    query : str
        PubMed search query.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    list of str
        PubMed IDs.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_articles(pmids):
    """Fetch full article records for a list of PMIDs.

    Parameters
    ----------
    pmids : list of str

    Returns
    -------
    list of dict
        PubMed article records.
    """
    if not pmids:
        return []
    ids = ",".join(pmids)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()
    return records.get("PubmedArticle", [])


def extract_emails_from_text(text):
    """Extract email addresses from a text string.

    Parameters
    ----------
    text : str
        Text that may contain email addresses.

    Returns
    -------
    list of str
        Email addresses found, excluding common false positives.
    """
    if not text:
        return []
    found = EMAIL_PATTERN.findall(text)
    return [
        e for e in found
        if not e.endswith("@example.com")
        and not e.endswith("@sentry.io")
        and "google" not in e.lower()
    ]


def extract_authors(record):
    """Extract first and last author information from a PubMed record.

    Parameters
    ----------
    record : dict
        A single PubMed article record.

    Returns
    -------
    dict
        Keys: 'first_author', 'last_author'. Each has 'first_name',
        'last_name', 'affiliation', and 'emails'.
    """
    authors = (
        record.get("MedlineCitation", {})
        .get("Article", {})
        .get("AuthorList", [])
    )

    def _extract(author):
        affil_list = author.get("AffiliationInfo", [])
        affiliation = affil_list[0].get("Affiliation", "") if affil_list else ""
        # Extract emails from all affiliation strings
        emails = []
        for affil_info in affil_list:
            affil_text = affil_info.get("Affiliation", "")
            emails.extend(extract_emails_from_text(affil_text))
        return {
            "first_name": author.get("ForeName", ""),
            "last_name": author.get("LastName", ""),
            "affiliation": affiliation,
            "emails": list(set(emails)),
        }

    if not authors:
        empty = {"first_name": "", "last_name": "", "affiliation": "", "emails": []}
        return {"first_author": empty, "last_author": empty}

    return {
        "first_author": _extract(authors[0]),
        "last_author": _extract(authors[-1]),
    }


def find_emails_from_web(name, affiliation, timeout=10):
    """Search the web for a researcher's email using their name and affiliation.

    Uses a Google search to find institutional pages and scrapes email
    addresses from the results.

    Parameters
    ----------
    name : str
        Full name of the researcher.
    affiliation : str
        Institutional affiliation.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    list of str
        Email addresses found.
    """
    query = f"{name} {affiliation} email"
    search_url = "https://www.google.com/search"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    emails = set()
    try:
        resp = requests.get(
            search_url, params={"q": query}, headers=headers, timeout=timeout
        )
        if resp.status_code == 200:
            found = EMAIL_PATTERN.findall(resp.text)
            emails.update(found)
    except (requests.RequestException, Exception):
        pass

    # Filter out common false positives
    filtered = [
        e for e in emails
        if not e.endswith("@example.com")
        and not e.endswith("@sentry.io")
        and "google" not in e.lower()
    ]
    return filtered


def find_author_emails(query, max_results=5, web_search=False):
    """Search PubMed and find email addresses for first/last authors.

    Parameters
    ----------
    query : str
        PubMed search query.
    max_results : int
        Number of articles to process.
    web_search : bool
        If True, fall back to web scraping for authors without
        affiliation emails. Disabled by default since Google
        blocks automated requests.

    Returns
    -------
    list of dict
        Each dict has 'name', 'affiliation', 'role', and 'emails'.
    """
    pmids = search_pubmed(query, max_results=max_results)
    articles = fetch_articles(pmids)

    results = []
    for article in articles:
        author_info = extract_authors(article)

        for role in ("first_author", "last_author"):
            author = author_info[role]
            name = f"{author['first_name']} {author['last_name']}".strip()
            if not name:
                continue

            emails = author.get("emails", [])
            if not emails and web_search:
                emails = find_emails_from_web(name, author["affiliation"])
                time.sleep(1)  # be polite to search engines

            results.append({
                "name": name,
                "first_name": author["first_name"],
                "last_name": author["last_name"],
                "affiliation": author["affiliation"],
                "role": role.replace("_", " "),
                "emails": emails,
            })

    return results


if __name__ == "__main__":
    print("PubMed Author Email Finder")
    print("=" * 60)
    print('Searching PubMed for "cognitive control"...\n')

    results = find_author_emails("cognitive control", max_results=20)

    for entry in results:
        print(f"  {entry['role'].title()}: {entry['name']}")
        print(f"    Affiliation: {entry['affiliation']}")
        if entry["emails"]:
            for email in entry["emails"]:
                print(f"    Email: {email}")
        else:
            print("    Email: not found")
        print()
