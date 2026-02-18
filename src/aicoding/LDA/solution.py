"""Download PubMed abstracts and perform LDA topic modeling.

Downloads abstracts matching a query, cleans text by removing stopwords
and lemmatizing, then applies Latent Dirichlet Allocation to find topics.
"""

from Bio import Entrez
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

Entrez.email = "user@example.com"

# Ensure NLTK data is available
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def download_abstracts(query, max_results=1000):
    """Download abstracts from PubMed matching a query.

    Args:
        query: PubMed search query string.
        max_results: Maximum number of abstracts to retrieve.

    Returns:
        List of abstract text strings.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    search_results = Entrez.read(handle)
    handle.close()

    id_list = search_results["IdList"]
    if not id_list:
        return []

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    for rec in records["PubmedArticle"]:
        article = rec["MedlineCitation"]["Article"]
        abstract_data = article.get("Abstract", {})
        abstract_texts = abstract_data.get("AbstractText", [])
        if abstract_texts:
            full_abstract = " ".join(str(t) for t in abstract_texts)
            abstracts.append(full_abstract)

    return abstracts


def clean_text(text):
    """Clean text by removing stopwords and lemmatizing.

    Args:
        text: Raw text string.

    Returns:
        List of cleaned, lemmatized tokens.
    """
    if not text:
        return []

    lemmatizer = WordNetLemmatizer()
    # Remove non-alphabetic characters and lowercase
    tokens = re.findall(r"[a-z]+", text.lower())
    # Remove stopwords and lemmatize
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 2
    ]
    return cleaned


def run_lda(cleaned_docs, num_topics=10, num_words=10):
    """Run LDA topic modeling on a list of cleaned documents.

    Args:
        cleaned_docs: List of lists of tokens (cleaned documents).
        num_topics: Number of topics to extract.
        num_words: Number of top words to return per topic.

    Returns:
        Tuple of (LdaModel, list of topics) where each topic is a list
        of (word, weight) tuples.
    """
    if not cleaned_docs:
        raise ValueError("Cannot run LDA on an empty corpus.")

    dictionary = corpora.Dictionary(cleaned_docs)
    corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]

    model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15,
        random_state=42,
    )

    topics = []
    for topic_id in range(num_topics):
        top_words = model.show_topic(topic_id, topn=num_words)
        topics.append(top_words)

    return model, topics


if __name__ == "__main__":
    print("Downloading 1000 abstracts for 'cognitive control'...")
    abstracts = download_abstracts("cognitive control", max_results=1000)
    print(f"Downloaded {len(abstracts)} abstracts.")

    cleaned = [clean_text(abstract) for abstract in abstracts]
    cleaned = [doc for doc in cleaned if doc]  # Remove empty docs

    print(f"Running LDA with 10 topics on {len(cleaned)} documents...")
    model, topics = run_lda(cleaned, num_topics=10)

    for i, topic in enumerate(topics):
        words = ", ".join(f"{word} ({weight:.3f})" for word, weight in topic)
        print(f"\nTopic {i + 1}: {words}")
