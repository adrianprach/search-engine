import argparse
import json
import math

from inverted_index import InvertedIndex
from utils import tokenize

STOPS = []
MOVIES = []

def define_stopwords():
    with open("./data/stopwords.txt", "r") as f:
        return f.read().strip().split("\n")

def define_dataset() -> list[dict]:
    with open("./data/movies.json", "r") as f:
        return json.load(f)["movies"]

def search(query: str):
    indexes = InvertedIndex()
    indexes.load()
    tokenized_query = set(tokenize(query, STOPS))
    results = []
    for token in tokenized_query:
        results.extend(indexes.get_documents(token))
    print(f"Results: {len(results)}, top 5 results:")
    for idx, doc_id in enumerate(results[:5]):
        movie = indexes.docmap[doc_id]
        print(f"{idx+1}. {movie['title']}, Doc ID: {movie['id']}")

def get_tf(doc_id: int, term: str):
    if doc_id < 0:
        raise Exception("Doc_id should not be below 0.")
    if term is None or len(term) < 1:
        raise Exception("Search term should not be empty.")
    indexes = InvertedIndex()
    indexes.load()
    tf = indexes.get_tf(doc_id, term.strip().lower())
    print(f"TF: {term}: {tf} times.")
    return tf

def get_idf(term: str):
    if term is None or len(term) < 1:
        raise Exception("Term should not be empty.")
    indexes = InvertedIndex()
    indexes.load()
    tokenized = tokenize(term, STOPS)
    # results = indexes.get_documents(tokenized[0])
    results = indexes.index[tokenized[0]]
    total_doc_count = len(indexes.docmap)
    total_match_doc_count = len(results)
    idf = math.log((total_doc_count + 1) / (total_match_doc_count + 1))
    print(f"Inverse document frequency of {term}: {idf}.")
    return idf

def get_tfidf(doc_id: int, term: str):
    if term is None or len(term) < 1:
        raise Exception("Term cannot be empty!")
    idx = InvertedIndex()
    idx.load()
    tf = get_tf(doc_id, term)
    idf = get_idf(term)
    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
    return tf_idf



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build index and docmap for movies")
    tf_parser = subparsers.add_parser("tf", help="Get term frequencies from specified doc id")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequencies")
    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF for document rating")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Doc id")
    tf_parser.add_argument("term", type=str, help="Term for term frequencies")
    idf_parser.add_argument("term", type=str, help="Term for inverse document frequencies")

    tfidf_parser.add_argument("doc_id", type=int, help="Doc id")
    tfidf_parser.add_argument("term", type=str, help="Term for tf-idf")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build(MOVIES, STOPS)
            inverted_index.save()
        case "tf":
            get_tf(args.doc_id, args.term)
        case "idf":
            get_idf(args.term)
        case "tfidf":
            get_tfidf(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    STOPS = define_stopwords()
    MOVIES = define_dataset()
    main()
