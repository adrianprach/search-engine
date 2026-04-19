import argparse
import json

from inverted_index import InvertedIndex

MOVIES = []


def define_dataset() -> list[dict]:
    with open("./data/movies.json", "r") as f:
        return json.load(f)["movies"]


def search(query: str):
    indexes = InvertedIndex()
    indexes.load()
    results = indexes.search(query)
    print(f"Results: {len(results)}, top 5 results:")
    for idx, doc_id in enumerate(results[:5]):
        movie = indexes.docmap[doc_id]
        print(f"{idx + 1}. {movie['title']}, Doc ID: {movie['id']}")


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
    idf = indexes.get_idf(term)
    print(f"Inverse document frequency of {term}: {idf}.")


def get_tfidf(doc_id: int, term: str):
    if term is None or len(term) < 1:
        raise Exception("Term cannot be empty!")
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tfidf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document id '{doc_id}': {tf_idf:.2f}")

def get_bm25idf(term: str):
    if term is None or len(term) < 1:
        raise Exception("Term cannot be empty!")

    idx = InvertedIndex()
    idx.load()
    bm25idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build index and docmap for movies")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequencies from specified doc id"
    )
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequencies")
    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF for document rating")
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Doc id")
    tf_parser.add_argument("term", type=str, help="Term for term frequencies")
    idf_parser.add_argument(
        "term", type=str, help="Term for inverse document frequencies"
    )

    tfidf_parser.add_argument("doc_id", type=int, help="Doc id")
    tfidf_parser.add_argument("term", type=str, help="Term for tf-idf")

    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build(MOVIES)
            inverted_index.save()
        case "tf":
            get_tf(args.doc_id, args.term)
        case "idf":
            get_idf(args.term)
        case "tfidf":
            get_tfidf(args.doc_id, args.term)
        case "bm25idf":
            get_bm25idf(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    MOVIES = define_dataset()
    main()
