import argparse
import json

from inverted_index import InvertedIndex

MOVIES = []

# Default tuning parameters for bm25 
BM25_K1 = 1.5
BM25_B = 0.75

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

def get_bm25tf(doc_id: int, term: str, k1: float, b: float):
    idx = InvertedIndex()
    idx.load()
    tuned_tf = idx.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {tuned_tf:.2f}")

def search_with_bm25(query: str, limit: int):
    idx = InvertedIndex()
    idx.load()
    results = idx.search_with_bm25(query, limit)
    for index, movie in enumerate(results):
        print(f"{index + 1}. {movie}")


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
    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring.")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Doc id")
    tf_parser.add_argument("term", type=str, help="Term for term frequencies")
    idf_parser.add_argument(
        "term", type=str, help="Term for inverse document frequencies"
    )

    tfidf_parser.add_argument("doc_id", type=int, help="Doc id")
    tfidf_parser.add_argument("term", type=str, help="Term for tf-idf")

    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameters")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable B parameters.")

    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Limit results")

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
        case "bm25tf":
            get_bm25tf(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            search_with_bm25(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    MOVIES = define_dataset()
    # idx = InvertedIndex()
    # idx.add_doc(1, MOVIES[0]["description"])
    # print(idx.docmap, idx.term_frequencies[1]["anbuselvan"])
    main()
