import argparse
import json
import string
from stopword import STOPS


def erase_stopword(text: list[str]):
    return list(filter(lambda ele: ele not in STOPS, text))


def matching_pipeline(query: str, title: str):
    query = (query.lower().translate(str.maketrans("", "", string.punctuation))).split(" ")
    title = (title.lower().translate(str.maketrans("", "", string.punctuation))).split(" ")

    for ele_query in query:
        if ele_query in title:
            return True
        for ele_title in title:
            if ele_query in ele_title:
                return True
    # query = erase_stopword(query)
    # title = erase_stopword(title)
    # print("query: ", query, title)
    # return query in title
    return False


def search(query: str):
    with open("./data/movies.json", "r") as f:
        movies: list = json.load(f)["movies"]
        result = list(filter(lambda m: matching_pipeline(query, m["title"]), movies))
        sorted_result = sorted(result, key=lambda m: m["id"])[:5]
        for idx, mov in enumerate(sorted_result):
            print(f"{idx + 1}. {mov['title']}")
        print(len(result), len(movies))


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
