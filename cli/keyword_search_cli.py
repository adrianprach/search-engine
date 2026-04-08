import argparse
import json

def search(query: str):
    with open('./data/movies.json', 'r') as f:
        movies: list = json.load(f)['movies']
        result = list(filter(lambda m: query in m["title"], movies))
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
