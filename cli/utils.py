import argparse
import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def define_stopwords():
    with open("./data/stopwords.txt", "r") as f:
        return f.read().strip().split("\n")


def stem_tokens(text: list[str]):
    return list(map(lambda token: stemmer.stem(token), text))


def erase_stopword(text: list[str], STOPS: list[str]):
    return list(filter(lambda ele: ele not in STOPS, text))


def clear_punctuation(text: str):
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str, STOPS: list[str]):
    return stem_tokens(
        erase_stopword(clear_punctuation(text.lower()).split(" "), STOPS)
    )


def matching_pipeline(query: str, title: str):
    query = stem_tokens(
        erase_stopword(
            (query.lower().translate(str.maketrans("", "", string.punctuation))).split(
                " "
            )
        )
    )
    title = stem_tokens(
        erase_stopword(
            (title.lower().translate(str.maketrans("", "", string.punctuation))).split(
                " "
            )
        )
    )

    for ele_query in query:
        if ele_query in title:
            return True
        for ele_title in title:
            if ele_query in ele_title:
                return True
    return False
