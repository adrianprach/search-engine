import math
import os
import pickle
from utils import tokenize


class InvertedIndex:
    stopwords_path = "./data/stopwords.txt"
    idx_pickel_path = "./cache/index.pkl"
    docmap_pickel_path = "./cache/docmap.pkl"
    term_frequencies_pickel_path = "./cache/term_frequencies.pkl"
    doc_lengths_path = "./cache/doc_lengths.pkl"
    avg_doc_length: None | float = None

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text, self.stopwords)
        tf = self.term_frequencies.get(doc_id, dict())
        for idx, token in enumerate(tokens):
            self.index[token] = self.index.get(token, set()).union(set({doc_id}))
            tf[token] = tf.get(token, 0) + 1
        self.term_frequencies[doc_id] = tf
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self):
        if self.avg_doc_length is not None:
            return self.avg_doc_length
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        self.avg_doc_length = avg_doc_length
        # print(f"AVG DOC LENGTH: {avg_doc_length}")
        return avg_doc_length

    def add_doc(self, doc_id: int, text: str):
        self.load_stopword()
        self.__add_document(doc_id, text)

    def get_documents(self, term: str):
        return sorted(self.index.get(term.lower(), []))

    def search(self, query: str):
        tokenized_query = set(tokenize(query, self.stopwords))
        results = []
        for token in tokenized_query:
            results.extend(self.get_documents(token))
        return results

    def search_with_bm25(self, query: str, limit: int = 5):
        terms = set(tokenize(query, self.stopwords))
        matched = set(self.search(query))
        results = []

        for doc_id in matched:
            bm25 = 0
            movie = self.docmap[doc_id]
            for token in terms:
                bm25_tf = self.get_bm25_tf(movie["id"], token)
                bm25_idf = self.get_bm25_idf(token)
                bm25 += (bm25_tf * bm25_idf)
            # print(f"{movie['title']}, {bm25:.2f}")
            results.append({**movie, "score": bm25})
        results = sorted(results, key=lambda movie: movie["score"], reverse=True)
        results = list(map(lambda movie: f"({movie['id']}) {movie['title']} - Score: {movie['score']:.2f}", results))[:limit]
        return results

    def get_tf(self, doc_id, term):
        doc_terms = self.term_frequencies[doc_id]
        # print(f"doc_terms: {doc_terms}")
        return doc_terms.get(term, 0)

    def get_idf(self, term: str):
        tokenized_term = tokenize(term, self.stopwords)
        results = self.get_documents(tokenized_term[0])
        total_doc_count = len(self.docmap)
        total_match_doc_count = len(results)
        idf = math.log((total_doc_count + 1) / (total_match_doc_count + 1))
        return idf

    def get_tfidf(self, doc_id: int, term: str):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        tf_idf = tf * idf
        return tf_idf

    def get_bm25_idf(self, term: str):
        all_docs_len = len(self.docmap)
        df = len(self.search(term))
        bm25 = math.log((all_docs_len - df + 0.5) / (df + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = 1.5, b: float = 0.75):
        tokenized_term = tokenize(term, self.stopwords)
        # print(f"term: {term} => {tokenized_term}")
        tf = self.get_tf(doc_id, tokenized_term[0])

        doc_length = self.doc_lengths.get(doc_id)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        tuned_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        # print(f"DOC LENGTH: {doc_length}, AVG DOC LENGTH: {avg_doc_length}")
        return tuned_tf

    def build(self, movies: dict):
        self.load_stopword()
        for idx, mov in enumerate(movies):
            # text = f"{mov['title']}"
            text = f"{mov['title']} {mov['description']}"
            # print(f"Adding {idx} - {text}")
            self.__add_document(mov["id"], text)
            self.docmap[mov["id"]] = mov
        pass

    def save(self):
        os.makedirs(os.path.dirname(self.idx_pickel_path), exist_ok=True)
        with open(self.idx_pickel_path, "wb") as f:
            pickle.dump(self.index, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.docmap_pickel_path, "wb") as f:
            pickle.dump(self.docmap, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.term_frequencies_pickel_path, "wb") as f:
            pickle.dump(self.term_frequencies, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        if not os.path.exists(self.idx_pickel_path):
            raise Exception(f"File Index: {self.idx_pickel_path} is not there.")
        if not os.path.exists(self.docmap_pickel_path):
            raise Exception(f"File Docmap: {self.docmap_pickel_path} is not there.")
        if not os.path.exists(self.term_frequencies_pickel_path):
            raise Exception(
                f"File Term frequencies: {self.term_frequencies_pickel_path} is not there."
            )
        if not os.path.exists(self.doc_lengths_path):
            raise Exception(f"File Doc Length: {self.doc_lengths_path} is not there.")

        with open(self.idx_pickel_path, "rb") as file:
            self.index = pickle.load(file=file)
        with open(self.docmap_pickel_path, "rb") as file:
            self.docmap = pickle.load(file=file)
        with open(self.term_frequencies_pickel_path, "rb") as file:
            self.term_frequencies = pickle.load(file=file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file=file)
        self.load_stopword()

    def load_stopword(self):
        if not os.path.exists(self.stopwords_path):
            raise Exception(f"File Stopwords: {self.stopwords_path} is not there.")
        with open("./data/stopwords.txt", "r") as file:
            self.stopwords = file.read().strip().split("\n")
