import os
import pickle
from utils import tokenize


class InvertedIndex:
    idx_pickel = "./cache/index.pkl"
    docmap_pickel = "./cache/docmap.pkl"
    term_frequencies_pickel = "./cache/term_frequencies.pkl"

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text, self.stopwords)
        tf = self.term_frequencies.get(doc_id, dict())
        for idx, token in enumerate(tokens):
            self.index[token] = self.index.get(token, set()).union(set({doc_id}))
            tf[token] = tf.get(token, 0) + 1
        self.term_frequencies[doc_id] = tf

    def get_documents(self, term: str):
        # print("term", term)
        return sorted(self.index.get(term.lower(), []))

    def get_tf(self, doc_id, term):
        doc_terms = self.term_frequencies[doc_id]
        # print(f"doc_terms: {doc_terms}")
        return doc_terms.get(term, 0)

    def build(self, movies: dict, stopwords: list[str]):
        self.stopwords = stopwords
        for idx, mov in enumerate(movies):
            # text = f"{mov['title']}"
            text = f"{mov['title']} {mov['description']}"
            # print(f"Adding {idx} - {text}")
            self.__add_document(mov['id'], text)
            self.docmap[idx] = mov
        pass

    def save(self):
        os.makedirs(os.path.dirname(self.idx_pickel), exist_ok=True)
        with open(self.idx_pickel, 'wb') as f:
            pickle.dump(self.index, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.docmap_pickel, 'wb') as f:
            pickle.dump(self.docmap, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.term_frequencies_pickel, 'wb') as f:
            pickle.dump(self.term_frequencies, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        if not os.path.exists(self.idx_pickel):
            raise Exception(f"File Index: {self.idx_pickel} is not there.")
        if not os.path.exists(self.docmap_pickel):
            raise Exception(f"File Docmap: {self.docmap_pickel} is not there.")
        if not os.path.exists(self.term_frequencies_pickel):
            raise Exception(f"File Term frequencies: {self.term_frequencies_pickel} is not there.")
        with open(self.idx_pickel, 'rb') as file:
            self.index = pickle.load(file=file)
        with open(self.docmap_pickel, 'rb') as file:
            self.docmap = pickle.load(file=file)
        with open(self.term_frequencies_pickel, 'rb') as file:
            self.term_frequencies = pickle.load(file=file)



        



