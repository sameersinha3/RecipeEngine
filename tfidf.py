import numpy as np
import pandas as pd
import math

class TFIDF:
    vocab = np.array([])
    IDF = None
    K = 1.2

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy().reset_index(drop=True)
        self.clean_text()

    # Turn items/lists into a valid string
    def flatten(self, val):
        if isinstance(val, list):
            return " ".join(map(str, val))
        
        if pd.isna(val):
            return ""
        
        return str(val)

    def clean_text(self):
        texts = []
        for _, row in self.dataset.iterrows():
            parts = [
                self.flatten(row.get("title", "")),
                self.flatten(row.get("ingredients", "")),
                self.flatten(row.get("instructions", "")),
                self.flatten(row.get("tags", "")),
                self.flatten(row.get("description", "")),
            ]

            text = " ".join(parts).lower()
            texts.append(text)

        self.dataset["text"] = texts

    def build_vocabulary(self, max_vocab_size: int = 5000):
        words = " ".join(self.dataset["text"]).split()
        
        #list unique words & their counts, sort by decreasing order of frequency    
        values, counts = np.unique(words, return_counts=True)
        vocab = values[np.argsort(-counts)][:max_vocab_size]

        self.vocab = vocab

    # include words from the query into the vocabulary
    def adapt_vocab_query(self, query):
        vocab = list(self.vocab)

        #Split  words
        query_list = set(query.lower().split())

        for word in query_list:
            if word not in vocab:
                vocab.append(word)

        self.vocab = np.array(vocab)

    def compute_IDF(self):
        M = self.dataset.shape[0]
        collection = self.dataset["text"]

        self.IDF = np.zeros(self.vocab.size)

        for idx, word in enumerate(self.vocab):
            k = 0

            for doc in collection:
                if word in doc.split():
                    k += 1
            
            #Handle vocab words that don't appear in doc
            if k == 0:
                self.IDF[idx] = 0.0
            else:
                self.IDF[idx] = math.log((M + 1) / k)

    def text2TFIDF(self, text: str, applyBM25_and_IDF: bool = False):
        words = text.lower().split()
        tfidfVector = np.zeros(self.vocab.size)

        for idx, word in enumerate(self.vocab):
            tf = words.count(word)

            if tf == 0:
                continue

            tfidfVector[idx] = tf

            if applyBM25_and_IDF:
                y = ((self.K + 1) * tf) / (tf + self.K)

                tfidfVector[idx] = y * self.IDF[idx]

        return tfidfVector

    def tfidf_score(self, query: str, doc: str, applyBM25_and_IDF: bool = False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc, applyBM25_and_IDF)

        # Return the relevance 
        return np.dot(q, d)

    def execute_search_TF_IDF(self, query: str, applyBM25_and_IDF: bool = False):
        self.adapt_vocab_query(query)

        # No parameters needed, compute_IDF computes these
        self.compute_IDF()

        relevances = np.zeros(self.dataset.shape[0])
        for idx, row in self.dataset.iterrows():
            relevances[idx] = self.tfidf_score(query, row["text"], applyBM25_and_IDF=applyBM25_and_IDF)

        return relevances