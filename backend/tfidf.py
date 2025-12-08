import numpy as np
import pandas as pd
import math
from collections import Counter
from tqdm import tqdm

class TFIDF:
    vocab = np.array([])
    IDF = None
    K = 1.2

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy().reset_index(drop=True)
        self.IDF = None  # Will be computed once after vocabulary is built
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
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc="  Preparing text", leave=False):
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
        word_counts = Counter()
        
        print(f"  Counting words from {len(self.dataset):,} recipes...", flush=True)
        for text in tqdm(self.dataset["text"], desc="  Building vocabulary", unit=" recipes"):
            words = text.split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(max_vocab_size)
        vocab = np.array([word for word, count in most_common])
        
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
        if self.IDF is not None:
            return  # Already computed
        
        M = self.dataset.shape[0]
        collection = self.dataset["text"]

        # Create a set of vocabulary words for fast lookup
        vocab_set = set(self.vocab)
        
        # Count document frequency for each word in a single pass
        # doc_freq[word] = number of documents containing that word
        doc_freq = {word: 0 for word in self.vocab}
        
        # Single pass: iterate through documents once
        for doc in tqdm(collection, desc="  Computing IDF", unit=" docs"):
            # Get unique words in this document
            doc_words = set(doc.split())
            
            # Count which vocab words appear in this document
            for word in doc_words:
                if word in vocab_set:
                    doc_freq[word] += 1

        self.IDF = np.zeros(self.vocab.size)
        for idx, word in enumerate(self.vocab):
            df = doc_freq[word]  # document frequency
            if df == 0:
                self.IDF[idx] = 0.0
            else:
                self.IDF[idx] = math.log((M + 1) / df)

    def text2TFIDF(self, text: str, applyBM25_and_IDF: bool = False):
        words = text.lower().split()
        tfidfVector = np.zeros(self.vocab.size)

        # Store original vocab size to handle expanded vocab
        original_vocab_size = len(self.IDF) if self.IDF is not None else len(self.vocab)

        for idx, word in enumerate(self.vocab):
            tf = words.count(word)

            if tf == 0:
                continue

            tfidfVector[idx] = tf

            if applyBM25_and_IDF and self.IDF is not None:
                y = ((self.K + 1) * tf) / (tf + self.K)
                
                if idx < original_vocab_size:
                    tfidfVector[idx] = y * self.IDF[idx]
                else:
                    # New query words not in original vocab get IDF = 0
                    tfidfVector[idx] = 0

        return tfidfVector

    def tfidf_score(self, query: str, doc: str, applyBM25_and_IDF: bool = False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc, applyBM25_and_IDF)

        # Return the relevance 
        return np.dot(q, d)

    def execute_search_TF_IDF(self, query: str, applyBM25_and_IDF: bool = False):
        self.adapt_vocab_query(query)

        # Ensure IDF is computed (only once at initialization)
        if self.IDF is None:
            self.compute_IDF()

        # Pre-compute query vector once
        query_vector = self.text2TFIDF(query, applyBM25_and_IDF)

        # Vectorized computation: compute all document vectors at once
        texts = self.dataset["text"].tolist()
        
        # Compute all document vectors
        doc_vectors = np.array([
            self.text2TFIDF(text, applyBM25_and_IDF) 
            for text in texts
        ])
        
        # Vectorized dot product: compute all similarities at once
        relevances = np.dot(doc_vectors, query_vector)

        return relevances