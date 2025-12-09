import numpy as np
import pandas as pd
import math
from collections import Counter
from typing import Optional
from tqdm import tqdm
import faiss

class TFIDF:
    vocab = np.array([])
    IDF = None
    K = 1.2

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy().reset_index(drop=True)
        self.IDF = None  
        self.doc_vectors = None 
        self.index = None
        self.clean_text()

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

    def adapt_vocab_query(self, query):
        vocab = list(self.vocab)

        query_list = set(query.lower().split())

        for word in query_list:
            if word not in vocab:
                vocab.append(word)

        self.vocab = np.array(vocab)

    def compute_IDF(self):
        if self.IDF is not None:
            return 
        
        M = self.dataset.shape[0]
        collection = self.dataset["text"]

        vocab_set = set(self.vocab)

        doc_freq = {word: 0 for word in self.vocab}
        
        for doc in tqdm(collection, desc="  Computing IDF", unit=" docs"):
            doc_words = set(doc.split())
            
            for word in doc_words:
                if word in vocab_set:
                    doc_freq[word] += 1

        self.IDF = np.zeros(self.vocab.size)
        for idx, word in enumerate(self.vocab):
            df = doc_freq[word]  
            if df == 0:
                self.IDF[idx] = 0.0
            else:
                self.IDF[idx] = math.log((M + 1) / df)

    def build_document_vectors(self):
        if self.doc_vectors is not None:
            return
        
        print("  Pre-computing TF-IDF document vectors...", flush=True)
        M = len(self.dataset)
        texts = self.dataset["text"].tolist()
        
        doc_vectors = []
        for text in tqdm(texts, desc="  Computing doc vectors", unit=" docs", leave=False):
            vec = self.text2TFIDF(text, applyBM25_and_IDF=True)
            doc_vectors.append(vec)
        
        self.doc_vectors = np.array(doc_vectors).astype('float32')
        
        print("  Building FAISS index for TF-IDF...", end=" ", flush=True)
        vector_dim = self.doc_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dim)
        
        faiss.normalize_L2(self.doc_vectors)
        self.index.add(self.doc_vectors)

    def text2TFIDF(self, text: str, applyBM25_and_IDF: bool = False):
        words = text.lower().split()
        tfidfVector = np.zeros(self.vocab.size)

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
                    tfidfVector[idx] = 0

        return tfidfVector

    def tfidf_score(self, query: str, doc: str, applyBM25_and_IDF: bool = False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc, applyBM25_and_IDF)

        return np.dot(q, d)

    def execute_search_TF_IDF(self, query: str, applyBM25_and_IDF: bool = False, top_k: Optional[int] = None):
        if self.IDF is None:
            self.compute_IDF()
        if self.doc_vectors is None:
            self.build_document_vectors()


        original_vocab = self.vocab.copy()
        
        words = query.lower().split()
        query_vector = np.zeros(original_vocab.size)
        
        for idx, word in enumerate(original_vocab):
            tf = words.count(word)
            if tf > 0:
                if applyBM25_and_IDF:
                    y = ((self.K + 1) * tf) / (tf + self.K)
                    query_vector[idx] = y * self.IDF[idx]
                else:
                    query_vector[idx] = tf
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        if top_k is not None:
            top_k = min(top_k, len(self.doc_vectors))
            distances, indices = self.index.search(query_vector, top_k)
            return distances[0], indices[0].astype(int)
        else:
            distances, _ = self.index.search(query_vector, len(self.doc_vectors))
            return distances[0]