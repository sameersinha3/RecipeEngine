import numpy as np
import pandas as pd
from typing import Optional
import gensim.downloader as api
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class W2V:

    def __init__(self, dataset: pd.DataFrame, model_name: str = "glove-wiki-gigaword-100"):
        self.dataset = dataset.copy().reset_index(drop=True)
        self.model = None
        self.doc_vectors = None

        self._load_pretrained_model(model_name)

    def flatten(self, val):
        if isinstance(val, list):
            return " ".join(map(str, val))

        if pd.isna(val):
            return ""

        return str(val)

    def preprocess_text(self, text: str) -> list:
        return text.lower().split()

    def prepare_documents(self) -> list:
        documents = []

        for _, row in self.dataset.iterrows():
            parts = [
                self.flatten(row.get("title", "")),
                self.flatten(row.get("ingredients", "")),
                self.flatten(row.get("instructions", "")),
                self.flatten(row.get("tags", "")),
                self.flatten(row.get("description", "")),
            ]

            text = " ".join(parts)
            tokens = self.preprocess_text(text)
            documents.append(tokens)

        return documents

    def _load_pretrained_model(self, model_name: str):
        try:
            logger.info(f"Loading pre-trained Word2Vec model: {model_name}...")
            self.model = api.load(model_name)
            logger.info(f"Model loaded. Vocabulary size: {len(self.model)}")
            self._compute_doc_vectors()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _compute_doc_vectors(self):
        logger.info("Computing document vectors...")
        documents = self.prepare_documents()
        self.doc_vectors = []
        vector_size = self.model.vector_size

        for tokens in documents:
            # Get vectors for all words in document that exist in vocabulary
            vectors = []
            for token in tokens:
                try:
                    if token in self.model:
                        vectors.append(self.model[token])
                except KeyError:
                    pass

            # Compute mean vector, or zero vector if no words in vocab
            if vectors:
                doc_vec = np.mean(vectors, axis=0)
            else:
                doc_vec = np.zeros(vector_size)

            self.doc_vectors.append(doc_vec)

        self.doc_vectors = np.array(self.doc_vectors)
        logger.info(f"Computed vectors for {len(self.doc_vectors)} documents")

    def get_query_vector(self, query: str) -> np.ndarray:
        tokens = self.preprocess_text(query)
        vectors = []
        vector_size = self.model.vector_size

        for token in tokens:
            try:
                if token in self.model:
                    vectors.append(self.model[token])
            except KeyError:
                pass

        if vectors:
            query_vec = np.mean(vectors, axis=0)
        else:
            # If no words in vocabulary, return zero vector
            query_vec = np.zeros(vector_size)

        return query_vec

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def rank_documents(self, query: str) -> np.ndarray:
        query_vec = self.get_query_vector(query)
        scores = np.zeros(len(self.doc_vectors))

        for i, doc_vec in enumerate(self.doc_vectors):
            scores[i] = self.compute_cosine_similarity(query_vec, doc_vec)

        return scores