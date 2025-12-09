import numpy as np
import pandas as pd
from typing import Optional
import gensim.downloader as api
import logging
from tqdm import tqdm
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class W2V:

    def __init__(self, dataset: pd.DataFrame, model_name: str = "glove-wiki-gigaword-100"):
        self.dataset = dataset.copy().reset_index(drop=True)
        self.model = None
        self.doc_vectors = None
        self.index = None 

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

        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc="  Preparing documents", leave=False):
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
            print(f"  Loading Word2Vec model: {model_name}...", end=" ", flush=True)
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

        print(f"  Computing vectors for {len(documents):,} documents...", flush=True)
        for tokens in tqdm(documents, desc="  Computing document vectors", unit=" docs"):
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

        self.doc_vectors = np.array(self.doc_vectors).astype('float32')
        logger.info(f"Computed vectors for {len(self.doc_vectors)} documents")
        print(f"  Computed {len(self.doc_vectors):,} document vectors")
        
        print(f"  Building FAISS index...", end=" ", flush=True)
        vector_dim = self.doc_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dim)
        
        faiss.normalize_L2(self.doc_vectors)
        self.index.add(self.doc_vectors)

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

    def rank_documents(self, query: str, top_k: Optional[int] = None):
        query_vec = self.get_query_vector(query)
        
        if query_vec is None or np.linalg.norm(query_vec) == 0:
            if top_k is None:
                return np.zeros(len(self.doc_vectors))
            else:
                return np.zeros(top_k), np.zeros(top_k, dtype=int)
        
        query_vec = query_vec.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        if top_k is not None:
            top_k = min(top_k, len(self.doc_vectors))
            distances, indices = self.index.search(query_vec, top_k)
            return distances[0], indices[0].astype(int)
        else:
            distances, _ = self.index.search(query_vec, len(self.doc_vectors))
            return distances[0]