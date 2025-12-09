import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.tfidf import TFIDF 
from backend.w2v import W2V


def rrf_fusion(scores_list: List[np.ndarray], k: int = 60) -> np.ndarray:
    n = scores_list[0].shape[0]
    fused = np.zeros(n, dtype=float)
    for scores in scores_list:
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        fused += 1.0 / (k + ranks)
    return fused


def load_data(limit: int = 10000) -> pd.DataFrame:
    data_path = os.path.join(WORKSPACE_ROOT, "data", "processed_recipes.csv")
    df = pd.read_csv(data_path)
    df = df.head(limit)

    # Parse list-like columns
    for col in ["ingredients", "instructions", "tags"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(
                lambda x: x.split("|") if isinstance(x, str) and x else []
            )
    return df


def format_result(idx: int, row: pd.Series) -> str:
    title = row.get("title", "Untitled")
    rating = row.get("rating", None)
    total_time = row.get("total_time", None)
    rating_str = f"{rating:.1f}" if pd.notna(rating) else "NA"
    time_str = f"{int(total_time)}m" if pd.notna(total_time) else "NA"
    return f"[{idx}] {title} (rating: {rating_str}, time: {time_str})"


def main(queries: List[str]):
    if not queries:
        queries = [
            "healthy snacks",
            "comfort food",
            "low-carb",
            "budget meals",
            "vegan protein",
            "kid-friendly recipes",
        ]
    print("Loading data...")
    df = load_data(limit=10000)
    n = len(df)
    print(f"Loaded {n} recipes (first 1K for speed)")

    print("Building TF-IDF...")
    tfidf_engine = TFIDF(df)
    tfidf_engine.build_vocabulary()
    tfidf_engine.compute_IDF()
    print("  Building document vectors and FAISS index...")
    tfidf_engine.build_document_vectors()

    print("Building W2V vectors...")
    w2v_engine = W2V(df)

    for q in queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        print("=" * 80)

        tfidf_similarities, tfidf_indices = tfidf_engine.execute_search_TF_IDF(q, applyBM25_and_IDF=True, top_k=1000)
        w2v_similarities, w2v_indices = w2v_engine.rank_documents(q, top_k=1000)
        
        candidate_indices = np.unique(np.concatenate([tfidf_indices, w2v_indices]))
        
        tfidf_candidate_scores = np.zeros(len(candidate_indices))
        w2v_candidate_scores = np.zeros(len(candidate_indices))
        
        tfidf_score_map = dict(zip(tfidf_indices, tfidf_similarities))
        w2v_score_map = dict(zip(w2v_indices, w2v_similarities))
        
        for i, candidate_idx in enumerate(candidate_indices):
            tfidf_candidate_scores[i] = tfidf_score_map.get(candidate_idx, 0.0)
            w2v_candidate_scores[i] = w2v_score_map.get(candidate_idx, 0.0)
        
        candidate_fused = np.zeros(len(candidate_indices))
        k = 60
        for scores in [tfidf_candidate_scores, w2v_candidate_scores]:
            order = np.argsort(scores)[::-1]
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(candidate_indices) + 1)
            candidate_fused += 1.0 / (k + ranks)
        
        top_candidate_idx = np.argsort(candidate_fused)[::-1][:10]
        top_indices = candidate_indices[top_candidate_idx]
        
        tfidf_full_scores = np.zeros(n)
        w2v_full_scores = np.zeros(n)
        tfidf_full_scores[tfidf_indices] = tfidf_similarities
        w2v_full_scores[w2v_indices] = w2v_similarities
        
        fused_full = np.zeros(n)
        for scores in [tfidf_full_scores, w2v_full_scores]:
            order = np.argsort(scores)[::-1]
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, n + 1)
            fused_full += 1.0 / (k + ranks)

        def show_top(label: str, scores: np.ndarray, indices: np.ndarray = None):
            if indices is not None:
                order = indices[:10]
            else:
                order = np.argsort(scores)[::-1][:10]
            print(f"\nTop 10 - {label}")
            for rank, idx in enumerate(order, start=1):
                print(f"{rank:2d}. {format_result(idx, df.iloc[idx])}")

        show_top("TF-IDF", tfidf_full_scores)
        show_top("W2V", w2v_full_scores)
        show_top("RRF", None, top_indices)


if __name__ == "__main__":
    main(sys.argv[1:])

