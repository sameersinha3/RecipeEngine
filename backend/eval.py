import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Ensure imports work when running directly
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.tfidf import TFIDF  # type: ignore
from backend.w2v import W2V      # type: ignore


def rrf_fusion(scores_list: List[np.ndarray], k: int = 60) -> np.ndarray:
    n = scores_list[0].shape[0]
    fused = np.zeros(n, dtype=float)
    for scores in scores_list:
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        fused += 1.0 / (k + ranks)
    return fused


def load_data(limit: int = 1000) -> pd.DataFrame:
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
            "chocolate cake",
            "vegan pasta",
            "gluten free cookies",
            "chicken curry",
            "beef stew",
            "vegetarian lasagna",
            "pizza",
            "salad",
            "soup",
            "pasta",
        ]

    print("Loading data...")
    df = load_data(limit=1000)
    n = len(df)
    print(f"Loaded {n} recipes (first 1K for speed)")

    print("Building TF-IDF...")
    tfidf_engine = TFIDF(df)
    tfidf_engine.build_vocabulary()
    tfidf_engine.compute_IDF()

    print("Building W2V vectors...")
    w2v_engine = W2V(df)

    for q in queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        print("=" * 80)

        tfidf_scores = tfidf_engine.execute_search_TF_IDF(q, applyBM25_and_IDF=True)
        w2v_scores = w2v_engine.rank_documents(q)
        fused_scores = rrf_fusion([tfidf_scores, w2v_scores])

        def show_top(label: str, scores: np.ndarray):
            order = np.argsort(scores)[::-1][:10]
            print(f"\nTop 10 - {label}")
            for rank, idx in enumerate(order, start=1):
                print(f"{rank:2d}. {format_result(idx, df.iloc[idx])}")

        show_top("TF-IDF", tfidf_scores)
        show_top("W2V", w2v_scores)
        show_top("RRF", fused_scores)


if __name__ == "__main__":
    main(sys.argv[1:])

