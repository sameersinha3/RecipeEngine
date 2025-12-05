import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# For imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
	sys.path.insert(0, WORKSPACE_ROOT)

from tfidf import TFIDF

app = Flask(__name__)

data_path = os.path.join(WORKSPACE_ROOT, "data", "processed_recipes.csv")
df = pd.read_csv(data_path)
engine_tfidf = TFIDF(df)
engine_tfidf.build_vocabulary()
n = df.shape[0]

def rrf_fusion(scores_list: List[np.ndarray], k: int = 60) -> np.ndarray:
	fused = np.zeros(n, dtype=float)

	for scores in scores_list:
		# argsort ascending, then reverse for descending
		order = np.argsort(scores)[::-1]
		ranks = np.empty_like(order)
		ranks[order] = np.arange(1, n + 1)
		fused += 1.0 / (k + ranks)

	return fused

def get_word2vec_scores(query: str) -> np.ndarray:
    # Placeholder
    scores = np.random.rand(n)
    return scores

@app.route("/search", methods=["POST"])
def search() -> Any:
	payload = request.get_json(silent=True) or {}
	query = payload.get("query", "").strip()
	top_k = min(n, int(payload.get("top_k", 20))) # Default return 20 results

	if not query:
		return jsonify({"error": "Missing 'query' in request body"}), 400


	tfidf_scores = engine_tfidf.execute_search_TF_IDF(query, applyBM25_and_IDF=True)
	w2v_scores = get_word2vec_scores(query)
	fused = rrf_fusion([tfidf_scores, w2v_scores])

	# Return results
	order = np.argsort(fused)[::-1][:top_k]
	results: List[Dict[str, Any]] = []
	for i in order:
		item = df.iloc[i].to_dict()
		results.append({
			"index": int(i),
			"title": item.get("title"),
			# Add other information
		})

	return jsonify({
		"results": results,
	})


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5000"))
	app.run(host="0.0.0.0", port=port)

