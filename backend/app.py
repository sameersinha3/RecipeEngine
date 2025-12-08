import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tqdm import tqdm

# For imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if WORKSPACE_ROOT not in sys.path:
	sys.path.insert(0, WORKSPACE_ROOT)

from tfidf import TFIDF
from w2v import W2V

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("=" * 60)
print("Initializing Recipe Search Engine...")
print("=" * 60)
data_path = os.path.join(WORKSPACE_ROOT, "data", "processed_recipes.csv")
df = pd.read_csv(data_path)
# Limit to first 5K documents for faster searches
df = df.head(1000)
print(f"Loaded {len(df):,} recipes (limited to 1K for performance)")

tqdm.pandas()
for col in ['ingredients', 'instructions', 'tags']:
	if col in df.columns:
		df[col] = df[col].fillna('').progress_apply(
			lambda x: x.split('|') if isinstance(x, str) and x else []
		)

engine_tfidf = TFIDF(df)
engine_tfidf.build_vocabulary()
print(f"Vocabulary built ({len(engine_tfidf.vocab)} words)")
print("  Pre-computing IDF values...")
engine_tfidf.compute_IDF()
print("  IDF computed")

print("\n[4/5] Loading Word2Vec model and computing document vectors...")
print("  (This may take several minutes...)")
engine_w2v = W2V(df)

print("\n[5/5] Initialization complete!")
print("=" * 60)

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

def perform_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
	if not query:
		return []

	top_k = min(n, top_k)

	# Compute scores for both methods (fast with 5K docs and vectorized operations)
	tfidf_scores = engine_tfidf.execute_search_TF_IDF(query, applyBM25_and_IDF=True)
	w2v_scores = engine_w2v.rank_documents(query)
	
	# RRF fusion on all documents
	fused = rrf_fusion([tfidf_scores, w2v_scores])
	
	order = np.argsort(fused)[::-1][:top_k]
	results: List[Dict[str, Any]] = []
	
	# Helper function to handle NaN values for JSON serialization
	def clean_value(val):
		if pd.isna(val):
			return None
		if isinstance(val, (np.integer, np.int64)):
			return int(val)
		if isinstance(val, (np.floating, np.float64)):
			return float(val) if not pd.isna(val) else None
		return val
	
	for i in order:
		item = df.iloc[i].to_dict()
		
		ingredients = item.get("ingredients", [])
		if isinstance(ingredients, str):
			ingredients = ingredients.split('|') if ingredients else []
		
		tags = item.get("tags", [])
		if isinstance(tags, str):
			tags = tags.split('|') if tags else []
		
		instructions = item.get("instructions", [])
		if isinstance(instructions, str):
			instructions = instructions.split('|') if instructions else []
		
		results.append({
			"index": int(i),
			"title": clean_value(item.get("title")),
			"ingredients": ingredients,
			"instructions": instructions,
			"tags": tags,
			"prep_time": clean_value(item.get("prep_time")),
			"cook_time": clean_value(item.get("cook_time")),
			"total_time": clean_value(item.get("total_time")),
			"n_steps": clean_value(item.get("n_steps")),
			"n_ingredients": clean_value(item.get("n_ingredients")),
			"rating": clean_value(item.get("rating")),
			"review_count": clean_value(item.get("review_count")),
			"author_id": clean_value(item.get("author_id")),
			"author_name": clean_value(item.get("author_name")),
			"category": clean_value(item.get("category")),
			"description": clean_value(item.get("description")),
		})

	return results

@app.route("/api/search", methods=["POST"])
def search() -> Any:
	try:
		payload = request.get_json(silent=True) or {}
		query = payload.get("query", "").strip()
		top_k = min(n, int(payload.get("top_k", 20)))

		if not query:
			return jsonify({"error": "Missing 'query' in request body"}), 400

		results = perform_search(query, top_k)
		return jsonify({"results": results})
	except Exception as e:
		print(f"Search error: {e}", flush=True)
		import traceback
		traceback.print_exc()
		return jsonify({"error": f"Search failed: {str(e)}"}), 500

frontend_dir = os.path.join(WORKSPACE_ROOT, "frontend")

@app.route("/")
def index():
	return send_from_directory(frontend_dir, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
	return send_from_directory(frontend_dir, filename)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5001"))
	print(f"Server starting at http://localhost:{port}/")
	print(f"API endpoint: http://localhost:{port}/api/search")
	app.run(host="0.0.0.0", port=port)

