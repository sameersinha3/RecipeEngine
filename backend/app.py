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
print("[1/5] Loading recipe data from CSV...")
data_path = os.path.join(WORKSPACE_ROOT, "data", "processed_recipes.csv")
df = pd.read_csv(data_path)
df = df.head(10000)
print(f"Loaded {len(df):,} recipes")

print("\n[2/5] Parsing ingredients, instructions, and tags...")
tqdm.pandas()
for col in ['ingredients', 'instructions', 'tags']:
	if col in df.columns:
		df[col] = df[col].fillna('').progress_apply(
			lambda x: x.split('|') if isinstance(x, str) and x else []
		)

print("\n[3/5] Building TF-IDF...")
engine_tfidf = TFIDF(df)
engine_tfidf.build_vocabulary()
print(f"Vocabulary built ({len(engine_tfidf.vocab)} words)")
print("  Pre-computing IDF values...")
engine_tfidf.compute_IDF()
print("  IDF computed")
print("  Building document vectors and FAISS index...")
engine_tfidf.build_document_vectors()

print("\n[4/5] Loading Word2Vec model and computing document vectors...")
engine_w2v = W2V(df)

print("\n[5/5] Initialization complete!")
print("=" * 60)

n = df.shape[0]


def perform_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
	if not query:
		return []

	top_k = min(n, top_k)
	
	# Use FAISS to get top 1000 candidates from each method
	candidate_size = 1000
	
	# Get top candidates from TF-IDF using FAISS
	tfidf_similarities, tfidf_candidate_indices = engine_tfidf.execute_search_TF_IDF(
		query, applyBM25_and_IDF=True, top_k=candidate_size
	)
	
	# Get top candidates from W2V using FAISS
	w2v_similarities, w2v_candidate_indices = engine_w2v.rank_documents(query, top_k=candidate_size)
	
	# Merge candidate sets (union of top candidates from both methods)
	candidate_indices = np.unique(np.concatenate([tfidf_candidate_indices, w2v_candidate_indices]))
	
	# Create score arrays for candidates only
	tfidf_candidate_scores = np.zeros(len(candidate_indices))
	w2v_candidate_scores = np.zeros(len(candidate_indices))
	
	# Map FAISS results to candidate indices
	tfidf_score_map = dict(zip(tfidf_candidate_indices, tfidf_similarities))
	w2v_score_map = dict(zip(w2v_candidate_indices, w2v_similarities))
	
	# Fill in scores for candidates
	for i, candidate_idx in enumerate(candidate_indices):
		tfidf_candidate_scores[i] = tfidf_score_map.get(candidate_idx, 0.0)
		w2v_candidate_scores[i] = w2v_score_map.get(candidate_idx, 0.0)
	
	# RRF fusion on candidate set only (much smaller than full dataset)
	candidate_fused = np.zeros(len(candidate_indices))
	k = 60
	
	for scores in [tfidf_candidate_scores, w2v_candidate_scores]:
		order = np.argsort(scores)[::-1]
		ranks = np.empty_like(order)
		ranks[order] = np.arange(1, len(candidate_indices) + 1)
		candidate_fused += 1.0 / (k + ranks)
	
	# Get top-k from fused candidate scores
	top_candidate_idx = np.argsort(candidate_fused)[::-1][:top_k]
	order = candidate_indices[top_candidate_idx]
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

