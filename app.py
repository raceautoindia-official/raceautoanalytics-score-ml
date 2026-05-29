# app.py
import os
import time
from typing import Dict, List, Tuple, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

# ====== CONFIG ======
API_URL = os.getenv("SCORES_API_URL", "http://127.0.0.1:3003/api/saveScores")
API_TOKEN = os.getenv("ML_API_TOKEN", "")  # optional; sent if present
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
DEFAULT_CLUSTERS = int(os.getenv("DEFAULT_CLUSTERS", "3"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "60"))

app = Flask(__name__)
CORS(app, resources={
    r"/calculate_ranges": {"origins": [o.strip() for o in ALLOW_ORIGINS.split(",")]},
})

# cache keyed by (graphId, email, clusters)
_cache: Dict[Tuple[int, str, int], Dict[str, object]] = {}

# ====== HELPERS ======
def _get_headers() -> Dict[str, str]:
    # Only attach Authorization header if provided
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

def fetch_real_scores(graph_id: int, email: Optional[str]) -> List[Dict]:
    """
    Calls Next.js GET /api/saveScores?graphId=&email=
    Returns a flat list of {year_index, question_id, score} (skipping None/skipped).
    Also locally filters by graph_id/email as a safety net.
    """
    params = {"graphId": str(graph_id)}
    if email:
        params["email"] = email

    last_err = None
    for _ in range(3):
        try:
            r = requests.get(API_URL, headers=_get_headers(), params=params, timeout=15)
            ct = r.headers.get("Content-Type", "")
            if "application/json" not in ct:
                raise RuntimeError(f"Expected JSON, got {ct}")
            r.raise_for_status()
            data = r.json()
            subs = data.get("submissions", [])

            # Safety filter in case upstream isn’t filtering for some env
            all_scores: List[Dict] = []
            for sub in subs:
                if int(sub.get("graphId", -1)) != graph_id:
                    continue
                if email and sub.get("userEmail") != email:
                    continue
                for item in sub.get("scores", []):
                    if item.get("skipped") or item.get("score") is None:
                        continue
                    try:
                        yidx = int(item["yearIndex"])
                        qid = int(item["questionId"])
                        s = float(item["score"])
                        if np.isfinite(s):
                            all_scores.append({"year_index": yidx, "question_id": qid, "score": s})
                    except Exception:
                        # ignore malformed rows
                        continue
            return all_scores
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise RuntimeError(f"Failed to fetch scores: {last_err}")

def _cluster_range(values: np.ndarray, n_clusters: int) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0
    v = np.clip(v, 0.0, 10.0)

    if v.size < n_clusters:
        lo, hi = float(v.min()), float(v.max())
        return round(lo, 2), round(hi, 2)

    try:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = km.fit_predict(v.reshape(-1, 1))
        counts = np.bincount(labels)
        dom = v[labels == counts.argmax()]

        if dom.size >= 5:
            lo, hi = np.percentile(dom, [10, 90])
        elif np.unique(dom).size == 1:
            center = float(dom[0])
            lo, hi = max(0.0, center - 0.5), min(10.0, center + 0.5)
        else:
            lo, hi = float(dom.min()), float(dom.max())
        return round(max(0.0, lo), 2), round(min(10.0, hi), 2)
    except Exception:
        lo, hi = float(v.min()), float(v.max())
        return round(lo, 2), round(hi, 2)

def calculate_ranges_with_kmeans(scores: List[Dict], n_clusters: int) -> Dict:
    grouped = defaultdict(list)
    for s in scores:
        grouped[(s["year_index"], s["question_id"])].append(s["score"])

    result = []
    for (year_index, question_id), vals in grouped.items():
        lo, hi = _cluster_range(np.array(vals, dtype=float), n_clusters)
        result.append({
            "year_index": int(year_index),
            "question_id": int(question_id),
            "lower_range": lo,
            "upper_range": hi,
        })
    result.sort(key=lambda x: (x["year_index"], x["question_id"]))
    return {"data": result}

def _from_cache(key: Tuple[int, str, int]):
    slot = _cache.get(key)
    if not slot:
        return None
    if time.time() - slot["ts"] > CACHE_TTL_SEC:
        _cache.pop(key, None)
        return None
    return slot["payload"]

def _to_cache(key: Tuple[int, str, int], payload: Dict):
    _cache[key] = {"ts": time.time(), "payload": payload}

# ====== ROUTES ======
@app.get("/calculate_ranges")
def get_calculated_json():
    """
    GET /calculate_ranges?graphId=123[&email=user@x.com][&clusters=3][&cache=1]
    Returns: { data: [...] } where data is ONLY for the requested graphId (and email if provided).
    """
    try:
        graph_id_raw = request.args.get("graphId")
        if graph_id_raw is None:
            return jsonify({"error": "Missing required query param 'graphId'"}), 400
        try:
            graph_id = int(graph_id_raw)
        except ValueError:
            return jsonify({"error": "Invalid 'graphId' (must be integer)"}), 400

        email = request.args.get("email") or None
        n_clusters = int(request.args.get("clusters", DEFAULT_CLUSTERS))
        use_cache = request.args.get("cache", "1") == "1"

        cache_key = (graph_id, email or "", n_clusters)
        if use_cache:
            cached = _from_cache(cache_key)
            if cached is not None:
                return jsonify(cached)

        scores = fetch_real_scores(graph_id=graph_id, email=email)
        result = calculate_ranges_with_kmeans(scores, n_clusters=n_clusters)

        # add minimal meta (non-breaking)
        result["clusters"] = n_clusters
        result["count_groups"] = len(result["data"])

        if use_cache:
            _to_cache(cache_key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # local dev; run behind gunicorn/nginx in prod
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5501")), debug=False, threaded=True)


# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import requests
# import numpy as np
# from collections import defaultdict
# from sklearn.cluster import KMeans

# app = Flask(__name__)
# CORS(app)

# # fetch real score data from API
# def fetch_real_scores():
#     url = "http://145.223.18.170:3001/api/saveScores"
#     headers = {
#         "Authorization": "Bearer your-very-strong-random-string-here"
#     }
#     response = requests.get(url, headers=headers)
#     if "application/json" not in response.headers.get("Content-Type", ""):
#         raise Exception("Expected JSON response")
#     if response.status_code != 200:
#         raise Exception(f"API error {response.status_code}: {response.text}")
#     data = response.json()
#     if "submissions" not in data:
#         raise Exception("'submissions' key missing in response")
#     all_scores = []
#     for submission in data["submissions"]:
#         for item in submission.get("scores", []):
#             if item.get("skipped") or item.get("score") is None:
#                 continue
#             try:
#                 all_scores.append({
#                     "year_index": int(item["yearIndex"]),
#                     "question_id": int(item["questionId"]),
#                     "score": float(item["score"])
#                 })
#             except:
#                 continue
#     return all_scores
# # ML logic calculation 
# def calculate_ranges_with_kmeans(scores, n_clusters=3):
#     grouped = defaultdict(list)
#     result = []
#     for s in scores:
#         key = (s["year_index"], s["question_id"])
#         grouped[key].append(s["score"])

#     for (year_index, question_id), values in grouped.items():
#         if len(values) < n_clusters:
#             lower = round(min(values), 2)
#             upper = round(max(values), 2)
#         else:
#             try:
#                 data = np.array(values).reshape(-1, 1) 
#                 kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
#                 kmeans.fit(data)
#                 labels = kmeans.labels_
#                 dominant_cluster = np.bincount(labels).argmax()
#                 dominant_scores = data[labels == dominant_cluster].flatten()
#                 unique_scores = np.unique(dominant_scores)

#                 if len(unique_scores) == 1:
#                     center = unique_scores[0]
#                     lower = round(max(0, center - 0.5), 2)
#                     upper = round(min(10, center + 0.5), 2)
#                 else:
#                     lower = round(max(0, np.min(dominant_scores)), 2)
#                     upper = round(min(10, np.max(dominant_scores)), 2)
#             except Exception as e:
#                 print(f"KMeans failed for ({year_index}, {question_id}): {e}")
#                 lower = round(min(values), 2)
#                 upper = round(max(values), 2)
#         result.append({
#             "year_index": year_index,
#             "question_id": question_id,
#             "lower_range": lower,
#             "upper_range": upper
#         })

#     return {"data": sorted(result, key=lambda x: (x["year_index"], x["question_id"]))}

# # Calculate ranges from live API
# @app.route('/calculate_ranges', methods=['GET'])
# def get_calculated_json():
#     try:
#         scores = fetch_real_scores()
#         print(scores)
#         result = calculate_ranges_with_kmeans(scores, n_clusters=3)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # run flask api
# if __name__ == '__main__':
#     app.run(debug=True, port=5501)
