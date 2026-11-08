"""Recommendation API: personalized, cold-start fallback, similar items"""
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query

B = joblib.load("model/recommender.joblib")
uidx = {int(u): i for i, u in enumerate(B["user_ids"])}
iidx = {int(i): j for j, i in enumerate(B["item_ids"])}
ITEMS = {r["item_id"]: r for r in B["items"]}
BEST = B["best"]

app = FastAPI(title="Recommender API")


def pack(order, scores):
    out = []
    for j in order:
        if not np.isfinite(scores[j]):
            continue
        meta = ITEMS[int(B["item_ids"][j])]
        out.append({"item_id": int(meta["item_id"]), "title": meta["title"],
                    "genre": meta["genre"], "score": round(float(scores[j]), 4)})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": BEST}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50)):
    if user_id in uidx:
        u = uidx[user_id]
        scores = np.where(B["seen"][u], -np.inf, B["scores"][BEST][u])
        cold = False
    else:  # cold start: notun user, history nai -> popular movie dekhao
        scores, cold = B["pop_score"].copy(), True
    order = np.argsort(-scores)[:k]
    return {"user_id": user_id, "model": "popularity" if cold else BEST,
            "cold_start": cold, "recommendations": pack(order, scores)}


@app.get("/similar-items/{item_id}")
def similar_items(item_id: int, k: int = Query(5, ge=1, le=50)):
    if item_id not in iidx:
        raise HTTPException(404, "item not found")
    s = B["sim"][iidx[item_id]]
    return {"item_id": item_id, "similar": pack(np.argsort(-s)[:k], s)}
