"""Recommender: popularity vs item-kNN vs SVD (matrix factorization) -> ranking metrics -> save"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

K = 10
ratings = pd.read_csv("data/ratings.csv")
items = pd.read_csv("data/items.csv")
user_ids = np.sort(ratings.user_id.unique())
item_ids = np.sort(items.item_id.unique())
uidx = {u: i for i, u in enumerate(user_ids)}
iidx = {i: j for j, i in enumerate(item_ids)}
nu, ni = len(user_ids), len(item_ids)


def to_matrix(df):
    M = np.zeros((nu, ni))
    M[df.user_id.map(uidx).values, df.item_id.map(iidx).values] = df.rating.values
    return M


# ---- Split: protita user er 20% rating test e (time/user-wise holdout) ----
rng = np.random.default_rng(42)
test_mask = np.zeros(len(ratings), bool)
for _, idx in ratings.groupby("user_id").indices.items():
    test_mask[rng.choice(idx, max(1, int(0.2 * len(idx))), replace=False)] = True
R = to_matrix(ratings[~test_mask])      # train
T = to_matrix(ratings[test_mask])       # test
rated = R > 0
global_mean = R[rated].mean()
user_mean = np.where(rated.sum(1) > 0, R.sum(1) / np.maximum(rated.sum(1), 1), global_mean)
Rc = (R - user_mean[:, None]) * rated   # user-wise mean-centered (kono user beshi generous, kono user kripon)

# ---- Model 1: Popularity (baseline, shobar age eta dekhte hoy) ----
m = 5  # bayesian shrinkage: kom rating wala movie er score ke global mean er dike tani
pop_score = (R.sum(0) + m * global_mean) / (rated.sum(0) + m)
S_pop = np.tile(pop_score, (nu, 1))

# ---- Model 2: Item-based Collaborative Filtering ----
sim = cosine_similarity(Rc.T)
np.fill_diagonal(sim, 0)
S_knn = (Rc @ sim) / (rated @ np.abs(sim) + 5)

# ---- Model 3: Matrix Factorization (SVD) ----
svd = TruncatedSVD(n_components=8, random_state=42)
uf = svd.fit_transform(Rc)
S_svd = uf @ svd.components_ + user_mean[:, None]

models = {"popularity": S_pop, "item_knn": S_knn, "svd": S_svd}


def ranking_metrics(S, k=K):
    prec, rec, ndcg, recommended = [], [], [], set()
    for u in range(nu):
        relevant = set(np.where(T[u] >= 4)[0])
        if not relevant:
            continue
        s = np.where(rated[u], -np.inf, S[u])           # already dekha movie recommend korbo na
        top = np.argsort(-s)[:k]
        recommended.update(top.tolist())
        hits = [1 if j in relevant else 0 for j in top]
        prec.append(sum(hits) / k)
        rec.append(sum(hits) / len(relevant))
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg)
    return {f"precision@{k}": float(np.mean(prec)), f"recall@{k}": float(np.mean(rec)),
            f"ndcg@{k}": float(np.mean(ndcg)), "catalog_coverage": len(recommended) / ni}


def rmse(S):
    mask = T > 0
    return float(np.sqrt(np.mean((np.clip(S, 1, 5)[mask] - T[mask]) ** 2)))


results = {}
for name, S in models.items():
    results[name] = ranking_metrics(S)
    if name != "item_knn":  # knn score rating na, tai RMSE onorthok
        results[name]["rmse"] = rmse(S)
    print(f"{name:11s}", {k: round(v, 3) for k, v in results[name].items()})

best = max(results, key=lambda n: results[n][f"ndcg@{K}"])
print("\nbest (by NDCG@10):", best)

# ---- Final: shob rating (train+test) dekhe serving, seen mask full data theke ----
seen = to_matrix(ratings) > 0
joblib.dump({"user_ids": user_ids, "item_ids": item_ids, "scores": models, "sim": sim,
             "seen": seen, "pop_score": pop_score, "best": best,
             "items": items.to_dict("records")}, "model/recommender.joblib")
json.dump({"best": best, "k": K, "results": results}, open("model/metrics.json", "w"), indent=2)
print("saved model/recommender.joblib")
