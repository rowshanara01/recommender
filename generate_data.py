"""Synthetic movie ratings: 600 user, 200 movie, hidden taste (latent factors) + popularity bias"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_users, n_items, k = 600, 200, 4
U = rng.normal(0, 1, (n_users, k))          # user er hidden pochondo
V = rng.normal(0, 1, (n_items, k))          # movie er hidden dhoron
pop = rng.zipf(1.6, n_items).clip(1, 50).astype(float)
pop /= pop.sum()                            # kichu movie onek popular (real life er moto)

rows = []
for u in range(n_users):
    items = rng.choice(n_items, rng.integers(15, 45), replace=False, p=pop)
    for i in items:
        r = 3.2 + 0.55 * U[u] @ V[i] + rng.normal(0, 0.4)
        rows.append((u + 1, i + 1, int(np.clip(round(r), 1, 5))))

pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).to_csv("data/ratings.csv", index=False)
genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance"]
pd.DataFrame({"item_id": np.arange(1, n_items + 1),
              "title": [f"Movie {i}" for i in range(1, n_items + 1)],
              "genre": rng.choice(genres, n_items)}).to_csv("data/items.csv", index=False)
print("ratings:", len(rows), "| sparsity:", round(1 - len(rows) / (n_users * n_items), 3))
