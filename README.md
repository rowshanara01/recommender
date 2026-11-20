#### Recommender System (Netflix / Amazon Style) ####

A recommendation engine that suggests items to users based on their past behavior and on what similar users like.

#### How to Run ####

```bash
pip install -r requirements.txt
python generate_data.py
python train.py                    # compares 3 models: popularity vs item-kNN vs SVD
pytest -q
uvicorn app.main:app --reload      # localhost:8000/docs
```

APIs endpoints:

```http
GET /recommend/1?k=10              # personalized recommendations
GET /recommend/999999              # cold-start (new user)
GET /similar-items/5               # "people who watched this also watched..."
```

#### What Companies Use in Practice (and This Project Includes)

- **Baseline first** (popularity), then more complex models. If a complex model doesn't beat the baseline, it isn't needed.
- **Collaborative Filtering** (item-kNN) and **Matrix Factorization** (SVD).
- **Ranking metrics:** Precision@K, Recall@K, NDCG@K (not plain accuracy), plus catalog coverage.
- **Per-user holdout evaluation**, excluding items the user has already seen.
- **Cold-start handling:** new users fall back to popularity.
- API validation, tests, and Docker.

#### Challenges ####

1. Tune SVD `n_components` (4, 8, 16, 32) using a validation split.
2. Use item genres (`items.csv`) to build a content-based and hybrid model.
3. Replace the data with a real dataset: MovieLens 100K.
4. Build an implicit feedback version (clicks/views, no ratings).
5. Write an online A/B test plan: which metrics will you use to decide?
6. Cache recommendations with Redis.
