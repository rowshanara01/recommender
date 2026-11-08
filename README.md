# Recommender System (Netflix/Amazon style) #

## Chalanor niyom ##
pip install -r requirements.txt
python generate_data.py
python train.py                    # 3 model compare: popularity vs item-kNN vs SVD
pytest -q
uvicorn app.main:app --reload      # localhost:8000/docs
  GET /recommend/1?k=10            # personalized
  GET /recommend/999999            # cold-start (notun user)
  GET /similar-items/5             # "ei movie jara dekhe, tara eta-o dekhe"

## Company te ja ja use hoy (ei project e ache) ##
- Baseline age (popularity), tarpor complex model; jodi complex model baseline er cheye bhalo na hoy, dorkar nai
- Collaborative Filtering (item-kNN) + Matrix Factorization (SVD)
- Ranking metric: Precision@K, Recall@K, NDCG@K (accuracy na), catalog coverage
- Per-user holdout evaluation, already-dekha item bad dewa
- Cold-start handling (notun user -> popularity fallback)
- API validation, tests, Docker

## Challenge
1. SVD er n_components (4, 8, 16, 32) validation split diye tune koro
2. Item genre (items.csv) use kore content-based + hybrid model banao
3. Real dataset: MovieLens 100K diye replace koro
4. Implicit feedback (click/view) version banao (rating nai)
5. Online A/B test er plan lekho: kon metric dekhe decide korbe?
6. Redis diye recommendation cache koro
