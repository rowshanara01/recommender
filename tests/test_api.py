import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
ratings = pd.read_csv("data/ratings.csv")


def test_recommendations_exclude_seen_items():
    seen = set(ratings[ratings.user_id == 1].item_id)
    recs = client.get("/recommend/1?k=10").json()["recommendations"]
    assert len(recs) == 10
    assert not seen & {r["item_id"] for r in recs}


def test_cold_start_user_gets_popular_items():
    r = client.get("/recommend/999999").json()
    assert r["cold_start"] is True and r["model"] == "popularity"


def test_similar_items_and_404():
    r = client.get("/similar-items/5?k=3").json()
    assert len(r["similar"]) == 3 and 5 not in {x["item_id"] for x in r["similar"]}
    assert client.get("/similar-items/99999").status_code == 404


def test_k_validation():
    assert client.get("/recommend/1?k=0").status_code == 422
    assert client.get("/recommend/1?k=500").status_code == 422
