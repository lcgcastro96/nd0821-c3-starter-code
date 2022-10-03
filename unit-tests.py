import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# root unit test
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the Salary Prediction App!"

# invoke incorrect path
def test_incorrect_path(client):
    """Test response for non existent path"""

    res = client.get("/foo")

    assert res.status_code != 200
    assert res.json() == {"detail":"Not Found"}

# test a priori expected
def test_post_below(client):
    """Test for salary below 50K"""
    res = client.post("/model", json={
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Separated",
        "occupation": "Prof-specialty",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'Result': '<=50K'}