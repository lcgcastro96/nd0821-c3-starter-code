import json
from fastapi.testclient import TestClient
import sys

try:
    from main import app
except ModuleNotFoundError:
    sys.path.append('./')
    from main import app

client = TestClient(app)

# root unit test
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the Salary Prediction App!"

# invoke incorrect path for get
def test_incorrect_path():
    """Test response for non existent path"""

    res = client.get("/foo")

    assert res.status_code != 200
    assert res.json() == {"detail":"Not Found"}

# invoke incorrect path
def test_incorrect_path_post():
    """Test response for non existent path"""

    res = client.post("/foo")

    assert res.status_code != 200
    assert res.json() == {"detail":"Not Found"}

# test a priori expected v2
def test_post_below_2():
    """Test for salary below 50K"""
    res = client.post("/model", json={
        "age": 41,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Separated",
        "occupation": "Handlers-cleaners",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == '<=50K'


# test a priori expected
def test_post_below():
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
    assert res.json() == '<=50K'

# test a priori expected
def test_post_above():
    """Test for salary below 50K"""
    res = client.post("/model", json={
        "age": 60,
        "workclass": "Private",
        "fnlgt": 141297,
        "education": "Masters",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == '>50K'