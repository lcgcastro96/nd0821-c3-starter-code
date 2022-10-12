import json
from fastapi.testclient import TestClient
import sys
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

# imports
try:
    from main import app
    from ml.data import process_data
    from ml.model import inference, train_model, compute_model_metrics
except ModuleNotFoundError:
    sys.path.append('./')
    sys.path.append('./starter/')
    from main import app
    from ml.data import process_data
    from ml.model import inference, train_model, compute_model_metrics

client = TestClient(app)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

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

# input ddata
@pytest.fixture()
def input_data():
    df = pd.read_csv("./data/census.csv")
    train, test = train_test_split(df, test_size=0.2)
    return train, test

# test the inference, prediction array length needs to be equal to x_train
def test_inference(input_data):

    train_df, _ = input_data

    X_train, y_train, _, _ = process_data(
        X=train_df,
        categorical_features=cat_features,
        label='salary',
        training=True
    )

    tm = train_model(X_train, y_train)
    preds = inference(tm, X_train)

    assert len(preds) == len(X_train)


# assert that the metrics are all above 0 and below 100
def test_compute_metrics(input_data):

    train, _ = input_data

    X_train, y_train, encoder, lb = process_data(
        X=train,
        categorical_features=cat_features,
        label='salary',
        training=True
    )

    clf = train_model(X_train, y_train)
    preds = inference(clf, X_train)
    pr, rc, fscore = compute_model_metrics(y_train, preds)

    # Assert no metric has a value above 1.0
    for metric in [pr, rc, fscore]:
        assert metric <= 1.0
        assert metric >= 0.0