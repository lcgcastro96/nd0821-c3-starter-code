import json
import requests

data = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "India"
}

response = requests.post('http://127.0.0.1:8000', data=json.dumps(data))

print(response.status_code)
print(response.json())
