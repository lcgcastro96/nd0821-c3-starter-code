import json
import requests

data = {
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
}

response = requests.post('http://127.0.0.1:8000', data=json.dumps(data))
#response = requests.post('https://nd0821-c3-starter-code-udacity.herokuapp.com/', data=json.dumps(data))

print(response.status_code)
print(response.json())
