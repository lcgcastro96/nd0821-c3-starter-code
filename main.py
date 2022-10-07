import os
import pickle as pkl
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI(
    title="Salary Model",
    description="Given census data, it will predict if the person earns above 50K",
    version="1.1.0",
)

PATH_MODEL = "model/model.pkl"
with open(PATH_MODEL, 'rb') as f:
    encoder, lb, model = pkl.load(f)

# Pydantic sample class
class Sample(BaseModel):
    age: int = Field(None)
    workclass: str = Field(None)
    fnlgt: int = Field(None)
    education: str = Field(None)
    education_num: int = Field(None)
    marital_status: str = Field(None)
    occupation: str = Field(None)
    relationship: str = Field(None)
    race: str = Field(None)
    sex: str = Field(None)
    capital_gain: int = Field(None)
    capital_loss: int = Field(None)
    hours_per_week: int = Field(None)
    native_country: str = Field(None)


@app.get("/")
async def hello_world():
    """
    Welcome function for returning home directory.
    Output:
       GET request home welcome message
    """
    return 'Welcome to the Salary Prediction App!'

@app.post("/model")
async def predict(sample: Sample):
    sample = {key.replace('_', '-'): [value] for key, value in sample.__dict__.items()}
    data = pd.DataFrame.from_dict(sample)

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

    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, 
        training=False, encoder=encoder, lb=lb)

    pred = int(inference(model, X)[0])
    return '<=50K' if pred == 0 else '>50K'
