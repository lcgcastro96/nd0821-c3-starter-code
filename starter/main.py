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
    version="1.0.0",
)

PATH_MODEL = "model/model.pkl"
with open(PATH_MODEL, 'rb') as f:
    encoder, lb, model = pkl.load(f)

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
async def prediction(sample: Sample):
    """
    Prediction POST method
    Input:
        Input data
    Output:
        Input data prediction.
    """
    # Formatting input_data
    input_dict = {key.replace('_', '-'): [value] for key, value in sample.__dict__.items()}
    input_df = pd.DataFrame(input_dict)

    # categorical cols
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

    x_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get predictions and return
    pred = inference(model, x_data)
    return {"Result": "<=50K" if int(pred[0]) == 0 else ">50K"}
