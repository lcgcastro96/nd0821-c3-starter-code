# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml import model
from ml.data import process_data
import pickly as pkl

# Definition of necessary variables
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
PATH_MODEL = "/model/model.pkl"
PATH_DATA = "/data/census.csv"

# Add code to load in the data.
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Process the test data with the process_data function
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
	test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
trained_model = model.train_model(X_train, y_train)
with open(MODEL_PATH, "wb") as file:
   pkl.dump([encoder, lb, mod], file)
