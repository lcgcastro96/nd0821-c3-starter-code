# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml import model
from ml.data import process_data
from ml.model import inference, compute_model_metrics
import pickle as pkl

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
PATH_MODEL = "./model/model.pkl"
PATH_DATA = "./data/census.csv"
PATH_SLICE = "./model/slice_output.txt"
PATH_PERF = "./model/performance.csv"

# Add code to load in the data.
data = pd.read_csv(PATH_DATA)

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
with open(PATH_MODEL, "wb") as file:
   pkl.dump([encoder, lb, trained_model], file)

# Get performance on test set
preds = inference(trained_model, X_test)
perf = compute_model_metrics(y_test, preds)


# Get slice performance
model.slice_performance(
    df = test, 
    model = trained_model, 
    cat_columns = cat_features, 
    target = "salary", 
    encoder = encoder, 
    lb = lb, 
    path = PATH_SLICE)
