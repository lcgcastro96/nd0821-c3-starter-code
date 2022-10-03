from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def slice_performance(df, 
                      model, 
                      cat_columns, 
                      target, 
                      encoder, 
                      lb, 
                      path):
   """
   Function which outputs slice performance of the model

   Input:
      - df (Pandas DataFrame): input data
      - model (sklearn model): trained model
      - cat_columns (list): categorical columns
      - target (string): target variable
      - encoder (sklearn one hot encoder): one hot encoder
      - lb (sklearn label binarizer): label binarizer
      - path (string): output path

   Output:
     - metrics_df (Pandas DataFrame): output with performance metrics
   """

   # Create empty metrics df, to fill with each slice's performance
   m_columns = ["column_name", "slice", "precision", "recall", "f1-score"]
   metrics_list = []

   # Iterate through all categorical columns, evaluate each slice
   for col in cat_columns:
       curr_unique_vals = df[col].unique()
       # for each slice
       for val in curr_unique_vals:
           current_results = {}
           current_df = df[df[col] == val]
           x, y, _, _ = process_data(
              X = current_df,
              categorical_features = cat_columns,
              label = target,
              training = False,
              encoder = encoder,
              lb = lb
           )
           
           # generate predictions
           predictions = inference(model, x)

           # compute metrics and save them for this slice
           precision, recall, fscore = compute_model_metrics(y, predictions)
           current_results['column_name'] = col
           current_results['slice'] = val
           current_results['precision'] = precision
           current_results['recall'] = recall
           current_results['f1-score'] = fscore
           metrics_list.append(current_results)
       
   metrics_df = pd.DataFrame(metrics_list, columns = m_columns)
  
   # write to disk
   if(path):
       metrics_df.to_csv(path, index = False)

   return metrics_df
