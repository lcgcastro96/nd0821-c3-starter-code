# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest Classifier. Performs binary classification, using census data, on whether an individual has an income superior (or not) to 50K USD.

## Intended Use
Model intended for educational use.

## Training Data
Census Bureau data, publicly available.

## Evaluation Data
Small split of 20% from the same Census Bureau data mentioned above.

## Metrics
Precision = ~73,20%
Recall = ~61,74%
F1-Score = ~66,99%

## Ethical Considerations
Model may be biased based on the data it used, especially regarding gender, race and workclass.

## Caveats and Recommendations
Better feature engineering may improve results.
