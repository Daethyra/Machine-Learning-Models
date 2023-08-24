# Prediction Phase for Financial Aid Analysis
*Prediction Work in Progress 1*

*`(Creation Date: 8 / 23 / 2023)`*

## Introduction

The prediction phase is an essential part of our project, aimed at identifying regions most in need of foreign aid based on economic, health, and social indicators. With a well-trained Gaussian Mixture Model (GMM), we proceed to apply the model to new data to make insightful predictions.

## Prediction Process

### 1. Loading the Trained Model

* Utilized the serialized model file `FinancialAidGMM_model.pkl` containing the trained Gaussian Mixture Model.
* The model was trained on preprocessed data, considering various socio-economic factors.

### 2. Data Preparation for Prediction

* New data must be properly preprocessed using the same transformations applied during the training phase.
* This includes handling missing values, converting currency and percentage columns, and applying scaling based on global wealth inequality.

### 3. Making Predictions

* The trained GMM model was applied to new data to predict the cluster memberships, representing different levels of financial aid needs.
* The prediction process considers the prioritized features, including Agricultural Land(%), Unemployment Rate, Tax Revenue(%), Out of Pocket Health Expenditure, and health indicators like Life Expectancy, Infant Mortality, Maternal Mortality Ratio.

### 4. Analysis and Visualization

* Analyzed the predicted clusters to understand the regions' characteristics and needs.
* Visualization techniques were used to represent the clusters and interpret the findings effectively.

## How to Use the Prediction Script

A Python script named `predict.py` encapsulates the prediction process. Here's an example of how to use it:

```python
from predict import FinancialAidPredictor

predictor = FinancialAidPredictor()
predictions = predictor.predict_new_data(new_data)
```

## Conclusion

The prediction phase leveraged the trained GMM model to identify regions in need of foreign aid. Careful preparation of the data and insightful analysis of the predictions allowed for effective decision-making. The collaboration continues to focus on refining the process and utilizing the predictions to make informed foreign aid decisions.