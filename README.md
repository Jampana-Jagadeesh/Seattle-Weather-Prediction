# Seattle Weather Prediction Project

[Google Colab File](https://colab.research.google.com/drive/18gpdE-45XwCFL71IjQ6cuLRvgUFvL3ef?usp=sharing)

## Overview

This project aims to analyze and predict weather patterns in Seattle using machine learning techniques. The dataset used in this project contains information about various weather parameters, such as precipitation, temperature, and wind, collected over a period of time.

## Dataset

The dataset, `seattle-weather.csv`, is loaded using the Pandas library in Python. It contains the following columns:

- `date`: Date of the recorded weather data.
- `precipitation`: Precipitation levels.
- `temp_max`: Maximum temperature.
- `temp_min`: Minimum temperature.
- `wind`: Wind speed.
- `weather`: Categorical variable indicating the weather condition (e.g., rain, sun, drizzle).

## Exploratory Data Analysis (EDA)

The EDA is performed using the Seaborn and Matplotlib libraries to visualize relationships and distributions within the dataset. Key visualizations include pair plots, count plots, scatter plots, histograms, and box plots.

## Data Preprocessing

The weather variable is encoded using Label Encoder for classification purposes. The `date` column is dropped during preprocessing.

## Machine Learning Models

The project employs several machine learning models for weather prediction:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- XGBoost Classifier

The models are trained and evaluated on the dataset, and accuracy scores are reported for each model.

## Model Comparison

A variety of machine learning models are compared, including Logistic Regression, Random Forest Classifier, Decision Tree Classifier, Support Vector Machine, and K-Nearest Neighbors. The models are trained, predictions are made, and accuracy scores are presented. Confusion matrices and classification reports are also included for further evaluation.

## Model Usage

The trained Gradient Boosting Classifier is used to predict the weather condition based on user input for precipitation, maximum temperature, minimum temperature, and wind speed.



