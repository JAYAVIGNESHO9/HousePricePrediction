# House Price Prediction using Machine Learning

##  Overview

This project predicts the price of a house based on its features such as area, number of bedrooms, number of floors, year built, location, condition, and garage availability. The goal of the project is to demonstrate how machine learning can be used to solve real-world problems like property price estimation.

The model is trained using historical housing data and can generate price predictions based on user-provided inputs through a simple command-line interface.



##  Project Objectives

* Understand and implement a complete machine learning workflow
* Clean and preprocess real-world structured data
* Train and evaluate regression models
* Build a system that can take user input and generate predictions
* Save and reuse trained models for future predictions



##  Machine Learning Model Used

This project uses **Random Forest Regression**, a powerful ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. Linear Regression was also considered as a baseline model for comparison.

---

##  Features Used for Prediction

The model predicts house prices using the following features:

* Area
* Bedrooms
* Bathrooms
* Floors
* Year Built
* Location
* Condition
* Garage Availability

Categorical features such as Location, Condition, and Garage are converted into numerical format using one-hot encoding before training the model.

---

##  Technologies and Libraries

* Python
* Pandas (data handling)
* NumPy (numerical operations)
* Scikit-learn (machine learning models and evaluation)
* Joblib (model saving and loading)

---



##  How the Project Works

1. The dataset is loaded and cleaned.
2. Categorical features are encoded into numerical form.
3. The model is trained using the training data.
4. The trained model is saved for future use.
5. The user enters house details.
6. The model predicts the estimated price based on those inputs.



##  Example Usage
![Prediction Output](images/prediction_output.png)

##  Model Evaluation

The model is evaluated using the following metrics:

* **RMSE (Root Mean Squared Error)** – measures prediction error
* **R² Score** – indicates how well the model explains the variance in house prices

These metrics help assess how accurate and reliable the predictions are.



##  Learning Outcomes

Through this project, the following concepts were implemented and understood:

* Data preprocessing and feature engineering
* Handling categorical variables using one-hot encoding
* Training regression models using scikit-learn
* Evaluating models using statistical metrics
* Building a user-interactive prediction system



This project was developed as part of a machine learning learning exercise to understand regression models and real-world data workflows.
