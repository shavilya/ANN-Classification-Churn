# ANN Classification for Customer Churn Prediction

This repository contains the implementation of an Artificial Neural Network (ANN) to predict customer churn. The project is developed using **TensorFlow**, **Streamlit**, and **Scikit-learn**, with a focus on preprocessing inputs such as geography, gender, age, and other features.

## Overview

The goal of this project is to predict whether a customer will churn using an artificial neural network (ANN). The dataset used is `Churn_Modelling.csv`, and the model has been trained using TensorFlow.

The project includes:

- **`Streamlit Application`** (`app.py`): A web interface that takes user inputs and predicts if a customer is likely to churn.
- **.`Preprocessing Pipelines`**: Label encoding for gender and one-hot encoding for geography, as well as scaling for numerical features.
- **`Trained Model`**: The ANN model saved in `model.h5`.
- **`Experiments`**: The `experiments.ipynb` notebook includes the experiments conducted during model training.

## Files Explained

- **`Churn_Modelling.csv`**: This is the dataset used to train the model. It contains customer information like credit score, geography, gender, and whether the customer churned.
- **`app.py`**: The main Streamlit app where users can input data and get churn predictions.
- **`experiments.ipynb`**: A Jupyter notebook used for exploring and training the model.
- **`label_encoder_gender.pkl`**: A pickle file containing the `LabelEncoder` object for encoding the `Gender` column.
- **`ohe_geo.pkl`**: A pickle file containing the `OneHotEncoder` object for encoding the `Geography` column.
- **`scaler.pkl`**: A pickle file containing the `StandardScaler` object used to scale the features.
- **`model.h5`**: The trained ANN model saved in HDF5 format.
- **`predictions.ipynb`**: A notebook for testing and making predictions using the trained model.
- **`requirements.txt`**: Contains all the dependencies required to run this project (e.g., TensorFlow, Streamlit, Scikit-learn).

## How to Run the Project

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```
### 2.Install Dependencies 
Make sure you have Python installed on your machine. Install the project dependencies using pip:
```bash
pip install -r requirements.txt
```
### 3.Run the Streamtlit App 
To launch the Streamlit app for customer churn prediction, run the following command:
```bash
streamlit run app.py
```
### 4.Use the Jupyter Notebooks
1.experiments.ipynb: View this notebook to see how the model was trained.
2.predictions.ipynb: Use this notebook to make predictions with the trained model.

## Model Details
The ANN model used for this project was trained on the customer churn dataset using the following architecture:

- `Input Layer`: Numerical and encoded categorical features.
- `Hidden Layers`: Two dense hidden layers with ReLU activation.
- `Output Layer`: A single node with sigmoid activation for binary classification.
  
The model was trained to predict whether a customer is likely to churn based on various features such as credit score, age, geography, and balance.

### Preprocessing Steps:
- `Label Encoding`: The Gender column is label encoded into numerical values.
- `One-Hot Encoding `: The Geography column is one-hot encoded.
- `Scaling`: The numeric features like CreditScore, Age, Balance, etc., are scaled using the StandardScaler.

### How the Streamlit App Works
1.The user is prompted to enter customer details like geography, gender, age, balance, etc.<br>
2.The app processes these inputs using the saved encoders (label_encoder_gender.pkl, ohe_geo.pkl) and scaler (scaler.pkl).<br>
3.The processed inputs are then passed through the trained ANN model (model.h5) to predict whether the customer is likely to churn or not.<br>
4.The prediction result is displayed on the app.<br>

## License
This project is licensed under the terms of the LICENSE.

## Acknowledgement 
1.The dataset used in this project was sourced from the Churn Modelling dataset on Kaggle.<br>
2.Special thanks to Krish Naik for the guidance provided in the Generative AI course.<br>

