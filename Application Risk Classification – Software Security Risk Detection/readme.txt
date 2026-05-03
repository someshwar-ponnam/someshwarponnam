Project 2 – Application Risk Classification
Overview

This project focuses on building a machine learning model that can classify whether a software application represents a high security risk based on behavioural feature data.

The dataset contains application behaviour represented as sparse feature vectors. Each record includes a risk score and a set of feature index–value pairs describing the behaviour of the application.

To convert this into a classification problem, applications with a risk score ≥ 0.3 are labelled as High Risk, while those below this threshold are labelled as Low Risk.

The goal of the project is to:

Process raw behavioural data into a structured dataset

Train a machine learning model

Evaluate the model’s performance

Package the trained model for reuse

Demonstrate how the model can be used to classify new applications

Project Structure

The project contains the following key files:

Project2/
│
├── train_model.py
├── test_model_script.py
├── feature_name_to_number_mapping.csv
├── project2_raw_data/
│   ├── *.txt
│
├── application_risk_model.pkl
├── feature_importance.png
├── model_evaluation.txt
│
├── Project2_Documentation.pdf
└── README.md
File Descriptions

train_model.py

Main training script that:

Processes raw dataset files

Cleans the data

Creates training, validation and test splits

Trains the XGBoost model

Evaluates performance

Saves the trained model

test_model_script.py

A script demonstrating how to use the trained model for predictions.

It loads the packaged model and generates predictions for a sample dataset.

feature_name_to_number_mapping.csv

Mapping file used to convert feature indices from the raw data into structured feature names.

project2_raw_data/

Folder containing the raw behavioural dataset in sparse text format.

application_risk_model.pkl

Serialized machine learning model saved using Python pickle.
This file allows the trained model to be reused without retraining.

feature_importance.png

Visualisation of the top 20 most important features used by the model.

model_evaluation.txt

Text file containing the validation performance metrics produced during training.

Project2_Documentation.pdf

Full report describing the project approach, data processing, model development, and evaluation.

Dataset Format

Each line in the raw dataset follows this structure:

risk_score feature_index:value feature_index:value ...

Example:

0.82 1:0.54 5:1.00 14:0.23 27:0.61

Where:

risk_score represents the initial risk score

feature_index:value pairs represent behavioural features

A binary label is created using the rule:

risk_score >= 0.3 → High Risk
risk_score < 0.3 → Low Risk
Model Details

The final model used in this project is an XGBoost Gradient Boosting Classifier.

XGBoost was selected because it performs well on structured tabular datasets and is widely used in machine learning applications.

Model Parameters
n_estimators = 800
max_depth = 10
learning_rate = 0.03
subsample = 0.8
colsample_bytree = 0.8
objective = binary:logistic
eval_metric = logloss

The model outputs probabilities indicating the likelihood that an application is high risk.

Decision Threshold

Predictions are generated using a custom probability threshold.

Probability ≥ 0.45 → High Risk
Probability < 0.45 → Low Risk

This threshold provides a good balance between precision and recall for the classification task.

Model Performance

Example performance results on the held-out test set:

Precision: ~0.926
Recall: ~0.979
Accuracy: ~0.91

These results demonstrate strong predictive performance and exceed the project requirement of 90% performance.

Running the Training Script

To train the model, run:

python train_model.py

Optional arguments can be used to specify custom paths:

python train_model.py --data_dir project2_raw_data --feature_map feature_name_to_number_mapping.csv

Running the script will:

Train the model

Generate evaluation metrics

Save the trained model

Save the feature importance plot

Running the Test Script

To test the trained model, run:

python test_model_script.py

The script will:

Load the saved model

Read a sample input dataset

Generate predictions

Output risk classifications

Example output:

Sample 1: High Risk (Probability: 0.94)
Sample 2: Low Risk (Probability: 0.12)
Conclusion

This project demonstrates a complete machine learning workflow including:

Raw data processing

Data cleaning

Feature handling

Model training

Model evaluation

Model deployment

The final XGBoost model achieves strong predictive performance and is capable of identifying high-risk applications based on behavioural feature data.

The provided scripts allow the model to be retrained or reused for predicting the risk level of new software applications.
