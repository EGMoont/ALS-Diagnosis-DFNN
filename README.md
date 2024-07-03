# ALS-Diagnosis-DFNN

Project Overview

This project involves the application of various statistical, mathematical, and machine learning techniques to analyze and classify data from the "Minsk2020_ALS_dataset.csv" dataset. The primary goal is to predict the diagnosis of ALS (Amyotrophic Lateral Sclerosis) using a Deep Feedforward Neural Network (DFNN) and evaluate the model's performance using various metrics.
Libraries and Tools
Statistical and Mathematical Libraries

Pandas: Used for data manipulation and analysis.
NumPy: Used for numerical operations and handling arrays.

Deep Neural Network Libraries

    TensorFlow: A comprehensive open-source platform for machine learning.
    TensorFlow Keras: An API within TensorFlow for building and training neural networks.

Data Handling and Scaling

    Scikit-Learn (sklearn): A machine learning library for data preprocessing, model selection, and evaluation.
        StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
        train_test_split: Splits the dataset into training and testing sets.

Fuzzy Functions and Modules

    scikit-fuzzy (skfuzzy): Used for implementing fuzzy logic, including membership functions.

Metrics and Plotting

    Matplotlib: A plotting library for creating static, interactive, and animated visualizations in Python.
    Scikit-Learn Metrics: Tools for evaluating the performance of machine learning models.
        precision_recall_curve, average_precision_score: Metrics for precision-recall analysis.
        roc_curve, auc: Metrics for ROC analysis.

Workflow

    Load ALS Dataset

    python

    df = pd.read_csv('Minsk2020_ALS_dataset.csv')

    Data Preprocessing
        Drop unnecessary columns (ID, Sex).
        Separate features and target variable.
        Scale numerical features using StandardScaler.

    Data Splitting
        Split the data into training and testing sets using train_test_split.

    Fuzzification of Data
        Fuzzify input data using triangular membership functions from the skfuzzy library.

    Define the DFNN Architecture
        Build a Sequential model with multiple Dense layers and Dropout for regularization.

    Model Compilation and Training
        Compile the model with adam optimizer and binary_crossentropy loss.
        Train the model on fuzzified training data and validate on fuzzified test data.

    Model Evaluation
        Evaluate the model on the test set to obtain loss and accuracy.
        Generate a classification report to assess precision, recall, F1-score, etc.

    ROC and Precision-Recall Curves
        Compute and plot ROC curve.
        Compute and plot Precision-Recall curve.

Usage

To run this project, ensure you have the required libraries installed. You can install them using pip:

bash

pip install pandas numpy tensorflow scikit-learn scikit-fuzzy matplotlib

Execute the provided script to preprocess the data, train the neural network, and evaluate its performance. The script includes the steps for loading the data, preprocessing, fuzzification, model definition, training, evaluation, and plotting the ROC and Precision-Recall curves.
Important Notes

    Adjust the fuzzification method and parameters as needed for different datasets or specific requirements.
    Ensure the dataset path is correct when loading the CSV file.
    Modify the neural network architecture, training epochs, and batch size based on the complexity of the problem and dataset size.
