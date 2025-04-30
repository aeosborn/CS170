# Random Forest Model for Ethereum Price Classification

## Overview

This script trains a Random Forest classifier to categorize Ethereum prices into three classes: **Low**, **Medium**, and **High**. Classification is based on historical prices and Ethereum blockchain activity from transactions and validators.

---

## Files Explained

- `main.py`  
  The main script: loads, cleans, and merges the datasets, adds lag features, labels prices, trains a Random Forest classifier, and outputs performance metrics.

- `random_classifier.ipynb`  
  A Jupyter notebook version of the pipeline for interactive use.

- `requirements.txt`  
  List of required Python libraries.

---

## What the Code Does

1. Imports libraries 
   Loads standard ML and visualization libraries: pandas, NumPy, scikit-learn, seaborn, matplotlib.

2. Loads datasets 
   Reads in the three CSV files mentioned above.

3. Cleans and preprocesses 
   - Converts date fields to datetime format.  
   - Ensures numeric fields (price, gas, value) are clean.  
   - Drops rows with missing or malformed data.

4. Aggregates blockchain data
   - Transactions and validators are grouped by day.  
   - Daily sum (for value) and mean (for gas used) are calculated.

5. Merges datasets  
   Ethereum prices are merged with the aggregated transaction and validator metrics based on date.

6. Labels the target variable 
   ETH closing prices are split into Low (bottom 33%), Medium (middle 33%), and High (top 33%) using quantiles.

7. Adds lag features 
   Adds closing price from previous 1 and 2 days as new features to give the model context.

8. Encodes and splits data  
   Encodes labels (Low, Medium, High) as integers and splits the data into training and testing sets.

9. Trains the Random Forest model 
   A RandomForestClassifier with balanced class weights and tuned hyperparameters is trained.

10. Evaluates performance  
   Prints classification report (precision, recall, f1) and confusion matrix.

11. Plots results 
   Generates a heatmap of the confusion matrix for visual inspection.