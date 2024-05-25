# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 00:48:04 2023

@author: sebas
"""
# Step 0. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#%% Step 1. Import the data, preprocess it, and split it into train and test sets

# Import the data
df = pd.read_csv('train.csv')

# Add all expense columns into a single column
df['expenses'] = df.filter(like='pq').sum(axis=1)
df = df.drop(columns=df.filter(like='pq').columns)


# Select X and y
y = df.iloc[:,0]
X = df.iloc[:,1:]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

#%% Step 2. Using the train set, tune the hyperparameters with k-fold cross validation

# Create model
gradient_boosting = GradientBoostingClassifier()

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2'],
}

# Create k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=85)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(gradient_boosting, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=2)
grid_search.fit(X_train, y_train)

# Get the best model configuration
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Print the best model's parameters and train set accuracy
print("Best Model Parameters:", best_params)
print("Best Train Set Accuracy:", best_accuracy)

#%% Step 3. Use the best params to train a model and use it to predict the test set

# Best Model Parameters: {'learning_rate': 0.1, 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

print('Test set accuracy: ', accuracy)
print(cm_df)

#%% Step 4. Make predictions on (actual) test data
# Import the data
df2 = pd.read_csv('test.csv')

# Add all expense columns into a single column
df2['expenses'] = df2.filter(like='pq').sum(axis=1)
df2 = df2.drop(columns=df2.filter(like='pq').columns)
X_predict = df2

# Retrain the model using the whole observations and the best hyperparameters
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=30, max_features='log2', min_samples_leaf=1, min_samples_split=2)
gb_model.fit(X, y)

#Generate predictions for the actual test set
y_predict = gb_model.predict(X_predict)

#%% Step 5. Create output file

# Concatenate the numbers and convert to a string
concatenated_string = ''.join(map(str, y_predict))

# Save the string to a text file
with open('predictions.txt', 'w') as file:
    file.write(concatenated_string)

