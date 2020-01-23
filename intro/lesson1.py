# Code you have previously used to load data
from learntools.machine_learning.ex3 import *
from learntools.core import binder
from sklearn.tree import DecisionTreeRegressor

import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
binder.bind(globals())

print("Setup Complete")

# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)

y = home_data.SalePrice

# Check your answer
step_1.check()

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF',
                 '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Check your answer
step_2.check()


# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())


# specify the model.
# For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit model
iowa_model.fit(X, y)

# Check your answer
step_3.check()

predictions = iowa_model.predict(X)
print(predictions)

# Check your answer
step_4.check()

# You can write code in this cell
print(home_data.head())
