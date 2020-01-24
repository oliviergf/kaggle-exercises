# Code you have previously used to load data
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from learntools.machine_learning.ex4 import *
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

# //////////////////////////////////////////////////////////////////////////////////////////////////

# Code you have previously used to load data

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF',
                   '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
binder.bind(globals())
print("Setup Complete")


# Import the train_test_split function and uncomment

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Check your answer
step_1.check()


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Check your answer
step_2.check()

val_predictions = iowa_model.predict(val_X)


val_mae = mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)
