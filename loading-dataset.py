import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.linear_model import LinearRegression

import seaborn as sns

# TODO:Loading dataset
df = pd.read_csv('data/boston/housing.data',
             sep='\s+',
              header=None)

# Setting columns to dataset
lab_CRIM = 'Per capita crime rate by town'
lab_ZN = 'Proportion of residential land zoned for lots over 25,000 sq.ft.'
lab_INDUS = 'Proportion of non-retail business acres per town'
lab_CHAS = 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)'
lab_NOX = ' nitric oxides concentration (parts per 10 million)'
lab_RM = 'Average number of rooms per dwelling'
lab_AGE = 'Proportion of owner-occupied units built prior to 1940'
lab_DIS = 'Weighted distances to five Boston employment centres'
lab_RAD = 'Index of accessibility to radial highways'
lab_TAX = 'Full-value property-tax rate per $10,000'
lab_PTRATIO = 'Pupil-teacher ratio by town'
lab_B = '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town'
lab_LSTAT = ' % lower status of the population'
lab_MEDV = 'Median value of owner-occupied homes in $1000 '


df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# TODO:Creating figure by separating features(x) and target(y)

X = df.drop('MEDV', axis=1)
y = df['MEDV']

prices = df['MEDV']
features = df.drop('MEDV', axis = 1)

print(f'Dataset X shape: {X.shape}')
print(f'Dataset y shape: {y.shape}')

# TODO: Minimum price of the data
minimum_price = np.mean(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(maximum_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

 # TODO:Example 2 - Loading data using sklearn dataset
from sklearn.datasets import load_boston
boston_house = load_boston()

# features(X) and target(y) are already separated
X = boston_house.data
y = boston_house.target

print(f'sklearn dataset X shape: {X.shape}')
print(f'sklearn dataset y shape: {y.shape}')

# What's inside this sklearn loaded dataset
print(f'keys: {boston_house.keys()}')
print(f'data: {boston_house.data}')
print(f'target: {boston_house.target}')
print(f'feature_names: {boston_house.feature_names}')


# c Rebuilding pandas DF from dataset (for plotting and statistical facts)
convert_to_df = pd.DataFrame(data=np.c[boston_house.data, boston_house.target], columns=boston_house.feature_names + ['target'])
print(convert_to_df.describe())




