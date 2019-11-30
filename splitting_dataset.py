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


# TODO:Creating figure

prices = df.drop('MEDV', axis=1)
features = df['MEDV']

print(f'Dataset X shape: {prices.shape}')
print(f'Dataset y shape: {features.shape}')

# splitting the dataset into : train and test

x_train, x_test, y_train, y_test = train_test_split(prices, features, test_size  = 0.4 , random_state = 0)
#0.40 means 40%  of the dataset

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# TODO: Print splited the dataset

print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")


