
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression

import seaborn as sns

# Loading dataset
df = pd.read_csv('data/boston/housing.data',
             sep='\s+',
              header=None)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# TODO:Creating figure by separating features(x) and target(y)

prices = df.drop('MEDV', axis=1)
features = df['MEDV']


# TODO:splitting the dataset into : train and test

x_train, x_test, y_train, y_test = train_test_split(prices, features, test_size  = 0.2 , random_state = 0)
#TODO 0.20 means 20%  of the dataset

lm = LinearRegression()
lm.fit(x_train, y_train)

#TODO:output of the training  is a model

print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_,),features}")

Y_pred = lm.predict(x_test)

plt.scatter(y_test, Y_pred)
plt.xlabel("Real Prices: $X_i$")
plt.ylabel("Predicted Prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.savefig('figures/prices_pred_price.png')