# final-project

Research Question

What are the environmental factors that influence the price of houses in suburbs of Boston, Massachusetts?

Introduction

Boston Dataset has been created based on the information collected by the U.S. Census Service concerning housing in the area of Boston Massachusetts.
The Boston Housing Dataset consists of price of houses in various places in Boston and the median value of house price in provided by MEDV ($1000) and we have a total of 14 attributes.
Alongside with price, the Boston dataset also provide information such as Crime (CRIM), portion of residential land zoned for lots over 25000 SQ/FT (ZN), areas of non-retail business in the town (INDUS), the age of people who own the house (AGE), the nitric oxides concentration (NOX), Charles River dummy variable (CHAS), average number of rooms per residence (RM), the walking distance to employments centers (DIS) ,index of accessibility to highways(RAD) ,full value of property-tax per $10000 (TAX), pupil-teacher ratio per town (PT)  and the percent of lower status of population(LSTAT) 

Challenge

The big challenge of Boston area is creating neighborhood solutions to coastal flooding from sea level rise and storms
City of Boston is preparing for climate change by creating The Vision Plan-first step in a long-term effort to increase recreational opportunities and respond to coastal flooding risks.

Objective

This report is looking to determine the influence of several neighborhood attributes on the prices of Boston housing. The specific dataset attributes to be considered are proximity to the Charles River and air pollution using nitrogen oxide concentrations as an explanatory variable for the median value of houses in Boston.
Our research is to create a working model that will have the capability to predict the house’s price using the above mentioned features: NOX and CHAS.

Data exploration

Exploratory Data Analysis is a very important step before training the model. MEDV it is the target and we will plot the distribution of the target to understand the connection with the features.
![Matrix](./figures/price_distribution.png)

Next we are creating a correlation matrix that measures the linear relationships between the variables. The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation
![Matrix](./figures/correlation_matrix.png)

After exploring the data our research cannot use NOX (-0.43) or INDUS (-0.48) because both of them have high negative correlation with MEDV.
To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV. 
An important point in selecting features for a linear regression model is to check for multi-co-linearity. 
Based on the analise of the correlation matrix we will use RM and LSTAT as our features. 


Methods

We are going to use a linear regression model with MEDV as the dependent variable and all the others variable as independent variables.
The data will be split into training and testing sets and we train the model with 40% of the data and test with the remaining 60%. 


------------------------------------------------------------------------------------
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import svm
from sklearn.linear_model import LinearRegression

import seaborn as sns


# Loading dataset
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

x_train, x_test, y_train, y_test = train_test_split(prices, features, test_size  = 0.2 , random_state = 0)
#0.40 means 40%  of the dataset

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Print splited the dataset

print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

# -Splitting features and target datasets into: train and test

x_train, x_test, y_train, y_test = train_test_split(prices, features, test_size  = 0.40 , random_state = 0)
#0.40 means 40%  of the dataset

lm = LinearRegression()
lm.fit(x_train, y_train)

# Predicting the results for our test dataset
predicted_values = lm.predict(x_test)

#  - Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
     print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

#  Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lm.score(x_test, y_test):.2f}/1 \n')
#
from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error(mean squared error): {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error(root-mean-square error): {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
print(f"R2 score : {metrics.r2_score(y_test, predicted_values)}")
------------------------------------------------------------------------------------------------------

Printing MAE error(avg abs residual): 3.633127374024617
Printing MSE error(mean squared error): 25.79036215070245
Printing RMSE error(root-mean-square error): 5.078421226198399
R2 score : 0.6882607142538019
R2 score : 0.6882607142538019
-----------------------------------------------------------------------------------------------------
To measure the model’s performance we will calculate the coefficient of determination R2.


![Matrix](./figures/prices_pred_price.png)

Conclusion
Our project been looking to find what are the environmental factors that influence the price of houses in Boston but we cannot answer.
The created model could provide a prediction of prices but we did not have any feature that could predict how the prices will affected by the global warming.


References

https://www.boston.gov/environment-and-energy/preparing-climate-change

https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

https://scikit-learn.org
