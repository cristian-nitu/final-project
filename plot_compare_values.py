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


# TODO:Loading dataset
df = pd.read_csv('data/boston/housing.data',
             sep='\s+',
              header=None)


df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = df.columns
#print(df.keys())

sns.set(rc={'figure.figsize':(11.7,9.7)})
sns.distplot(df['MEDV'], bins=30)
plt.savefig('figures/price_distribution')

correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig('figures/correlation_matrix')



fig, axes = plt.subplots(1, 1, figsize=(5, 5 ))
axes.scatter(df['MEDV'], df['NOX'], s=10, label='Noxes', color='black', marker='^')
axes.set_title('Polution vs Value')
axes.set_xlabel('Value')
axes.set_ylabel('Polution')
axes.legend()

plt.savefig('figures/boston_nox_value_scatter.png', dpi=300)

fig, axes = plt.subplots(1, 1, figsize=(5, 5 ))
axes.scatter(df['CHAS'], df['NOX'], s=10, label='RIVER', color='blUE', marker='^')
axes.set_title('River view vs Value')
axes.set_xlabel('Value')
axes.set_ylabel('River view')
axes.legend()

plt.savefig('figures/boston_river_value_scatter.png', dpi=300)


plt.close()
#
